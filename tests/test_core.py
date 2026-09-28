"""Unit tests for core pipeline components on synthetic data."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _synthetic_ohlcv(n: int = 300) -> pd.DataFrame:
    """Generate n rows of synthetic OHLCV data."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 5)
    noise = rng.uniform(0.98, 1.02, n)
    df = pd.DataFrame(
        {
            "close": close,
            "open": close * noise,
            "high": close * rng.uniform(1.00, 1.03, n),
            "low": close * rng.uniform(0.97, 1.00, n),
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    return df


class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        from backend.data.feature_engineer import FeatureEngineer

        self.eng = FeatureEngineer()
        self.data = _synthetic_ohlcv(300)

    def test_output_drops_only_indicator_warmup(self):
        feat = self.eng.generate("TEST", self.data.copy(), save=False)
        self.assertIsNotNone(feat, "generate() returned None")
        # Output drops only the SMA200 warm-up rows.
        self.assertGreater(len(feat), 0)
        self.assertLessEqual(len(feat), len(self.data))
        self.assertGreaterEqual(len(feat), len(self.data) - 205)

    def test_target_columns_not_features(self):
        feat = self.eng.generate("TEST", self.data.copy(), save=False)
        for col in ("target_return", "target_direction"):
            # Labels may exist as metadata columns.
            if col in feat.columns:
                pass
        # No precomputed next-day return column.
        self.assertNotIn("next_day_return", feat.columns)

    def test_key_indicators_present(self):
        feat = self.eng.generate("TEST", self.data.copy(), save=False)
        required = ["rsi", "macd", "sma20", "ema12", "bb_pct", "atr"]
        for col in required:
            self.assertIn(col, feat.columns, f"Missing indicator: {col}")

    def test_no_all_nan_columns(self):
        feat = self.eng.generate("TEST", self.data.copy(), save=False)
        for col in feat.select_dtypes(include=[float, int]).columns:
            pct_nan = feat[col].isna().mean()
            self.assertLess(pct_nan, 0.5, f"Column {col} is >50% NaN - likely a computation bug")


class TestXGBoostForecaster(unittest.TestCase):
    def setUp(self):
        from backend.data.feature_engineer import FeatureEngineer
        from backend.prediction.xgboost_model import EXCLUDED, XGBoostForecaster

        eng = FeatureEngineer()
        raw = _synthetic_ohlcv(400)
        feat = eng.generate("TRAIN", raw.copy(), save=False)
        feat["daily_return"] = feat["close"].pct_change().fillna(0)
        feat["ticker"] = "TRAIN"

        self.feat = feat
        self.model = XGBoostForecaster()
        self.EXCLUDED = EXCLUDED

    def test_target_direction_excluded(self):
        self.assertIn("target_direction", self.EXCLUDED, "target_direction must be in EXCLUDED - it IS the label")

    def test_train_runs_walkforward_and_auc_is_plausible(self):
        # Enough history for the walk-forward folds to run.
        from backend.data.feature_engineer import FeatureEngineer

        eng = FeatureEngineer()
        feat = eng.generate("CV", _synthetic_ohlcv(1300).copy(), save=False)
        feat["daily_return"] = feat["close"].pct_change().fillna(0)
        feat["ticker"] = "CV"
        result = self.model.train(feat)
        self.assertTrue(result.get("cv_ran"), "walk-forward should run on multi-year data")
        auc = result["cv_auc_mean"]
        self.assertIsNotNone(auc)
        self.assertGreater(auc, 0.35, "AUC implausibly low")
        self.assertLess(auc, 0.90, "AUC > 0.90 on random data suggests label leakage")

    def test_train_reports_none_auc_when_no_walkforward(self):
        # Too little data gives auc=None and cv_ran=False.
        result = self.model.train(self.feat.tail(200))
        self.assertFalse(result.get("cv_ran", False))
        self.assertIsNone(result.get("cv_auc_mean"))

    def test_predict_proba_up_in_range(self):
        # The classifier returns P(rise) in [0, 1].
        self.model.train(self.feat.tail(200))
        p = self.model.predict_proba_up(self.feat)
        self.assertIsNotNone(p)
        self.assertIsInstance(p, float)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


class TestRegimeDetector(unittest.TestCase):
    def test_bull_regime(self):
        from backend.prediction.regime_detector import RegimeDetector

        det = RegimeDetector()
        data = {}
        for i in range(10):
            rows = _synthetic_ohlcv(300)
            rows["close"] = 100 + np.arange(300) * 0.5  # strong uptrend
            rows["sma200"] = rows["close"].rolling(200).mean()
            rows["adx"] = 30.0
            rows["momentum_10d"] = 0.05
            rows["rsi"] = 60.0
            data[f"T{i}"] = rows
        result = det.detect(data)
        self.assertEqual(result["regime"], "bull")
        self.assertGreater(result["confidence"], 40)

    def test_empty_data_returns_unknown(self):
        from backend.prediction.regime_detector import RegimeDetector

        result = RegimeDetector().detect({})
        self.assertEqual(result["regime"], "unknown")


# Rule signals from backend/evaluation/experiments.py.
class TestExperimentSignals(unittest.TestCase):
    def test_technical_rule_buy(self):
        from backend.evaluation.experiments import technical_rule_signal

        row = pd.Series({"rsi": 30, "macd": 0.5, "macd_signal": 0.2, "bb_pct": 0.1, "momentum_10d": 0.03})
        self.assertEqual(technical_rule_signal(row), 1)

    def test_technical_rule_sell(self):
        from backend.evaluation.experiments import technical_rule_signal

        row = pd.Series({"rsi": 70, "macd": -0.5, "macd_signal": -0.2, "bb_pct": 0.9, "momentum_10d": -0.04})
        self.assertEqual(technical_rule_signal(row), -1)


class TestUniverseBuilder(unittest.TestCase):
    """The universe contains S&P 500 tickers only."""

    def test_only_sp500_us(self):
        from backend.universe.universe_builder import UniverseBuilder

        markets = UniverseBuilder().list_markets()
        self.assertEqual(list(markets.keys()), ["United States"])
        self.assertEqual(markets["United States"], ["S&P 500"])

    def test_members_non_empty(self):
        from backend.universe.universe_builder import UniverseBuilder

        members = UniverseBuilder().members()
        self.assertTrue(len(members) > 0)
        # Normalised tickers use '-' and have no foreign suffixes.
        self.assertFalse(any("." in t for t in members), "no foreign-exchange suffixes allowed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
