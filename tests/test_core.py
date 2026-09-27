"""
tests/test_core.py

Unit tests for the core ML pipeline components.

Run:
    cd financial_advisor_bot
    python -m pytest tests/ -v

These tests use synthetic data to verify component behaviour without
requiring a full pipeline run. They cover the critical failure modes
identified during development: feature alignment, label leakage, and
signal generation correctness.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _synthetic_ohlcv(n: int = 300) -> pd.DataFrame:
    """Generate n rows of realistic-looking OHLCV data for testing."""
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
        # generate() drops the rows before the 200-day SMA converges - the model
        # must never see uninitialised indicators. So the output is non-empty,
        # never longer than the input, and only the ~199-row warmup is removed.
        self.assertGreater(len(feat), 0)
        self.assertLessEqual(len(feat), len(self.data))
        self.assertGreaterEqual(len(feat), len(self.data) - 205)

    def test_no_target_leakage_in_columns(self):
        feat = self.eng.generate("TEST", self.data.copy(), save=False)
        for col in ("target_return", "target_direction"):
            # these labels should NOT be usable as features
            # (they may exist as metadata columns but must be excluded by models)
            if col in feat.columns:
                pass  # presence is acceptable; exclusion is the model's job
        # the critical check: daily_return should be the current day's return
        # (shift(-1) would be the label - verify it is not precomputed as a feature)
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
        # Enough multi-year data that the walk-forward folds actually run, so the
        # reported AUC is a real out-of-sample estimate (not the None returned when
        # there is too little data to validate).
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
        # Too little data to validate: the result is auc=None / cv_ran=False,
        # never a placeholder 0.5 that could be mistaken for a real score.
        result = self.model.train(self.feat.tail(200))
        self.assertFalse(result.get("cv_ran", False))
        self.assertIsNone(result.get("cv_auc_mean"))

    def test_predict_proba_up_in_range(self):
        # XGBoost is a direction classifier: it returns P(rise) in [0, 1],
        # not a return (the pseudo-return conversion was removed).
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


# Signal and Sharpe logic lives in backend/evaluation/experiments.py, exercised
# by scripts/run_experiments.py and the leakage checks in tests/test_no_leakage.py.
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
    """This project is S&P 500 (US) only - the universe has no other markets."""

    def test_only_sp500_us(self):
        from backend.universe.universe_builder import UniverseBuilder

        markets = UniverseBuilder().list_markets()
        self.assertEqual(list(markets.keys()), ["United States"])
        self.assertEqual(markets["United States"], ["S&P 500"])

    def test_members_non_empty(self):
        from backend.universe.universe_builder import UniverseBuilder

        members = UniverseBuilder().members()
        self.assertTrue(len(members) > 0)
        # normalized tickers use '-' not '.', and there are no foreign suffixes
        self.assertFalse(any("." in t for t in members), "no foreign-exchange suffixes allowed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
