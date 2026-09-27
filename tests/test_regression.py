"""
test_regression.py

One focused test per regression case. Each asserts the intended
behaviour, so it would have failed on the pre-fix archive and passes now. Fast:
no full model training or PPO learning - component-level checks only.
"""

import tempfile
import unittest

import numpy as np
import pandas as pd

from backend.config.settings import RL_STATE_FEATURES
from backend.portfolio.allocation import _returns_frame
from backend.portfolio.markowitz import MarkowitzOptimizer, validate_allocation


# --------------------------------------------------------------------------- #
# Accounting (markowitz.to_shares / validate_allocation)
# --------------------------------------------------------------------------- #
class TestAccounting(unittest.TestCase):
    def setUp(self):
        self.mkw = MarkowitzOptimizer()

    def test_flat_fee_never_negative_cash(self):
        # $100 budget with a $5 flat fee per trade must not overspend.
        a = self.mkw.to_shares({"A": 0.5, "B": 0.5, "CASH": 0.0}, {"A": 40.0, "B": 55.0}, budget=100.0, fee_flat=5.0)
        self.assertGreaterEqual(a["cash_remaining"], -0.01)
        # Negative cash must fail the budget reconciliation check.
        self.assertTrue(a["reconciles"])
        self.assertFalse(a["cash_remaining"] < -0.01 and a["reconciles"])

    def test_weights_over_one_are_capped(self):
        # input weights summing > 1 must not produce a >100% book
        a = self.mkw.to_shares({"A": 0.6, "B": 0.6, "CASH": 0.0}, {"A": 10.0, "B": 10.0}, budget=1000.0)
        self.assertLessEqual(a["invested_weight_pct"], 100.01)
        self.assertGreaterEqual(a["cash_remaining"], -0.01)

    def test_validate_rejects_negative_cash(self):
        bad = {
            "holdings": [{"ticker": "X", "realised_weight_pct": 120.0}],
            "cash_weight_pct": -20.0,
            "cash_remaining": -200.0,
            "reconciles": False,
        }
        self.assertFalse(validate_allocation(bad, "moderate")["ok"])

    def test_validate_treats_empty_book_as_warning_not_violation(self):
        # An all-cash book is a feasible optimum (there is no maximum-cash
        # constraint), so it is a warning rather than a hard-constraint violation.
        empty = self.mkw.to_shares({"CASH": 1.0}, {}, budget=1000.0)
        report = validate_allocation(empty, "aggressive")
        self.assertTrue(report["ok"])
        self.assertTrue(report.get("warnings"))


# --------------------------------------------------------------------------- #
# Smart rebalancing funding (markowitz.plan_rebalance) - the 106% bug
# --------------------------------------------------------------------------- #
class TestRebalanceFunding(unittest.TestCase):
    def test_suppressed_sales_do_not_fund_unfunded_buys(self):
        mkw = MarkowitzOptimizer()
        cur = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        tgt = {"A": 0.23, "B": 0.23, "C": 0.23, "D": 0.25, "E": 0.06}
        plan = mkw.plan_rebalance(cur, tgt, no_trade_band=0.03, min_trade=0.01, adjust_fraction=1.0)
        ew = plan["executed_weights"]
        stock_sum = sum(v for k, v in ew.items() if k != "CASH")
        self.assertLessEqual(stock_sum, 1.0001)  # never > 100% invested
        self.assertGreaterEqual(ew["CASH"], -1e-6)  # cash never negative


# --------------------------------------------------------------------------- #
# Covariance date alignment (allocation._returns_frame)
# --------------------------------------------------------------------------- #
class TestDateAlignment(unittest.TestCase):
    def _mk(self, start, n=300, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range(start, periods=n)
        return pd.DataFrame(
            {"close": 100 + rng.normal(0, 1, n).cumsum(), "volume": rng.integers(1e6, 5e6, n)}, index=idx
        )

    def test_non_overlapping_histories_not_mixed(self):
        md = {"A": self._mk("2020-01-02", seed=1), "B": self._mk("2021-06-01", seed=2)}  # disjoint
        rf = _returns_frame(md, ["A", "B"], lookback=252)
        # must NOT fabricate a 2-column matrix from non-overlapping dates
        self.assertFalse(rf.shape[0] > 0 and set(rf.columns) == {"A", "B"})

    def test_overlapping_histories_aligned_on_dates(self):
        md = {t: self._mk("2022-01-03", seed=i) for i, t in enumerate(["A", "B", "C"])}
        rf = _returns_frame(md, ["A", "B", "C"], lookback=252)
        self.assertIsInstance(rf.index, pd.DatetimeIndex)
        self.assertEqual(rf.shape[1], 3)


# --------------------------------------------------------------------------- #
# PPO environment (backend.portfolio.rl_env)
# --------------------------------------------------------------------------- #
class TestRLEnv(unittest.TestCase):
    def _md(self, starts, n=300):
        out = {}
        for i, (t, s) in enumerate(starts.items()):
            rng = np.random.default_rng(i + 1)
            idx = pd.bdate_range(s, periods=n)
            d = {c: rng.normal(0, 1, n) for c in RL_STATE_FEATURES}
            d["daily_return"] = rng.normal(0.0005, 0.01, n)
            d["close"] = 100 + np.arange(n) * 0.1
            out[t] = pd.DataFrame(d, index=idx)
        return out

    def test_calendar_alignment_and_check_env(self):
        from gymnasium.utils.env_checker import check_env

        from backend.portfolio.rl_env import PortfolioEnv

        env = PortfolioEnv(self._md({"A": "2022-01-03", "B": "2022-03-01", "C": "2022-02-01"}), ["A", "B", "C"])
        check_env(env, skip_render_check=True)
        self.assertGreater(env.episode_len, 60)

    def test_action_earns_next_return_not_current(self):
        from backend.portfolio.rl_env import PortfolioEnv

        md = self._md({"A": "2022-01-03", "B": "2022-01-03"})
        env = PortfolioEnv(md, ["A", "B"])
        env.reset(seed=0)
        t = env.t
        act = np.zeros(env.n_stocks + 1, dtype=np.float32)
        act[0] = 1.0
        r_next = env.ret_arrays["A"][t + 1]
        _, _, _, _, info = env.step(act)
        self.assertAlmostEqual(info["gross_step_return"], float(r_next), places=6)

    def test_weights_drift_after_returns(self):
        from backend.portfolio.rl_env import PortfolioEnv

        env = PortfolioEnv(self._md({"A": "2022-01-03", "B": "2022-01-03"}), ["A", "B"])
        env.reset(seed=0)
        env.ret_arrays["A"] = env.ret_arrays["A"].copy()
        env.ret_arrays["A"][env.t + 1] = 1.0  # A doubles
        act = np.zeros(env.n_stocks + 1, dtype=np.float32)
        act[0] = 0.5
        act[1] = 0.5
        env.step(act)
        self.assertGreater(env.weights[0], 0.5)  # drifted toward A, not stuck at 0.5

    def test_first_reward_is_bounded(self):
        from backend.portfolio.rl_env import PortfolioEnv

        env = PortfolioEnv(self._md({"A": "2022-01-03", "B": "2022-01-03"}), ["A", "B"])
        env.reset(seed=0)
        _, r, _, _, _ = env.step(np.ones(env.n_stocks + 1, dtype=np.float32))
        self.assertLess(abs(r), 1e4)  # no 1e-8-denominator blow-up


# --------------------------------------------------------------------------- #
# Label handling (fusion NaN target; LSTM excludes unlabelled rows)
# --------------------------------------------------------------------------- #
class TestLabels(unittest.TestCase):
    def test_fusion_keeps_missing_future_as_nan(self):
        from backend.data.fusion import FeatureFusion

        tmp = tempfile.mkdtemp()
        n = 60
        idx = pd.bdate_range("2022-01-03", periods=n)
        pd.DataFrame({"close": 100 + np.arange(n) * 0.5}, index=idx).to_csv(f"{tmp}/T.csv")
        ff = FeatureFusion(features_dir=tmp, sentiment_dir=tmp)
        out = ff.fuse_ticker("T", save=False)
        # the last PREDICTION_HORIZON rows have no observed future -> label must be NaN, not 0
        self.assertTrue(out["target_direction"].isna().sum() >= 1)
        self.assertTrue(out["target_direction"].tail(1).isna().all())

    def test_lstm_excludes_unlabelled_rows(self):
        from backend.prediction.lstm_model import TRAIN_FEATURES, LSTMForecaster

        n = 120
        df = pd.DataFrame({c: np.random.default_rng(0).normal(0, 1, n) for c in TRAIN_FEATURES})
        df["target_return"] = np.r_[np.random.default_rng(1).normal(0, 0.02, n - 21), [np.nan] * 21]
        df["ticker"] = "T"
        X, y, _ = LSTMForecaster()._build_sequences(df, fit_scaler=True)
        self.assertIsNotNone(y)
        self.assertFalse(np.isnan(y).any())  # no fabricated target survived


# --------------------------------------------------------------------------- #
# Second review batch: sentiment crash, LSTM alignment, experiment selection,
# approval constraints, ranker logic, drift holding-period, calibration artifact.
# --------------------------------------------------------------------------- #
class TestSentimentSerialisation(unittest.TestCase):
    def test_dedupe_output_is_json_serialisable(self):
        import json as _json

        from backend.news.sentiment_analyzer import SentimentAnalyzer

        a = SentimentAnalyzer(news_dir="/tmp", sentiment_dir="/tmp")
        arts = [
            {"title": "X Corp beats", "published_at": "2022-01-03T13:30:00Z", "content": "c"},
            {"title": "X Corp beats", "published_at": "2022-01-03T13:30:00Z"},
        ]  # duplicate
        out = a._dedupe(arts)
        self.assertEqual(len(out), 1)  # dedup worked
        self.assertIsInstance(out[0]["_date"], str)  # ISO string, not Timestamp
        _json.dumps(out)  # must NOT raise (was TypeError)
        daily = a._aggregate_daily(
            [{**out[0], "sentiment_label": "positive", "sentiment_compound": 0.5, "sentiment_score": 0.9}]
        )
        self.assertIsInstance(daily.index, pd.DatetimeIndex)


class TestLSTMAlignment(unittest.TestCase):
    def test_window_ends_at_its_own_label_row(self):
        from backend.prediction.lstm_model import TRAIN_FEATURES, LSTMForecaster

        n = 60
        idx = pd.bdate_range("2022-01-03", periods=n)
        rng = np.random.default_rng(0)
        data = {c: np.arange(n, dtype=float) + rng.normal(0, 0.01, n) for c in TRAIN_FEATURES}
        df = pd.DataFrame(data, index=idx)
        df["target_return"] = np.arange(n, dtype=float)  # sentinel: label == row index
        df["ticker"] = "T"
        f = LSTMForecaster(seq_len=5)
        X, y, _ = f._build_sequences(df, fit_scaler=True)
        # last sample must be anchored at the FINAL row: its label is that row's
        # target AND its window's last timestep is that row's (scaled) features.
        self.assertEqual(float(y[-1]), float(n - 1))
        expected_last = f.scaler.transform(df[TRAIN_FEATURES].values[[n - 1]])[0]
        np.testing.assert_allclose(X[-1, -1, :], expected_last, rtol=1e-5, atol=1e-6)


class TestExperimentSelection(unittest.TestCase):
    def test_top_k_picks_highest_signal(self):
        from backend.evaluation.experiments import trading_backtest

        idx = pd.to_datetime(["2022-01-03"]).repeat(2)
        panel = pd.DataFrame({"ticker": ["A", "B"], "_x": [1.0, 2.0], "trade_return": [-0.10, 0.20]}, index=idx)
        # Duplicate date indices must preserve B's higher signal and +20% return.
        res = trading_backtest(panel, lambda r: r["_x"], horizon=21, top_k=1, tx_cost=0.0)
        self.assertGreater(res["net_return_pct"], 0.0)
        self.assertAlmostEqual(res["net_return_pct"], 20.0, places=4)


class TestApprovalConstraints(unittest.TestCase):
    def setUp(self):
        from backend.portfolio.portfolio_manager import PortfolioManager

        self.pm = PortfolioManager()

    def test_single_buy_cannot_breach_position_cap(self):
        idx = pd.to_datetime(["2022-01-03"])
        md = {"Z": pd.DataFrame({"close": [900.0], "exec_close": [900.0]}, index=idx)}
        # $1000 total, $900 share, moderate 25% cap -> no room, BUY skipped (never a 90% position)
        res = self.pm.apply_recommendation({"ticker": "Z", "action": "BUY"}, [], 1000.0, md, risk_profile="moderate")
        self.assertIn("error", res)

    def test_buy_within_caps_respects_profile(self):
        idx = pd.to_datetime(["2022-01-03"])
        md = {"Z": pd.DataFrame({"close": [100.0], "exec_close": [100.0]}, index=idx)}
        res = self.pm.apply_recommendation({"ticker": "Z", "action": "BUY"}, [], 10000.0, md, risk_profile="moderate")
        self.assertNotIn("error", res)
        self.assertLessEqual(res["max_position_pct"], 25.5)
        self.assertTrue(res["respects_profile"])


class TestRankerLogic(unittest.TestCase):
    def test_negative_forecast_never_buys(self):
        from backend.prediction.recommender import classify_signal

        self.assertEqual(classify_signal(prob_up=0.10, exp_ret=-0.01), "SELL")
        self.assertEqual(classify_signal(prob_up=0.70, exp_ret=0.02), "BUY")
        self.assertEqual(classify_signal(prob_up=None, exp_ret=None), "HOLD")

    def test_single_stock_score_is_finite(self):
        from backend.prediction.ranker import StockRanker

        idx = pd.to_datetime(["2022-01-03"])
        md = {
            "A": pd.DataFrame(
                {
                    "close": [100.0],
                    "volatility": [0.02],
                    "rsi": [50.0],
                    "momentum_10d": [0.0],
                    "sent_score": [0.0],
                    "sent_news_count": [0],
                    "sent_label": ["neutral"],
                },
                index=idx,
            )
        }
        preds = {"A": {"expected_return": 0.01, "prob_up": 0.6, "horizon": 21}}
        df = StockRanker().rank(preds, md, "moderate")
        self.assertFalse(df["composite_score"].isna().any())


class TestDriftHoldingPeriod(unittest.TestCase):
    def test_outcome_uses_horizon_close_not_latest(self):
        import pathlib
        import tempfile

        import backend.infra.drift_monitor as dm

        tmp = pathlib.Path(tempfile.mkdtemp())
        dm.OUTCOMES_PATH = tmp / "out.json"
        dm.DRIFT_LOG_PATH = tmp / "log.json"
        mon = dm.DriftMonitor()
        n = 60
        idx = pd.bdate_range("2022-01-03", periods=n)
        close = np.concatenate([np.full(22, 100.0), np.linspace(100, 50, n - 22)])  # flat 21d, then crash
        md = {"T": pd.DataFrame({"close": close}, index=idx)}
        mon.log_recommendation("T", "BUY", 0.0, date=str(idx[0].date()), horizon=21)
        mon.evaluate_past_recommendations(md)
        o = mon._load_outcomes()[0]
        self.assertTrue(o["evaluated"])
        self.assertAlmostEqual(o["actual_return"], 0.0, places=1)  # ~0 at the horizon, NOT -50%


class TestCalibrationArtifact(unittest.TestCase):
    def test_save_removes_stale_calibrator(self):
        import pathlib
        import tempfile

        import backend.prediction.xgboost_model as xm

        tmp = pathlib.Path(tempfile.mkdtemp())
        xm.MODELS_DIR = tmp
        f = xm.XGBoostForecaster()
        f.model = {"dummy": 1}  # picklable stand-in
        f._feat_cols = ["a", "b"]
        (tmp / "calibrated.pkl").write_bytes(b"stale")
        f.calibrated_model = None  # this model has no calibrator
        f._save()
        self.assertFalse((tmp / "calibrated.pkl").exists())  # stale artifact removed


class TestUserStore(unittest.TestCase):
    def test_recommendation_ids_are_monotonic(self):
        import pathlib
        import tempfile

        import backend.store.user_store as us

        tmp = pathlib.Path(tempfile.mkdtemp())
        us.STORE = tmp / "users.json"
        store = us.UserStore()
        store.create_profile("u1", {"budget": 1000})
        store.add_recommendations("u1", [{"ticker": "A"}, {"ticker": "B"}])
        first_ids = [r["id"] for r in store.get("u1")["pending_recommendations"]]
        store.add_recommendations("u1", [{"ticker": "C"}])
        new_id = store.get("u1")["pending_recommendations"][0]["id"]
        self.assertEqual(first_ids, [1, 2])
        self.assertGreater(new_id, max(first_ids))  # never restarts at 1


# --------------------------------------------------------------------------- #
# Phase (a): membership, PPO controller returns, paired CI, explanation faithfulness,
# and request-model validation.
# --------------------------------------------------------------------------- #
class TestFaithfulExplanation(unittest.TestCase):
    def test_validate_explanation_catches_field_unit_and_claims(self):
        from backend.explain import explainer_modes as em

        f = em.make_facts("AAPL", "BUY", horizon_days=21, forecast_return_pct=3.2, confidence_pct=61, price=55.0)
        self.assertFalse(em.validate_explanation("Guaranteed 3.2% return over 21 trading days.", f)[0])
        self.assertFalse(em.validate_explanation("The current share price is $61.", f)[0])  # 61=confidence, not price
        self.assertFalse(em.validate_explanation("This investment cannot lose money.", f)[0])
        self.assertFalse(em.validate_explanation("Expect +3.2% over 5 trading days.", f)[0])  # wrong horizon
        self.assertTrue(
            em.validate_explanation("Forecast +3.2% over 21 trading days; confidence 61%. Price $55.", f)[0]
        )

    def test_live_faithfulness_gate(self):
        from backend.explain.explainer import Explainer

        e = Explainer()
        rec = {"current_price": 55.0, "horizon": 21}
        self.assertFalse(e._faithful_ok("We expect a gain tomorrow.", rec)[0])  # 1-day horizon implied
        self.assertFalse(e._faithful_ok("Buy around $61.", rec)[0])  # not the current price
        self.assertTrue(e._faithful_ok("A modest gain over ~21 trading days.", rec)[0])


class TestPairedBootstrap(unittest.TestCase):
    def test_paired_ci_flags_real_improvement(self):
        from backend.evaluation import experiments as ex

        idx = pd.to_datetime(np.repeat(pd.bdate_range("2022-01-03", periods=60), 3))
        n = len(idx)
        rng = np.random.default_rng(0)
        yt = rng.normal(0, 0.02, n)
        model = pd.DataFrame(
            {"y_true": yt, "y_pred": yt + rng.normal(0, 0.004, n), "ticker": ["A", "B", "C"] * 60}, index=idx
        )  # close to truth
        base = pd.DataFrame(
            {"y_true": yt, "y_pred": yt + rng.normal(0, 0.03, n), "ticker": ["A", "B", "C"] * 60}, index=idx
        )  # worse
        ci = ex.paired_metric_ci(model, base, metric="abs_error")
        self.assertGreater(ci["mean_diff"], 0)  # model reduces |error|
        self.assertTrue(ci["significant"])  # CI excludes 0


class TestRequestValidation(unittest.TestCase):
    def test_portfolio_request_rejects_bad_input(self):
        import importlib

        pyd = importlib.import_module("pydantic")
        from backend.api.schemas import PortfolioRequest

        # valid
        r = PortfolioRequest(risk_profile="Aggressive", budget=5000, top_n=5)
        self.assertEqual(r.risk_profile, "aggressive")
        # non-positive budget rejected
        with self.assertRaises(pyd.ValidationError):
            PortfolioRequest(budget=0)
        # unknown profile rejected
        with self.assertRaises(pyd.ValidationError):
            PortfolioRequest(risk_profile="yolo")


class TestDeployedEvalHarness(unittest.TestCase):
    def test_regression_track_reports_baselines_and_model(self):
        import backend.evaluation.experiments as ex
        import scripts.evaluate_deployed as ed

        rng = np.random.default_rng(0)
        tech = ["sma20", "rsi", "macd", "momentum_10d", "volatility", "atr_pct", "bb_pct"]
        sent = ["sent_score", "sent_weighted", "sent_news_count"]
        bidx = pd.bdate_range("2016-01-01", periods=400)
        data = {}
        for t in ("A", "B", "C", "D"):
            close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.015, 400))
            df = pd.DataFrame({c: rng.normal(0, 1, 400) for c in tech}, index=bidx)
            for c in sent:
                df[c] = rng.normal(0, 0.3, 400)
            df["close"] = close
            df["daily_return"] = np.r_[0, np.diff(close) / close[:-1]]
            data[t] = df
        panel = ex.build_panel(data, 21)
        avail = set(panel.columns)
        techf, both = ex.technical_features(avail), ex.all_features(avail, True)
        dev, test = ed._date_split(panel, 21)
        rt = ed.regression_track(panel, dev, test, techf, both, with_lstm=False)
        self.assertIn("zero_return", rt["models"])
        self.assertIn("dev_hist_mean", rt["models"])
        self.assertIn("xgb_regressor", rt["models"])
        for m in rt["models"].values():
            self.assertTrue(np.isfinite(m["mae"]) and m["mae"] >= 0)


class TestRebalanceToTarget(unittest.TestCase):
    def test_reduce_executes_toward_target_not_half(self):
        from backend.portfolio.portfolio_manager import PortfolioManager

        idx = pd.to_datetime(["2022-01-03"])
        md = {"Z": pd.DataFrame({"close": [100.0], "exec_close": [100.0]}, index=idx)}
        pm = PortfolioManager()
        holdings = [{"ticker": "Z", "shares": 30, "price": 100.0, "total_cost": 3000.0}]  # 30% of 10k
        rec = {
            "ticker": "Z",
            "action": "REBALANCE",
            "sub_action": "REDUCE",
            "target_weight": 25.0,
            "current_weight": 30.0,
        }
        res = pm.apply_recommendation(rec, holdings, 7000.0, md, risk_profile="moderate")
        z = next(h for h in res["holdings"] if h["ticker"] == "Z")
        # target 25% of $10k = $2500 = 25 shares: sell 5, NOT half (which would be ~15 shares)
        self.assertAlmostEqual(z["shares"], 25, delta=1)


class TestPPODeterministicEval(unittest.TestCase):
    def _md(self, n=120):
        from backend.config.settings import RL_STATE_FEATURES

        idx = pd.bdate_range("2022-01-03", periods=n)
        out = {}
        for i, t in enumerate(("A", "B")):
            rng = np.random.default_rng(i)
            d = {c: rng.normal(0, 1, n) for c in RL_STATE_FEATURES}
            d["daily_return"] = rng.normal(0, 0.01, n)
            d["close"] = 100 + np.arange(n) * 0.1
            out[t] = pd.DataFrame(d, index=idx)
        return out

    def test_deterministic_reset_is_fixed_start(self):
        from backend.portfolio.rl_env import PortfolioEnv

        env = PortfolioEnv(self._md(), ["A", "B"], deterministic_reset=True)
        env.reset(seed=1)
        t1 = env.t
        env.reset(seed=2)
        t2 = env.t
        self.assertEqual(t1, env.start_idx)
        self.assertEqual(t1, t2)  # same fixed OOS start every episode

    def test_strict_features_raises_on_missing(self):
        from backend.config.settings import RL_STATE_FEATURES
        from backend.portfolio.rl_env import PortfolioEnv

        md = self._md()
        md["A"] = md["A"].drop(columns=[RL_STATE_FEATURES[0]])
        with self.assertRaises(ValueError):
            PortfolioEnv(md, ["A", "B"], strict_features=True)


class TestMarkowitzRisk(unittest.TestCase):
    def test_covariance_no_fake_zeros_and_var_cvar_ordering(self):
        from backend.portfolio.markowitz import MarkowitzOptimizer

        mkw = MarkowitzOptimizer()
        idx = pd.bdate_range("2021-01-01", periods=300)
        rmat = pd.DataFrame(
            {"A": np.random.default_rng(0).normal(0, 0.01, 300), "B": np.random.default_rng(1).normal(0, 0.02, 300)},
            index=idx,
        )
        cov = mkw.covariance(rmat)
        self.assertTrue(np.all(np.diag(cov) > 0))  # real variance, not ~0 from fake fills
        res = mkw.optimize(["A", "B"], np.array([0.10, 0.10]), cov, "moderate")
        self.assertGreaterEqual(res["var_95_1day"], 0.0)
        self.assertGreaterEqual(res["cvar_95_1day"], res["var_95_1day"])  # ES >= VaR


if __name__ == "__main__":
    unittest.main(verbosity=2)
