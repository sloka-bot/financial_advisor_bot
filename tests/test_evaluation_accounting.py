"""Regression checks for delayed execution and bounded sequence inference."""

import numpy as np
import pandas as pd
import pytest

from backend.evaluation.experiments import trading_backtest
from backend.prediction import lstm_model as lm


def test_trading_charges_entry_and_terminal_exit():
    panel = pd.DataFrame({"ticker": ["A"], "trade_return": [0.1]}, index=pd.to_datetime(["2023-01-03"]))
    result = trading_backtest(panel, lambda row: 1, 1, capital=1000, tx_cost=0.01)
    assert result["final_value"] == round(1000 * 0.99 * 1.1 * 0.99, 2)
    assert result["total_cost_pct"] == 2
    assert result["trading_protocol"] == "next_close_v2"


def test_missing_benchmark_is_unavailable_not_flat():
    panel = pd.DataFrame(
        {"ticker": ["A", "B", "A"], "trade_return": [0.1, 0.1, 0.1]},
        index=pd.to_datetime(["2023-01-03", "2023-01-03", "2023-01-04"]),
    )
    result = trading_backtest(panel, lambda row: int(row["ticker"] == "A"), 1)
    assert result["buy_hold_return_pct"] is None
    assert result["benchmark_missing_tickers"] == ["B"]


def test_missing_selected_execution_price_cannot_become_zero_return():
    panel = pd.DataFrame(
        {"ticker": ["A", "B"], "trade_return": [np.nan, 0.1]}, index=pd.to_datetime(["2023-01-03", "2023-01-03"])
    )
    result = trading_backtest(panel, lambda row: int(row["ticker"] == "A"), 1)
    assert result["valid"] is False
    assert result["net_return_pct"] is None
    assert result["missing_tickers"] == ["A"]


def test_lazy_sequences_match_dense_and_batched_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(lm, "MODELS_DIR", tmp_path)
    dates = pd.bdate_range("2020-01-01", periods=160)
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(160, len(lm.TRAIN_FEATURES))), index=dates, columns=lm.TRAIN_FEATURES)
    frame["ticker"] = "A"
    frame["target_return"] = rng.normal(0, 0.01, 160)
    model = lm.LSTMForecaster()
    model.hidden = 4
    model.layers = 1
    dense, y, mask = model._build_sequences(frame)
    lazy, lazy_y, lazy_mask = model._build_sequences(frame, lazy=True)
    np.testing.assert_array_equal(y, lazy_y)
    np.testing.assert_array_equal(mask, lazy_mask)
    for i in (0, len(y) // 2, len(y) - 1):
        np.testing.assert_array_equal(dense[i], lazy[i])
    result = model.train(frame, epochs=1, batch_size=16)
    assert "error" not in result
    observations = frame.iloc[[100, 70, 140]]
    actual = model.predict_panel(frame, observations, batch_size=2)
    expected = [model.predict_ticker(frame.loc[:date]) for date in observations.index]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    changed = frame.copy()
    changed.loc[dates[141] :, lm.TRAIN_FEATURES] = 99999
    np.testing.assert_allclose(model.predict_panel(changed, observations), actual, rtol=1e-5, atol=1e-6)
    reloaded = lm.LSTMForecaster()
    np.testing.assert_allclose(reloaded.predict_panel(frame, observations), actual, rtol=1e-5, atol=1e-6)


def test_ppo_observes_previous_close_and_liquidates():
    from backend.config.settings import RL_STATE_FEATURES
    from backend.portfolio.rl_env import PortfolioEnv

    dates = pd.bdate_range("2023-01-01", periods=70)
    frame = pd.DataFrame({c: np.arange(70, dtype=float) for c in RL_STATE_FEATURES}, index=dates)
    frame["daily_return"] = 0.1
    env = PortfolioEnv({"A": frame}, ["A"], start_idx=30, end_idx=31, deterministic_reset=True, tx_cost=0.01)
    obs, _ = env.reset()
    np.testing.assert_allclose(obs[: len(RL_STATE_FEATURES)], frame.iloc[29][RL_STATE_FEATURES].to_numpy())
    _, _, done, _, info = env.step(np.array([1.0, 0.0]))
    assert done
    assert info["portfolio_value"] == pytest.approx(10000 * 0.99 * 1.1 * 0.99)
    env.reset()
    env.step(np.zeros(2))
    assert env.portfolio_value == 10000


def test_prediction_target_and_delayed_trade_are_distinct():
    from backend.evaluation.experiments import all_features, build_panel

    dates = pd.bdate_range("2023-01-01", periods=10)
    frame = pd.DataFrame({"close": [100, 200, 220, 230, 240, 250, 260, 270, 280, 290]}, index=dates)
    panel = build_panel({"A": frame}, 1)
    first = panel.iloc[0]
    assert first["fwd_ret"] == 1
    assert first["trade_return"] == pytest.approx(0.1)
    assert first["execution_date"] == dates[1]
    assert first["exit_date"] == dates[2]
    assert not {"trade_return", "execution_date", "exit_date", "fwd_ret", "label_end"} & set(
        all_features(set(panel.columns))
    )
