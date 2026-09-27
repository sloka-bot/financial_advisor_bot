"""Checks for matched evaluation windows, matured labels and unavailable evidence."""

import numpy as np
import pandas as pd
import pytest

from backend.evaluation import experiments as ex
from scripts import run_experiments


def test_rule_and_model_trading_use_same_final_window(monkeypatch):
    dates = pd.bdate_range("2020-01-01", periods=320)
    data = {
        ticker: pd.DataFrame(
            {"close": np.linspace(100, 140, len(dates)), "rsi": 30.0, "sent_score": 0.3, "sent_news_count": 1},
            index=dates,
        )
        for ticker in ("A", "B", "C")
    }
    monkeypatch.setattr(run_experiments, "_RUN", {"no_cache": True})
    monkeypatch.setattr(ex, "_fit_predict_xgb", lambda train, target, test: np.full(len(test), 0.01))
    result = run_experiments.run_horizon(data, 5, 2)
    trades = [
        result[key]["trading"]
        for key in ("A_technical_rule", "B_sentiment_rule", "C_xgb_technical", "D_xgb_sentiment", "E_xgb_tech_sent")
    ]
    assert len({trade["n_rebalances"] for trade in trades}) == 1
    assert len({trade["buy_hold_return_pct"] for trade in trades}) == 1
    assert result["C_xgb_technical"]["prediction_test"]["n"] == 78 * 3


def test_zero_news_cannot_be_reported_as_sentiment_evidence(monkeypatch):
    dates = pd.bdate_range("2020-01-01", periods=320)
    data = {"A": pd.DataFrame({"close": np.linspace(100, 140, 320), "rsi": 30, "sent_news_count": 0}, index=dates)}
    monkeypatch.setattr(run_experiments, "_RUN", {"no_cache": True})
    monkeypatch.setattr(ex, "_fit_predict_xgb", lambda train, target, test: np.zeros(len(test)))
    result = run_experiments.run_horizon(data, 5, 2)
    for key in ("B_sentiment_rule", "D_xgb_sentiment", "E_xgb_tech_sent"):
        assert result[key]["unavailable"]
    assert "sentiment_contribution_E_minus_C" not in result


def test_fold_training_excludes_labels_maturing_in_future():
    dates = pd.bdate_range("2020-01-01", periods=400)
    data = {
        ticker: pd.DataFrame({"close": np.linspace(100, 150, 400), "rsi": 50}, index=dates)
        for ticker in ("A", "B", "C")
    }
    panel = ex.build_panel(data, 5)
    panel.loc[panel.index[:30].unique(), "rsi"] = 999
    panel.loc[panel["rsi"] == 999, "label_end"] = dates[-1]
    calls = []

    def model(train, target, test):
        assert not np.any(train == 999)
        calls.append(len(train))
        return np.zeros(len(test))

    ex._oos_predictions(panel, ["rsi"], 5, model)
    assert calls


def test_short_history_and_short_bootstrap_do_not_claim_valid_evidence():
    panel = pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=20))
    with pytest.raises(ValueError, match="Insufficient dates"):
        ex.date_folds(panel, 21)
    result = ex.paired_date_bootstrap({day: [1.0] for day in pd.bdate_range("2020-01-01", periods=20)})
    assert result["lo"] is None and result["hi"] is None
    assert not result.get("significant", False)


def test_constant_predictions_have_unavailable_rank_correlation():
    result = ex.prediction_metrics(np.arange(10) / 100, np.zeros(10))
    assert result["ic"] is None
    assert result["mae"] is not None
