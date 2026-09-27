"""Sequence eligibility and matched-sample evaluation checks."""

import numpy as np
import pandas as pd

from scripts.evaluate_deployed import _sequence_eligible, regression_track


def test_sequence_eligibility_uses_past_rows_and_preserves_order():
    dates = pd.bdate_range("2020-01-01", periods=40)
    history = pd.DataFrame({"ticker": ["A"] * 40}, index=dates)
    rows = pd.DataFrame({"ticker": ["A", "A", "B", "A"]}, index=dates[[29, 28, 35, 39]])
    assert _sequence_eligible(history, rows, 30).tolist() == [True, False, False, True]
    assert _sequence_eligible(history.iloc[:30], rows.iloc[:2], 30).tolist() == [True, False]


def test_short_history_rows_do_not_invalidate_matched_evaluation(monkeypatch, tmp_path):
    import xgboost

    from backend.prediction.lstm_model import LSTMForecaster
    from scripts import evaluate_deployed

    dates = pd.bdate_range("2020-01-01", periods=220)

    def frame(ticker, index):
        return pd.DataFrame({"ticker": ticker, "feature": 1.0, "fwd_ret": 0.02}, index=index)

    panel = pd.concat([frame("A", dates), frame("B", dates), frame("NEW", dates[-10:])]).sort_index()

    class Regressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.full(len(x), 0.01)

    monkeypatch.setattr(xgboost, "XGBRegressor", Regressor)
    monkeypatch.setattr(LSTMForecaster, "train", lambda *args, **kwargs: {})

    def predict(self, history, observations, **kwargs):
        assert "NEW" not in set(observations["ticker"])
        return np.full(len(observations), 0.015)

    monkeypatch.setattr(LSTMForecaster, "predict_panel", predict)
    monkeypatch.setattr(evaluate_deployed, "OUT", tmp_path)
    result = regression_track(panel, dates[:160], dates[160:], ["feature"], ["feature"], with_lstm=True)
    assert result["n_test"] == 130
    assert result["lstm_coverage"]["n_excluded_short_history"] == 10
    assert result["models"]["lstm_actual"]["n_test"] == 120
    assert result["lstm_matched_comparison"]["n_test"] == 120
    assert result["lstm_matched_comparison"]["models"]["xgb_regressor"]["mae"] == 0.01
    assert result["lstm_matched_comparison"]["models"]["lstm_actual"]["mae"] == 0.005
