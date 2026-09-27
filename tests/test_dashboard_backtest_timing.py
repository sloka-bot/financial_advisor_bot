"""Execution timing and fee accounting for the dashboard backtest."""

import numpy as np
import pandas as pd

from backend.config.settings import TX_COST
from backend.evaluation.backtester import Backtester


def test_dashboard_uses_next_close_and_charges_final_exit(monkeypatch):
    import xgboost

    class Classifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            return self

        def predict_proba(self, x):
            prob = np.linspace(0.7, 0.9, len(x))
            return np.column_stack([1 - prob, prob])

    monkeypatch.setattr(xgboost, "XGBClassifier", Classifier)
    dates = pd.bdate_range("2020-01-01", periods=160)
    prices = 100 + 5 * np.sin(np.arange(160) / 4)
    prices[120] = 50
    frame = pd.DataFrame({"close": prices, "feature": np.arange(160, dtype=float)}, index=dates)
    result = Backtester().run("TEST", frame, capital=1000, horizon=5)
    expected_gross = prices[156] / prices[121]
    assert result["metrics"]["total_return"] == round(expected_gross * (1 - TX_COST) ** 2 - 1, 4)
    assert result["metrics"]["benchmark_total_return"] == round(expected_gross - 1, 4)
    assert result["equity_curve"][0]["date"] == str(dates[121].date())
    assert result["equity_curve"][-1]["date"] == str(dates[156].date())
    assert result["trading_protocol"] == "next_close_v2"
    frame.loc[dates[125], "feature"] = np.nan
    missing = Backtester().run("TEST", frame, capital=1000, horizon=5)
    assert missing["n_missing_signal_periods"] == 1
    assert missing["n_test_periods"] == result["n_test_periods"]
    assert missing["metrics"]["benchmark_total_return"] == result["metrics"]["benchmark_total_return"]
