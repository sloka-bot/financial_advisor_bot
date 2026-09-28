"""Checks for forward labels, excluded target columns and train-only scaling."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from backend.config.settings import PREDICTION_HORIZON


def test_target_is_forward_return():
    """target_return[t] is the future horizon return."""
    close = pd.Series(np.linspace(100, 200, 80))
    tgt = close.shift(-PREDICTION_HORIZON) / close - 1.0
    assert abs(tgt.iloc[0] - (close.iloc[PREDICTION_HORIZON] / close.iloc[0] - 1)) < 1e-9
    assert tgt.iloc[-1] != tgt.iloc[-1]  # last H rows are NaN and dropped in fusion


def test_features_exclude_label_columns():
    """The next-move label is excluded from model features."""
    from backend.prediction.xgboost_model import EXCLUDED, XGBoostForecaster

    for col in ("target_return", "target_direction", "ticker", "close"):
        assert col in EXCLUDED
    df = pd.DataFrame(
        {
            "rsi": [1.0, 2, 3, 4],
            "macd": [0.1, 0.2, 0.3, 0.4],
            "close": [10, 11, 12, 13],
            "target_return": [0.01, -0.01, 0.02, 0.0],
            "target_direction": [1, 0, 1, 0],
            "ticker": ["A"] * 4,
        }
    )
    feats = XGBoostForecaster()._select_features(df)
    assert {"target_direction", "target_return", "close", "ticker"}.isdisjoint(feats)
    assert "rsi" in feats and "macd" in feats


def test_labels_sourced_from_target_direction():
    from backend.prediction.xgboost_model import _make_labels

    df = pd.DataFrame({"target_direction": [1, 0, 1, 0], "close": [10, 11, 12, 13]})
    assert list(_make_labels(df)) == [1.0, 0.0, 1.0, 0.0]


def test_lstm_scaler_fit_on_train_rows_only():
    """The scaler is fitted on training rows only."""
    from backend.prediction.lstm_model import TRAIN_FEATURES, LSTMForecaster

    n = 200
    feats = {c: np.linspace(0, 1, n) for c in TRAIN_FEATURES}  # increasing, so the maximum is in validation
    d = dict(feats)
    d["ticker"] = ["A"] * n
    d["target_return"] = np.zeros(n)
    m = LSTMForecaster()
    X, y, val = m._build_sequences(pd.DataFrame(d), fit_scaler=True, val_frac=0.2)
    assert X is not None and val.any() and (~val).any()
    cut = int(n * 0.8)
    train_max = np.asarray([feats[c][:cut].max() for c in TRAIN_FEATURES])
    global_max = np.asarray([feats[c].max() for c in TRAIN_FEATURES])
    assert np.allclose(m.scaler.data_max_, train_max)  # fit on training rows
    assert np.all(m.scaler.data_max_ <= global_max + 1e-9)
    assert np.any(m.scaler.data_max_ < global_max - 1e-6)  # smaller than an all-rows fit
