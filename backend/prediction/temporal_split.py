"""
temporal_split.py

Single source of truth for time-aware data splitting shared by the models.

Financial panel data is ordered by ticker then date, so a split on ROW position
is not chronological - it silently splits by stock, not by time, and leaks the
future. Every model here therefore splits on unique CALENDAR DATE. This module
centralises that logic so the LSTM, the XGBoost classifier and any future model
compute train / validation / calibration boundaries, the leakage embargo and the
label-maturity purge the same way rather than re-deriving them inline.

Definitions used throughout:
  - embargo: number of trailing training dates dropped before a block boundary so
    that no training label (which matures `horizon` sessions later) overlaps the
    next block. `embargo_for(horizon)` in settings supplies the size.
  - purge: the same idea applied per-sequence - a training sequence whose forward
    label would reach into the validation region is dropped.
"""

import numpy as np
import pandas as pd


def to_unique_sorted_dates(dates) -> np.ndarray:
    """Sorted array of unique dates from an index or array-like of dates."""
    return np.sort(pd.unique(pd.to_datetime(np.asarray(dates))))


def date_cutoff(dates, val_frac: float) -> pd.Timestamp:
    """Chronological cutoff timestamp for a two-way split by unique date.

    Everything strictly before the returned timestamp is training; the last
    `val_frac` of unique dates is validation. The cutoff index is clamped to
    [1, n-1] so both sides are always non-empty when at least two dates exist.
    """
    udates = to_unique_sorted_dates(dates)
    split_idx = min(max(int(len(udates) * (1.0 - val_frac)), 1), len(udates) - 1)
    return pd.Timestamp(udates[split_idx])


def date_split_three(udates, es_frac: float = 0.15, cal_frac: float = 0.15, embargo: int = 0):
    """Three disjoint chronological date blocks: (train, early-stopping, calibration).

    The last `cal_frac` of dates is calibration; the last `es_frac` of what
    remains is the early-stopping block; the rest is training. An `embargo` drops
    the trailing training dates (and the trailing early-stopping dates) before
    their following block so no label matures across a boundary. Reproduces the
    block boundaries the XGBoost trainer previously computed inline.
    """
    udates = np.array(sorted(np.unique(np.asarray(udates))))
    n_cal = max(1, int(len(udates) * cal_frac))
    cal_dates = udates[-n_cal:]
    fit_dates = udates[:-n_cal]
    n_es = max(1, int(len(fit_dates) * es_frac))
    es_dates = fit_dates[-n_es:]
    tr_dates = fit_dates[:-n_es]
    if embargo > 0 and len(tr_dates) > embargo:
        tr_dates = tr_dates[:-embargo]
    if embargo:
        es_dates = es_dates[:-embargo]
    return tr_dates, es_dates, cal_dates
