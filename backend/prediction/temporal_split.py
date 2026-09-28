"""Split financial panels by shared trading dates with label-maturity embargoes."""

import numpy as np
import pandas as pd


def to_unique_sorted_dates(dates) -> np.ndarray:
    """Sorted array of unique dates from an index or array-like of dates."""
    return np.sort(pd.unique(pd.to_datetime(np.asarray(dates))))


def date_cutoff(dates, val_frac: float) -> pd.Timestamp:
    """Return the date separating earlier training rows from the final validation fraction."""
    udates = to_unique_sorted_dates(dates)
    split_idx = min(max(int(len(udates) * (1.0 - val_frac)), 1), len(udates) - 1)
    return pd.Timestamp(udates[split_idx])


def date_split_three(udates, es_frac: float = 0.15, cal_frac: float = 0.15, embargo: int = 0):
    """Return disjoint training, early-stopping and calibration dates with embargoes."""
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
