"""Gaussian HMM regime detection and heuristic fallback behaviour."""

import unittest

import numpy as np
import pandas as pd

from backend.prediction.regime_detector import RegimeDetector


def _market(daily):
    price = 100 * np.cumprod(1 + daily)
    idx = pd.bdate_range("2013-01-02", periods=len(daily))
    return {"MKT": pd.DataFrame({"close": price}, index=idx)}


class TestHMMRegime(unittest.TestCase):
    def _regime(self, daily):
        r = RegimeDetector().detect(_market(daily))
        self.assertEqual(r["method"], "hmm")
        self.assertTrue(r["metrics"]["converged"])
        return r

    def test_sustained_bull(self):
        d = np.random.default_rng(2).normal(0.0012, 0.007, 800)  # up, low volatility
        self.assertEqual(self._regime(d)["regime"], "bull")

    def test_mild_bull(self):
        d = np.random.default_rng(5).normal(0.0008, 0.008, 800)
        self.assertEqual(self._regime(d)["regime"], "bull")

    def test_sustained_bear(self):
        d = np.random.default_rng(6).normal(-0.0009, 0.013, 800)  # down, elevated volatility
        self.assertEqual(self._regime(d)["regime"], "bear")

    def test_flat_is_sideways(self):
        d = np.random.default_rng(4).normal(0.0, 0.010, 800)
        self.assertEqual(self._regime(d)["regime"], "sideways")

    def test_bull_then_crash_reads_bear(self):
        d = np.concatenate(
            [np.random.default_rng(9).normal(0.0010, 0.007, 550), np.random.default_rng(8).normal(-0.0022, 0.020, 220)]
        )
        self.assertEqual(self._regime(d)["regime"], "bear")

    def test_bear_then_recovery_reads_bull(self):
        d = np.concatenate(
            [np.random.default_rng(9).normal(-0.0018, 0.018, 550), np.random.default_rng(8).normal(0.0016, 0.007, 220)]
        )
        self.assertEqual(self._regime(d)["regime"], "bull")

    def test_confidence_in_range(self):
        d = np.random.default_rng(2).normal(0.0012, 0.007, 800)
        c = self._regime(d)["confidence"]
        self.assertTrue(0 <= c <= 100)


class TestHeuristicFallback(unittest.TestCase):
    def test_too_little_history_falls_back(self):
        tiny = {
            "A": pd.DataFrame(
                {
                    "close": [100, 101, 102],
                    "sma200": [90, 90, 90],
                    "adx": [30, 30, 30],
                    "momentum_10d": [0.02, 0.02, 0.02],
                    "rsi": [60, 60, 60],
                },
                index=pd.bdate_range("2023-01-02", periods=3),
            )
        }
        r = RegimeDetector().detect(tiny)
        self.assertEqual(r["method"], "heuristic")
        self.assertIn(r["regime"], {"bull", "bear", "sideways", "unknown"})

    def test_adx_gates_direction_in_heuristic(self):
        # Strong breadth with low ADX is not directional.
        idx = pd.bdate_range("2023-01-02", periods=5)
        weak = {
            f"T{i}": pd.DataFrame(
                {"close": [120] * 5, "sma200": [100] * 5, "adx": [10] * 5, "momentum_10d": [0.02] * 5, "rsi": [60] * 5},
                index=idx,
            )
            for i in range(5)
        }
        self.assertEqual(RegimeDetector().detect(weak)["regime"], "sideways")

    def test_empty_is_unknown(self):
        self.assertEqual(RegimeDetector().detect({})["regime"], "unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
