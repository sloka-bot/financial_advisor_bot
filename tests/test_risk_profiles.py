"""Risk-profile constraints, cost-aware optimisation and rebalancing rules."""

import unittest

import numpy as np

from backend.config.settings import RISK_CONSTRAINTS
from backend.portfolio.markowitz import MarkowitzOptimizer, validate_allocation


def _toy_market(n=8, seed=0):
    """Small market with distinct expected returns and a PSD covariance."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n)]
    mu = np.linspace(0.05, 0.25, n)  # annualised expected returns
    A = rng.normal(0, 1, (n, n))
    cov = (A @ A.T) / n * 0.04 + np.eye(n) * 0.02  # symmetric PSD
    prices = {t: 50.0 + 10 * i for i, t in enumerate(tickers)}
    return tickers, mu, cov, prices


class TestRiskProfiles(unittest.TestCase):
    def setUp(self):
        self.mkw = MarkowitzOptimizer()
        self.tickers, self.mu, self.cov, self.prices = _toy_market()

    # Profiles change the optimiser weights.
    def test_profiles_produce_different_allocations(self):
        allocs = {
            p: self.mkw.optimize(self.tickers, self.mu, self.cov, p)["weights"]
            for p in ("conservative", "moderate", "aggressive")
        }
        cash = {p: allocs[p]["CASH"] for p in allocs}
        # Safer profiles hold more cash.
        self.assertGreater(cash["conservative"], cash["aggressive"])
        self.assertGreaterEqual(cash["conservative"] + 1e-9, RISK_CONSTRAINTS["conservative"]["min_cash"])
        # The three weight vectors differ.
        self.assertNotEqual(allocs["conservative"], allocs["aggressive"])

    # Optimiser weights respect per-profile caps.
    def test_optimizer_respects_caps(self):
        for p, cons in RISK_CONSTRAINTS.items():
            w = self.mkw.optimize(self.tickers, self.mu, self.cov, p)["weights"]
            stock_w = {k: v for k, v in w.items() if k != "CASH"}
            self.assertLessEqual(
                max(stock_w.values()), cons["max_weight"] + 1e-6, f"{p}: a position exceeds max_weight"
            )
            self.assertLessEqual(
                sum(stock_w.values()), 1.0 - cons["min_cash"] + 1e-6, f"{p}: invested more than the cash floor allows"
            )

    # Constraints hold after rounding and fees.
    def test_caps_hold_after_rounding_and_fees(self):
        for p in RISK_CONSTRAINTS:
            w = self.mkw.optimize(self.tickers, self.mu, self.cov, p)["weights"]
            alloc = self.mkw.to_shares(w, self.prices, budget=100_000.0, allow_fractional=False)
            report = validate_allocation(alloc, p)
            self.assertTrue(report["ok"], f"{p}: {report['violations']}")
            self.assertTrue(alloc["reconciles"], f"{p}: allocation must reconcile to budget")

    # A small budget stays within the cap.
    def test_small_budget_respects_caps(self):
        w = self.mkw.optimize(self.tickers, self.mu, self.cov, "conservative")["weights"]
        alloc = self.mkw.to_shares(w, self.prices, budget=800.0, allow_fractional=False)
        self.assertTrue(validate_allocation(alloc, "conservative")["ok"])


class TestCostAwareOptimizer(unittest.TestCase):
    def setUp(self):
        self.mkw = MarkowitzOptimizer()
        self.tickers, self.mu, self.cov, _ = _toy_market()

    def test_cost_aware_reduces_turnover_from_current_book(self):
        # Cost-aware re-optimisation moves less than a cost-blind one.
        base = self.mkw.optimize(self.tickers, self.mu, self.cov, "moderate")["weights"]
        current = {t: base.get(t, 0.0) for t in self.tickers}
        mu2 = self.mu + np.random.default_rng(1).normal(0, 0.01, len(self.mu))

        blind = self.mkw.optimize(self.tickers, mu2, self.cov, "moderate")["weights"]
        aware = self.mkw.optimize(self.tickers, mu2, self.cov, "moderate", current_weights=current, cost_aversion=5.0)

        def turn(w):
            return sum(abs(w.get(t, 0.0) - current[t]) for t in self.tickers)

        self.assertIn("turnover_from_current", aware)
        self.assertLessEqual(
            turn(aware["weights"]), turn(blind) + 1e-9, "cost-aware should not trade more than cost-blind"
        )

    def test_cost_aware_backward_compatible(self):
        # Without current_weights the result matches the plain optimiser.
        a = self.mkw.optimize(self.tickers, self.mu, self.cov, "moderate")["weights"]
        b = self.mkw.optimize(self.tickers, self.mu, self.cov, "moderate", current_weights=None)["weights"]
        self.assertEqual(a, b)


class TestSmartRebalance(unittest.TestCase):
    def setUp(self):
        self.mkw = MarkowitzOptimizer()

    def test_no_trade_band_holds_small_gaps(self):
        current = {"A": 0.30, "B": 0.30, "CASH": 0.40}
        target = {"A": 0.32, "B": 0.20, "CASH": 0.48}  # A drifts 2%, B drifts 10%
        plan = self.mkw.plan_rebalance(current, target, no_trade_band=0.03, adjust_fraction=1.0, min_trade=0.005)
        self.assertEqual(plan["actions"]["A"]["action"], "HOLD")  # inside band
        self.assertEqual(plan["actions"]["B"]["action"], "REDUCE")  # outside band
        self.assertEqual(plan["n_trades"], 1)

    def test_partial_adjustment_moves_halfway(self):
        current = {"A": 0.10, "CASH": 0.90}
        target = {"A": 0.50, "CASH": 0.50}
        plan = self.mkw.plan_rebalance(current, target, no_trade_band=0.0, adjust_fraction=0.5, min_trade=0.0)
        # Halfway from 0.10 to 0.50 is 0.30.
        self.assertAlmostEqual(plan["executed_weights"]["A"], 0.30, places=6)

    def test_full_rebalance_when_levers_off(self):
        current = {"A": 0.10, "CASH": 0.90}
        target = {"A": 0.50, "CASH": 0.50}
        plan = self.mkw.plan_rebalance(current, target, no_trade_band=0.0, adjust_fraction=1.0, min_trade=0.0)
        self.assertAlmostEqual(plan["executed_weights"]["A"], 0.50, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
