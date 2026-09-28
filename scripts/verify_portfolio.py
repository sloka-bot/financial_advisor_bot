"""Run automated checks on portfolio outputs and exit non-zero on failure."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config.settings import RISK_CONSTRAINTS
from backend.explain import explainer_modes as em
from backend.portfolio.markowitz import MarkowitzOptimizer

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main():
    np.random.seed(1)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    mu = np.array([0.12, 0.20, 0.08, 0.15, 0.05])
    A = np.random.randn(5, 5)
    cov = (A @ A.T) / 50 + np.diag([0.04] * 5)
    prices = {"AAA": 173.2, "BBB": 88.5, "CCC": 412.9, "DDD": 25.7, "EEE": 60.1}
    budget = 10000.0
    opt = MarkowitzOptimizer()

    for prof in ("conservative", "moderate", "aggressive"):
        cons = RISK_CONSTRAINTS[prof]
        r = opt.optimize(tickers, mu, cov, prof)
        alloc = opt.to_shares(r["weights"], prices, budget, allow_fractional=False, fee_rate=0.001, fee_flat=0.0)

        # 1. Reconciliation.
        recon = abs(alloc["invested"] + alloc["fees"] + alloc["cash_remaining"] - budget)
        check(f"[{prof}] reconcile invested+fees+cash==budget", recon < 0.01, f"residual ${recon:.4f}")

        # 2. No negative holdings.
        neg = [h for h in alloc["holdings"] if h["shares"] < 0]
        check(f"[{prof}] no negative holdings", len(neg) == 0)

        # 3. Position limits after rounding.
        cap = cons["max_weight"] * 100
        over = [h for h in alloc["holdings"] if h["realised_weight_pct"] > cap + 0.5]
        check(f"[{prof}] position cap {cap:.0f}% holds after rounding", len(over) == 0, f"{len(over)} over cap")

        # Minimum cash floor.
        check(
            f"[{prof}] min-cash floor respected",
            alloc["cash_weight_pct"] >= cons["min_cash"] * 100 - 0.5,
            f"cash {alloc['cash_weight_pct']}% ≥ {cons['min_cash'] * 100:.0f}%",
        )

    # 4. Forecast versus realised returns.
    facts = em.make_facts(
        "AAA",
        "BUY",
        horizon_days=21,
        forecast_return_pct=3.2,
        confidence_pct=61,
        current_weight_pct=0.0,
        target_weight_pct=12.0,
        qty_change=5,
        est_cost=866.0,
        price=173.2,
        price_date="2023-12-29",
        news_cutoff="2023-12-29",
        model_version="xgb_v3",
        drivers=[{"name": "momentum_21d", "direction": "+", "value": 0.04}],
        portfolio_reason="Adds a growth name within the 25% cap.",
        realized_backtest_return_pct=1.8,
        reliability={"similar_hist_dir_acc": 0.56},
    )
    tech = em.render(facts, "technical")
    beg = em.render(facts, "beginner")
    dist = (
        "forecast" in beg.lower()
        and "already happened" in beg.lower()
        and "forecast(" in tech
        and "realised(backtest)" in tech
    )
    check("forecast vs realised clearly separated", dist)

    # 5. Unsupported-number check with a negative control.
    ok_true, _ = em.validate_numeric_claims(tech, facts)
    ok_false, bad = em.validate_numeric_claims(tech + " Guaranteed 999.99% return.", facts)
    check("explanation contains no unsupported numbers", ok_true)
    check("unsupported-number check catches a fake number (control)", (not ok_false) and 999.99 in bad)

    # 6. No-change recommendation.
    nc = em.make_facts("BBB", "NO_CHANGE", horizon_days=21, current_weight_pct=10.0, target_weight_pct=10.0)
    ie = em.make_facts("CCC", "INSUFFICIENT_EVIDENCE", horizon_days=21)
    check("NO_CHANGE recommendation renders", bool(em.render(nc, "beginner")) and bool(em.render(nc, "technical")))
    check(
        "INSUFFICIENT_EVIDENCE recommendation renders",
        bool(em.render(ie, "beginner")) and bool(em.render(ie, "technical")),
    )

    # 7. Beginner and technical numeric parity.
    check("beginner & technical modes expose identical numbers", em.explanation_numbers_supported(facts))

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 56)
    print(f"PORTFOLIO VERIFICATION: {len(results) - n_fail}/{len(results)} checks passed")
    print("=" * 56)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
