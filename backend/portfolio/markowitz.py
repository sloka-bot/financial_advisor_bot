"""
markowitz.py

Constrained mean-variance optimisation with a distinct, well-scoped
responsibility: given estimated returns and a covariance matrix,
produce a feasible allocation. Risk preferences change the ACTUAL optimisation
- the position cap, the minimum cash floor and the risk-aversion penalty all
come from `RISK_CONSTRAINTS[profile]` - not just the explanation text.

Objective (maximise):   wᵀμ  −  λ · wᵀΣw
subject to:             0 ≤ wᵢ ≤ max_weight,   Σ wᵢ ≤ 1 − min_cash

The leftover 1 − Σwᵢ is held as cash. That cash floor is what lets a genuinely
low-risk profile be satisfied - an S&P-500 stock-only book cannot always be
low-risk on its own.

Covariance uses Ledoit-Wolf shrinkage (sklearn) when enough history is present,
falling back to a diagonal matrix on short samples (Ledoit & Wolf, 2004).
"""

import logging

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm

from backend.config.settings import RF_ANNUAL, RISK_CONSTRAINTS, STD_EPS, TRADING_DAYS, TX_COST, WEIGHT_TOL
from backend.portfolio.transaction_costs import transaction_cost
from backend.portfolio.accounting import AccountingError, reconcile_allocation

logger = logging.getLogger(__name__)

CONFIDENCE = 0.95


class OptimizationError(RuntimeError):
    """Raised in strict/research mode when the optimiser does not converge, so an
    experiment can never silently report a result built from the initial guess."""


class MarkowitzOptimizer:
    """Solve constrained mean-variance allocations with optional trading costs."""

    def __init__(self, tx_cost: float = TX_COST):
        self.tx_cost = tx_cost
        self.last_covariance_method = None
        self.last_cov_joint_rows = None

    # ------------------------------------------------------------------ #
    def covariance(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Annualised covariance matrix from daily returns (Ledoit-Wolf shrinkage).

        Uses only JOINTLY-observed rows (dropna) for the full matrix; it never
        fills missing returns with 0.0, which would understate variance/covariance
        by injecting fake flat days. If too few jointly-observed rows remain, it
        falls back to a diagonal built from each asset's own observed variance.
        """
        n_assets = returns_df.shape[1]
        joint = returns_df.dropna(how="any")
        self.last_cov_joint_rows = int(len(joint))
        if len(joint) < 5 * n_assets or n_assets < 2:
            var = returns_df.var(axis=0, skipna=True).values  # per-asset, own observations
            cov = np.diag(np.nan_to_num(var))
            self.last_covariance_method = "diagonal_insufficient_history"
        else:
            R = joint.values
            try:
                from sklearn.covariance import LedoitWolf

                cov = LedoitWolf().fit(R).covariance_
                self.last_covariance_method = "ledoit_wolf"
            except Exception as e:
                logger.warning(f"Ledoit-Wolf failed ({e}); using diagonal covariance")
                cov = np.diag(np.nanvar(R, axis=0))
                self.last_covariance_method = "diagonal_ledoit_wolf_failed"
        return cov * TRADING_DAYS

    # ------------------------------------------------------------------ #
    def optimize(
        self, tickers, mu, cov, risk_profile="moderate", current_weights=None, cost_aversion=1.0, strict=False
    ):
        """Return feasible weights (incl. a 'CASH' slot) and portfolio risk stats.

        `mu`  : array-like of ANNUALISED expected returns per ticker.
        `cov` : annualised covariance matrix (n×n) aligned to `tickers`.

        Cost-aware mode: when `current_weights` is
        supplied the objective also charges the proportional cost of MOVING from
        those holdings to the new target, so the optimiser only trades when the
        expected-return gain beats the round-trip cost. This is the standard
        transaction-cost objective

            maximise  wᵀμ − λ·wᵀΣw − γ·Σ|wᵢ − wᵢ_prev|,   γ = tx_cost · cost_aversion

        and it produces a genuine no-trade region around the current book rather
        than churning to a fresh optimum every period. With `current_weights=None`
        the behaviour is identical to the plain optimiser (backward compatible).
        """
        cons = RISK_CONSTRAINTS.get(risk_profile, RISK_CONSTRAINTS["moderate"])
        max_w, min_cash, lam = cons["max_weight"], cons["min_cash"], cons["risk_aversion"]

        n = len(tickers)
        if n == 0:
            return {"weights": {"CASH": 1.0}, "expected_return": 0.0, "volatility": 0.0}
        mu = np.asarray(mu, float)
        cov = np.asarray(cov, float)

        # Input contract: shapes agree, values are finite, covariance is symmetric
        # and numerically positive semidefinite. Malformed inputs must not reach the
        # solver, where they would produce a meaningless "allocation".
        if not (len(tickers) == len(mu) == cov.shape[0] == cov.shape[1]):
            raise ValueError(f"shape mismatch: {len(tickers)} tickers, mu={mu.shape}, cov={cov.shape}")
        if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(cov))):
            raise ValueError("non-finite values in mu or cov")
        if not np.allclose(cov, cov.T, atol=1e-8):
            raise ValueError("covariance matrix is not symmetric")
        min_eig = float(np.linalg.eigvalsh((cov + cov.T) / 2.0).min())
        if min_eig < -1e-8:
            raise ValueError(f"covariance matrix is not positive semidefinite (min eig {min_eig:.2e})")

        invest_cap = max(1e-6, 1.0 - min_cash)  # max total stock weight
        bounds = [(0.0, max_w)] * n

        cost_aware = current_weights is not None
        w_prev = None
        if cost_aware:
            # proportional trading cost on turnover from the current book
            gamma = float(self.tx_cost) * float(cost_aversion)
            w_prev = np.array([float(current_weights.get(t, 0.0)) for t in tickers])

            # Augmented smooth NLP: z = [w (n), s (n)] with s_i ≥ |w_i − w_prev_i|.
            # Minimise −(wᵀμ − λwᵀΣw) + γ·Σ s_i  s.t.  s_i ± (w_i − w_prev_i) ≥ 0,
            # Σw ≤ invest_cap. SLSQP solves this well because everything is smooth.
            def neg_utility(z):
                """Return the minimisation objective for the constrained allocation solver."""
                w = z[:n]
                s = z[n:]
                return -(float(w @ mu) - lam * float(w @ cov @ w)) + gamma * float(s.sum())

            constraints = [
                {"type": "ineq", "fun": lambda z: invest_cap - z[:n].sum()},
                {"type": "ineq", "fun": lambda z: z[n:] - (z[:n] - w_prev)},  # s ≥ w − w_prev
                {"type": "ineq", "fun": lambda z: z[n:] + (z[:n] - w_prev)},  # s ≥ w_prev − w
            ]
            z_bounds = bounds + [(0.0, 1.0)] * n
            z0 = np.concatenate([np.clip(w_prev, 0, max_w), np.zeros(n)])
            res = opt.minimize(
                neg_utility,
                z0,
                method="SLSQP",
                bounds=z_bounds,
                constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 1000},
            )
            solver_label = "cost-aware SLSQP"
            w = res.x[:n]
        else:

            def neg_utility(w):
                """Return the minimisation objective for the constrained allocation solver."""
                ret = float(w @ mu)
                risk = float(w @ cov @ w)
                return -(ret - lam * risk)

            constraints = [{"type": "ineq", "fun": lambda w: invest_cap - w.sum()}]  # Σw ≤ cap
            # feasible, budget-respecting start
            x0 = np.full(n, min(max_w, invest_cap / n))
            res = opt.minimize(
                neg_utility,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 800},
            )
            solver_label = "SLSQP"
            w = res.x

        # Validate solver convergence and allocation constraints.
        # An optimiser failure must never be presented as a successful
        # allocation. Beyond SLSQP's own success flag, confirm the returned
        # weights satisfy the box and budget constraints to a tolerance; a
        # "converged" result that violates them is treated as a failure too.
        con_tol = 1e-4
        feasible = bool(
            np.all(w >= -con_tol)
            and np.all(w <= max_w + con_tol)
            and (float(w.sum()) <= invest_cap + con_tol)
        )
        solver_ok = bool(res.success) and feasible
        solver_status = int(getattr(res, "status", -1))
        solver_message = str(getattr(res, "message", ""))
        solver_iterations = int(getattr(res, "nit", -1))
        if not solver_ok:
            reason = res.message if not res.success else "solution violated allocation constraints"
            if strict:
                raise OptimizationError(f"{solver_label} did not converge: {reason}")
            # Recommendation mode: surface an explicit unavailable state rather
            # than silently returning the initial guess as if it were optimal.
            return {
                "weights": {"CASH": 1.0},
                "available": False,
                "risk_profile": risk_profile,
                "expected_return": 0.0,
                "volatility": 0.0,
                "solver_ok": False,
                "solver_status": solver_status,
                "solver_message": str(reason),
                "solver_iterations": solver_iterations,
                "error": f"{solver_label} failed to produce a feasible allocation",
            }

        w = np.clip(w, 0, max_w)
        if w.sum() > invest_cap and w.sum() > 0:  # project back into the feasible set
            w *= invest_cap / w.sum()

        cash = float(1.0 - w.sum())
        port_ret = float(w @ mu)
        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        sharpe = (port_ret - RF_ANNUAL) / (port_vol + STD_EPS)

        # Parametric 1-day VaR / CVaR as positive LOSS magnitudes, including the
        # mean term so the figures follow the standard VaR/CVaR definition rather
        # than a zero-mean volatility threshold.
        daily_ret = port_ret / TRADING_DAYS
        daily_vol = port_vol / np.sqrt(TRADING_DAYS)
        z = norm.ppf(CONFIDENCE)
        var_1d = float(max(0.0, z * daily_vol - daily_ret))
        cvar_1d = float(max(0.0, daily_vol * norm.pdf(z) / (1 - CONFIDENCE) - daily_ret))

        weights = {t: round(float(wi), 6) for t, wi in zip(tickers, w)}
        weights["CASH"] = round(cash, 6)
        result = {
            "weights": weights,
            "risk_profile": risk_profile,
            "constraints_applied": {"max_weight": max_w, "min_cash": min_cash, "risk_aversion": lam},
            "expected_return": round(port_ret, 6),
            "volatility": round(port_vol, 6),
            "sharpe": round(sharpe, 4),
            "var_95_1day": round(var_1d, 6),
            "cvar_95_1day": round(cvar_1d, 6),
            "available": True,
            "solver_ok": solver_ok,
            "solver_status": solver_status,
            "solver_message": solver_message,
            "solver_iterations": solver_iterations,
        }
        if cost_aware:
            turnover = float(np.abs(w - w_prev).sum())
            result["cost_aware"] = True
            result["turnover_from_current"] = round(turnover, 6)
            result["est_trade_cost_pct"] = round(transaction_cost(turnover, self.tx_cost) * 100, 4)
            if cost_aversion != 1.0:
                # gamma is the optimisation penalty (tx_cost x cost_aversion), not the cost.
                result["est_penalty_pct"] = round(transaction_cost(turnover, gamma) * 100, 4)
        return result

    # ------------------------------------------------------------------ #
    def to_shares(
        self,
        weights: dict,
        prices: dict,
        budget: float,
        allow_fractional: bool = False,
        fee_rate: float = None,
        fee_flat: float = 0.0,
        strict: bool = False,
    ) -> dict:
        """Turn target weights into share quantities under a budget, with fees.

        Budget mainly affects quantities and rounding. Whole-share
        rounding leaves residual cash; fractional shares use the budget exactly.
        Trading fees are charged per buy (a `fee_rate` fraction of notional plus
        an optional `fee_flat` per trade), and the result reconciles exactly:
        notional + fees + cash_remaining == budget.
        """
        fee_rate = self.tx_cost if fee_rate is None else fee_rate
        budget = float(budget)
        holdings = []
        notional = 0.0
        fees = 0.0
        # Convert target weights to shares against a running balance, reserving
        # each trade's fees first and filling the largest weights first so the book
        # never overspends the budget.
        available = budget
        items = sorted(((t, w) for t, w in weights.items() if t != "CASH" and w > 0), key=lambda kv: -kv[1])
        for t, w in items:
            price = float(prices.get(t, 0) or 0)
            if price <= 0 or available - fee_flat <= 0:
                continue
            # spend at most this name's budget share, and never more than remains
            target_spend = min(budget * w, available - fee_flat)
            alloc = target_spend / (1 + fee_rate)  # reserve the proportional fee
            shares = alloc / price if allow_fractional else float(int(alloc / price))
            if shares <= 0:
                continue
            cost = shares * price
            fee = cost * fee_rate + fee_flat
            if cost + fee > available + 1e-9:  # hard guard: never overspend
                continue
            available -= cost + fee
            notional += cost
            fees += fee
            holdings.append(
                {
                    "ticker": t,
                    "target_weight_pct": round(w * 100, 2),
                    "shares": round(shares, 4) if allow_fractional else int(shares),
                    "price": round(price, 2),
                    "cost": round(cost, 2),
                    "fee": round(fee, 2),
                }
            )
        cash_remaining = budget - notional - fees  # == available, and >= 0 by construction
        for h in holdings:  # realised weights after rounding
            h["realised_weight_pct"] = round(h["cost"] / budget * 100, 2) if budget else 0.0
        weight_sum_pct = round(sum(h["realised_weight_pct"] for h in holdings), 2)
        # reconciles: cash must be non-negative and the parts must sum to the budget.
        reconciles, recon_detail = reconcile_allocation(
            notional, fees, cash_remaining, budget, weight_sum_pct
        )
        if not reconciles:
            logger.warning("Allocation ledger did not reconcile: %s", recon_detail)
            if strict:
                raise AccountingError(f"Allocation ledger did not reconcile: {recon_detail}")
        return {
            "holdings": holdings,
            "invested": round(notional, 2),
            "fees": round(fees, 2),
            "cash_remaining": round(cash_remaining, 2),
            "cash_weight_pct": round(cash_remaining / budget * 100, 2) if budget else 0.0,
            "invested_weight_pct": weight_sum_pct,
            "reconciles": bool(reconciles),
            "reconciliation_detail": recon_detail,
            "fractional": allow_fractional,
        }

    # ------------------------------------------------------------------ #
    def rebalance(self, current_weights: dict, target_weights: dict) -> dict:
        """Buy/hold/reduce deltas from current to target weights, with turnover."""
        keys = set(current_weights) | set(target_weights)
        deltas = {}
        turnover = 0.0
        for k in keys:
            if k == "CASH":
                continue
            cur = float(current_weights.get(k, 0.0))
            tgt = float(target_weights.get(k, 0.0))
            d = tgt - cur
            turnover += abs(d)
            deltas[k] = {
                "current_pct": round(cur * 100, 2),
                "target_pct": round(tgt * 100, 2),
                "action": "BUY" if d > 0.005 else ("REDUCE" if d < -0.005 else "HOLD"),
                "delta_pct": round(d * 100, 2),
            }
        return {
            "deltas": deltas,
            "turnover": round(turnover, 4),
            "est_cost_pct": round(transaction_cost(turnover, self.tx_cost) * 100, 4),
        }

    # ------------------------------------------------------------------ #
    def plan_rebalance(
        self,
        current_weights: dict,
        target_weights: dict,
        no_trade_band: float = 0.03,
        min_trade: float = 0.01,
        adjust_fraction: float = 1.0,
    ) -> dict:
        """Rebalancing that decides what to actually trade,
        rather than always snapping to the fresh target.

        Three levers, all off by default reproduce a full rebalance:
          * `no_trade_band`  - if a name's target is within this of its current
            weight, HOLD it (no trade). This is the no-trade region that stops
            tiny, cost-only churn.
          * `adjust_fraction`- move only this fraction of the way toward target
            (partial adjustment / smoothing), e.g. 0.5 halves every step.
          * `min_trade`      - after banding and partial adjustment, drop any
            residual trade smaller than this (not worth the ticket).

        Returns the executed per-name actions, the weights that RESULT after the
        planned trades, and the turnover actually incurred - so a caller can
        compare net-of-cost performance against a fixed-schedule full rebalance.
        """
        keys = sorted(k for k in (set(current_weights) | set(target_weights)) if k != "CASH")

        # 1) tentative executed weight per name after band / partial-step / min-trade
        tentative = {}
        for k in keys:
            cur = float(current_weights.get(k, 0.0))
            tgt = float(target_weights.get(k, 0.0))
            gap = tgt - cur
            if abs(gap) < no_trade_band:  # inside the band -> hold
                trade = 0.0
            else:
                trade = adjust_fraction * gap  # partial step toward target
                if abs(trade) < min_trade:  # too small to bother
                    trade = 0.0
            tentative[k] = max(0.0, cur + trade)

        # 2) Funding: keep the book within 100%. When suppressed sales would leave
        #    the executed buys unfunded, scale the executed book back so stock
        #    weights sum to <= 1 and cash stays >= 0. Trades and turnover are then
        #    measured from the
        #    FINAL funded weights, not the tentative ones.
        stock_total = sum(tentative.values())
        if stock_total > 1.0:
            scale = 1.0 / stock_total
            tentative = {k: v * scale for k, v in tentative.items()}

        # 3) build actions/turnover from the FINAL executed weights vs current
        actions = {}
        executed = {}
        turnover = 0.0
        for k in keys:
            cur = float(current_weights.get(k, 0.0))
            new_w = tentative.get(k, 0.0)
            trade = new_w - cur
            if new_w > WEIGHT_TOL:
                executed[k] = new_w
            turnover += abs(trade)
            actions[k] = {
                "current_pct": round(cur * 100, 2),
                "target_pct": round(float(target_weights.get(k, 0.0)) * 100, 2),
                "executed_pct": round(new_w * 100, 2),
                "trade_pct": round(trade * 100, 2),
                "action": "BUY" if trade > WEIGHT_TOL else ("REDUCE" if trade < -WEIGHT_TOL else "HOLD"),
            }
        executed["CASH"] = round(max(0.0, 1.0 - sum(v for k, v in executed.items())), 6)
        n_trades = sum(1 for a in actions.values() if a["action"] != "HOLD")
        return {
            "actions": actions,
            "executed_weights": executed,
            "turnover": round(turnover, 6),
            "est_cost_pct": round(transaction_cost(turnover, self.tx_cost) * 100, 4),
            "n_trades": n_trades,
            "params": {"no_trade_band": no_trade_band, "min_trade": min_trade, "adjust_fraction": adjust_fraction},
        }


# Risk-profile enforcement check: verify the profile's constraints hold on the
# final, executable allocation (after whole-share rounding and fees).
def validate_allocation(alloc: dict, risk_profile: str, tol: float = 0.005) -> dict:
    """Check an executed allocation against its profile's hard constraints.

    `alloc` is the dict returned by `MarkowitzOptimizer.to_shares` (it carries
    each holding's realised, post-rounding weight and the cash weight). Returns a
    report with an overall `ok` flag and the specific violations, so a test - or
    the running system - can assert the profile is genuinely enforced.
    """
    cons = RISK_CONSTRAINTS.get(risk_profile, RISK_CONSTRAINTS["moderate"])
    max_w, min_cash = cons["max_weight"], cons["min_cash"]
    violations = []
    holdings = alloc.get("holdings", [])

    for h in holdings:
        rw = float(h.get("realised_weight_pct", 0.0)) / 100.0
        if rw > max_w + tol:
            violations.append(f"{h.get('ticker')} realised weight {rw:.3f} exceeds max_weight {max_w:.3f}")

    cash_w = float(alloc.get("cash_weight_pct", 0.0)) / 100.0
    if cash_w < min_cash - tol:
        violations.append(f"cash weight {cash_w:.3f} below min_cash floor {min_cash:.3f}")
    if cash_w < -tol:  # negative cash is never valid
        violations.append(f"negative cash weight {cash_w:.3f} (overspent the budget)")

    invested_w = sum(float(h.get("realised_weight_pct", 0.0)) for h in holdings) / 100.0
    if invested_w > 1.0 + tol:  # can't invest more than the budget
        violations.append(f"invested weight {invested_w:.3f} exceeds 100% of budget")

    if not bool(alloc.get("reconciles", True)):
        violations.append("allocation does not reconcile (cash negative or parts != budget)")

    # An all-cash book is a feasible optimum when expected returns are poor (there
    # is no maximum-cash constraint), so it is a warning, not a constraint violation.
    warnings = []
    if not holdings and float(alloc.get("cash_remaining", 0.0)) > 0 and min_cash < 0.99:
        warnings.append("all-cash allocation: the optimiser chose to invest nothing")

    return {
        "risk_profile": risk_profile,
        "ok": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
        "constraints": {"max_weight": max_w, "min_cash": min_cash},
    }
