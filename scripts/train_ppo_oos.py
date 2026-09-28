"""Compare Markowitz and PPO portfolio allocation on a shared held-out window."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config.settings import (
    PRIMARY_HORIZON,
    RF_ANNUAL,
    RISK_CONSTRAINTS,
    TX_COST,
)
from backend.data.contracts import to_jsonable
from backend.evaluation import experiments as ex
from backend.portfolio.markowitz import MarkowitzOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT = Path("data/experiments")
OUT.mkdir(parents=True, exist_ok=True)
FEATURES_DIR = Path("data/features")

TEST_FRAC = 0.25  # held-out fraction for both methods


def load_master(limit, sp500_only, start, end):
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    files = sorted(FEATURES_DIR.glob("*_master.csv"))
    tickers = [f.stem.replace("_master", "") for f in files]
    if sp500_only:
        # Stop when membership cannot be established.
        from backend.universe.sp500_membership import SP500Membership, normalize

        eligible = set(SP500Membership().eligible_between(start, end))
        tickers = [t for t in tickers if normalize(t) in eligible]
        if not tickers:
            raise ValueError(
                "--sp500-only: no eligible S&P 500 members in the window "
                "(membership history missing?) - refusing to run unrestricted"
            )
    if limit:
        tickers = tickers[:limit]
    data = {}
    for t in tickers:
        try:
            df = pd.read_csv(FEATURES_DIR / f"{t}_master.csv", index_col=0, parse_dates=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()].sort_index()
            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]  # restrict to the window
            if len(df) > 300:
                data[t] = df
        except Exception as exc:
            raise ValueError(f"Cannot read portfolio comparison input {t}: {exc}") from exc
    logger.info(f"Loaded {len(data)} tickers (window {start_ts.date()}..{end_ts.date()})")
    return data


# Portfolio comparison on the held-out window using price history only.
def align_comparison_window(data, horizon, test_frac=TEST_FRAC):
    """Use one common calendar and a whole number of holding periods."""
    common = sorted(set.intersection(*(set(frame.index) for frame in data.values())))
    if len(common) < 300:
        raise ValueError("Fewer than 300 common observations; choose a documented comparison universe")
    split_index = int(len(common) * (1 - test_frac))
    periods = (len(common) - 2 - split_index) // horizon
    if periods < 2:
        raise ValueError("Insufficient complete test holding periods")
    end_index = split_index + 1 + periods * horizon
    dates = pd.DatetimeIndex(common[: end_index + 1])
    aligned = {ticker: frame.reindex(dates).copy() for ticker, frame in data.items()}
    # Recompute returns over the aligned price intervals for all strategies.
    for frame in aligned.values():
        frame["daily_return"] = frame["close"].pct_change()
    return aligned, dates[split_index]


def portfolio_comparison(master_data, horizon, risk_profile, split_date, top_k=10, capital=1000.0, point_in_time=False):
    """Compare portfolio-construction methods on the shared final window using price history only."""
    panel = ex.build_panel(master_data, horizon)
    if panel.empty:
        return {"error": "empty panel"}
    if point_in_time:
        # Keep ticker-date rows inside index membership.
        from backend.universe.universe_builder import UniverseBuilder

        panel = UniverseBuilder().filter_eligible_rows(panel, strict=True)
        if panel.empty:
            return {"error": "empty panel after point-in-time membership filter"}
    test_dates = sorted(d for d in panel.index.unique() if d >= split_date)
    if len(test_dates) < horizon + 2:
        return {"error": "insufficient test window"}
    test_panel = panel.loc[panel.index.isin(test_dates)]
    mkw = MarkowitzOptimizer()

    # Equal-weight buy-and-hold baseline.
    eq = ex.trading_backtest(test_panel, lambda row: 1.0, horizon, top_k=top_k, tx_cost=TX_COST, capital=capital)

    def historical_markowitz(
        cost_aware=False, smart_rebal=False, no_trade_band=0.03, adjust_fraction=0.5, lookback=252
    ):
        dates = np.array(sorted(test_panel.index.unique()))
        complete = test_panel.groupby(level=0)["trade_return"].apply(lambda values: np.isfinite(values).any())
        dates = dates[dates <= complete[complete].index.max()]
        rebal = dates[::horizon]
        equity = capital
        rets, turnovers = [], []
        solver_failures = 0
        prev = {}
        for d in rebal:
            day = test_panel.loc[test_panel.index == d]
            # Trailing realised returns from prices before d.
            hist = {}
            for t in day["ticker"].unique():
                r = master_data[t]["close"].pct_change()
                r = r[r.index < d].tail(lookback)
                if len(r) >= 60:
                    hist[t] = r
            mu_h = {t: float(r.mean()) * horizon for t, r in hist.items()}  # per-horizon expected return
            top = [t for t in sorted(mu_h, key=mu_h.get, reverse=True) if mu_h[t] > 0][:top_k]
            # Tradable universe is the new top picks plus names still held.
            held = [t for t in prev if prev.get(t, 0) > 0 and t in hist]
            universe = list(dict.fromkeys(top + held))
            if not universe:
                liquidation_cost = TX_COST * sum(prev.values())
                equity *= 1 - liquidation_cost
                rets.append(-liquidation_cost)
                turnovers.append(sum(prev.values()))
                prev = {}
                continue
            mu = np.array([mu_h[t] * (252 / horizon) for t in universe])  # annualise for the optimiser
            rmat = pd.DataFrame({t: hist[t] for t in universe})
            cov = mkw.covariance(rmat)
            cur = {t: prev.get(t, 0.0) for t in universe} if cost_aware else None
            opt = mkw.optimize(universe, mu, cov, risk_profile, current_weights=cur)
            if not opt.get("solver_ok", True):
                solver_failures += 1  # reported below
            w = opt["weights"]
            if smart_rebal:
                plan = mkw.plan_rebalance(
                    prev, w, no_trade_band=no_trade_band, adjust_fraction=adjust_fraction, min_trade=0.01
                )
                w = plan["executed_weights"]
            fwd = {
                t: float(day[day["ticker"] == t]["trade_return"].mean()) for t in universe if (day["ticker"] == t).any()
            }
            missing = [
                t for t, weight in w.items() if t != "CASH" and weight > 0 and not np.isfinite(fwd.get(t, np.nan))
            ]
            if missing:
                raise ValueError(f"Missing realised execution returns for {missing} on {d}")
            realised = float(sum(w.get(t, 0) * fwd.get(t, 0.0) for t in universe))
            turnover = sum(abs(w.get(t, 0) - prev.get(t, 0)) for t in set(w) | set(prev) if t != "CASH")
            net = (1 - TX_COST * turnover) * (1 + realised) - 1
            equity *= 1 + net
            rets.append(net)
            turnovers.append(turnover)
            prev = {t: w[t] * (1 + fwd.get(t, 0.0)) / max(1 + realised, 1e-9) for t in w if t != "CASH" and w[t] > 0}
        terminal_cost = TX_COST * sum(prev.values())
        equity *= 1 - terminal_cost
        if rets:
            rets[-1] = (1 + rets[-1]) * (1 - terminal_cost) - 1
        rets = np.array(rets)
        cum = np.r_[1.0, np.cumprod(1 + rets)]
        peak = np.maximum.accumulate(cum)
        mdd = float(((cum - peak) / peak).min()) if len(cum) else 0.0
        return {
            "final_value": round(equity, 2),
            "net_return_pct": round((equity / capital - 1) * 100, 2),
            "buy_hold_return_pct": eq["buy_hold_return_pct"],
            "max_drawdown_pct": round(mdd * 100, 2),
            "sharpe": round(
                float((np.mean(rets) - RF_ANNUAL * horizon / 252) / (np.std(rets) + 1e-9) * np.sqrt(252 / horizon)), 3
            )
            if len(rets) > 1
            else 0.0,
            "avg_turnover": round(float(np.mean(turnovers)), 4) if turnovers else 0.0,
            "total_turnover": round(float(np.sum(turnovers)), 4),
            "n_rebalances": int(len(rets)),
            "solver_failures": int(solver_failures),
            "valid": bool(solver_failures == 0),
        }

    return {
        "risk_profile": risk_profile,
        "test_window": {
            "start": str(test_dates[0].date()),
            "end": str(test_dates[-1].date()),
            "n_dates": len(test_dates),
        },
        "firewall": "portfolio methods consume PRICE HISTORY ONLY - no XGB/LSTM/HMM signals",
        "equal_weight": eq,
        "historical_markowitz": historical_markowitz(cost_aware=False, smart_rebal=False),
        "historical_markowitz_cost_aware": historical_markowitz(cost_aware=True, smart_rebal=False),
        "historical_markowitz_cost_aware_smart": historical_markowitz(cost_aware=True, smart_rebal=True),
    }


# PPO training on market features over the matched window.
def train_ppo(master_data, seeds, timesteps, risk_profile, split_date):
    try:
        from backend.portfolio.rl_agent import RLPortfolioAgent
        from backend.portfolio.rl_env import PortfolioEnv
    except Exception as e:
        return {"error": f"RL modules unavailable: {e}"}

    tickers = [t for t in master_data if len(master_data[t]) > 60]
    results = []
    for seed in range(seeds):
        try:
            # Train PPO on the first split_frac of dates and test on the rest.
            env = PortfolioEnv(master_data, tickers, require_features=True)
            split_index = int(env.dates.get_loc(split_date))
            L = env.episode_len
            env.start_idx = max(env.start_idx, int(L * 0.05))
            env.end_idx = split_index - 1  # training window with random starts
            agent = RLPortfolioAgent()
            _tr = agent.train(env, total_timesteps=timesteps, seed=seed)  # per-run seed
            # Deterministic evaluation over the full held-out window.
            test_env = PortfolioEnv(
                master_data,
                tickers,
                start_idx=split_index + 1,
                end_idx=L - 1,
                require_features=True,
                deterministic_reset=True,
            )
            # Evaluate on profile-constrained weights.
            perf = agent.evaluate(test_env, n_episodes=1, risk_profile=risk_profile)
            seed_dir = OUT / "ppo_seeds" / str(seed)
            seed_dir.mkdir(parents=True, exist_ok=True)
            agent.model.save(str(seed_dir / "policy"))
            perf["artifact"] = str(seed_dir / "policy.zip")
            perf["execution"] = "previous-close features; execute at current close"
            perf["seed"] = seed
            perf["learning_curve"] = _tr.get("learning_curve")
            results.append(perf)
            logger.info(f"  PPO seed {seed}: {perf}")
        except Exception as e:
            logger.warning(f"  PPO seed {seed} failed: {e}")
            results.append({"seed": seed, "error": str(e)})
    ok = [r for r in results if "sharpe" in r]
    summary = {}
    if ok:
        summary = {
            "mean_sharpe": round(float(np.mean([r["sharpe"] for r in ok])), 3),
            "std_sharpe": round(float(np.std([r["sharpe"] for r in ok])), 3),
            "mean_return_pct": round(float(np.mean([r["avg_return"] for r in ok])), 2),
            "worst_max_drawdown_pct": round(float(np.max([r["max_drawdown"] for r in ok])), 2),
            "n_seeds_ok": len(ok),
        }
    return {
        "per_seed": results,
        "summary": summary,
        "firewall": "PPO observes market state only - no XGB/LSTM/HMM signals",
    }


def main():
    from backend.infra.reproducibility import set_seed

    set_seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sp500-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--timesteps", type=int, default=50000)
    ap.add_argument("--risk-profile", default="moderate", choices=list(RISK_CONSTRAINTS))
    ap.add_argument("--no-ppo", action="store_true", help="portfolio comparison only")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2023-12-31")
    args = ap.parse_args()

    data = load_master(args.limit, args.sp500_only, args.start, args.end)
    if not data:
        logger.error("No master data - run fusion first")
        sys.exit(1)

    data, split_date = align_comparison_window(data, PRIMARY_HORIZON)
    logger.info(f"Shared held-out window starts {split_date.date()} (last {TEST_FRAC:.0%})")

    logger.info("Portfolio comparison (price history only)...")
    comp = portfolio_comparison(data, PRIMARY_HORIZON, args.risk_profile, split_date, point_in_time=args.sp500_only)

    out = {
        "schema_version": 3,
        "trading_protocol": "next_close_v2",
        "generated_at": datetime.now().isoformat(),
        "horizon": PRIMARY_HORIZON,
        "risk_profile": args.risk_profile,
        "shared_split_date": str(split_date.date()),
        "test_end": str(next(iter(data.values())).index[-1].date()),
        "limitations": [
            "Common-history universe excludes assets without full overlapping history; report this selection bias.",
            "historical_markowitz first screens to the top-K trailing-positive-return names "
            "(plus retained holdings), so it reflects a return-momentum selection stage, not "
            "mean-variance optimisation over the full eligible universe.",
            "Markowitz rebalances every horizon; PPO can rebalance daily. "
            "Gross buy-and-hold excludes transaction costs.",
            "PPO daily risk metrics and horizon-sampled Markowitz metrics are not directly comparable.",
            "Research uses fractional weights; the live allocator rounds to whole shares.",
            "Equal-weight top-K is unconstrained and is not a risk-profile-matched comparator.",
        ],
        "portfolio_comparison": comp,
    }

    if not args.no_ppo:
        logger.info(f"Training PPO (market state only, {args.seeds} seeds)...")
        out["ppo"] = train_ppo(data, args.seeds, args.timesteps, args.risk_profile, split_date)

    (OUT / "portfolio_results.json").write_text(json.dumps(to_jsonable(out), indent=2, default=str, allow_nan=False))
    logger.info(f"Results -> {OUT / 'portfolio_results.json'}")
    print(json.dumps(comp, indent=2, default=str))


if __name__ == "__main__":
    main()
