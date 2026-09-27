"""Regression tests for the engineering-hardening changes: shared temporal split,
model-artifact fingerprints, the shared transaction-cost and metrics modules,
point-in-time membership filtering, and Markowitz solver/input contracts."""

import numpy as np
import pandas as pd
import pytest

from backend.config.settings import RF_ANNUAL, TRADING_DAYS, TX_COST


# --------------------------------------------------------------------------- #
# temporal_split (#14)
# --------------------------------------------------------------------------- #
def test_date_cutoff_uses_last_fraction_of_unique_dates():
    from backend.prediction.temporal_split import date_cutoff

    dates = pd.to_datetime(pd.date_range("2020-01-01", periods=100, freq="D"))
    # duplicate every date (two tickers) to prove it splits on UNIQUE dates, not rows
    doubled = np.concatenate([dates.values, dates.values])
    cut = date_cutoff(doubled, val_frac=0.2)
    assert cut == pd.Timestamp("2020-03-21")  # unique date at index 80 of 100


def test_date_split_three_blocks_are_disjoint_and_embargoed():
    from backend.prediction.temporal_split import date_split_three

    udates = np.array(sorted(pd.date_range("2015-01-01", periods=200, freq="D")))
    tr, es, cal = date_split_three(udates, embargo=5)
    # disjoint
    assert set(tr).isdisjoint(es) and set(es).isdisjoint(cal) and set(tr).isdisjoint(cal)
    # chronological order: all train < all es < all cal
    assert max(tr) < min(es) < max(es) < min(cal)
    # embargo removed 5 dates before es
    tr0, es0, _ = date_split_three(udates, embargo=0)
    assert len(tr0) - len(tr) == 5


# --------------------------------------------------------------------------- #
# model-artifact fingerprint / compatibility (#5, #7-registry, #15)
# --------------------------------------------------------------------------- #
def test_fingerprint_matches_itself_and_rejects_schema_change():
    from backend.infra.model_registry import artifact_fingerprint, check_artifact_compatibility

    fp = artifact_fingerprint(["a", "b", "c"])
    assert check_artifact_compatibility(fp, ["a", "b", "c"]) == (True, [])
    ok, issues = check_artifact_compatibility(fp, ["a", "b", "X"])
    assert ok is False and any("feature-schema" in i for i in issues)


def test_fingerprint_rejects_pipeline_version_change():
    from backend.infra.model_registry import artifact_fingerprint, check_artifact_compatibility

    fp = artifact_fingerprint(["a", "b"])
    stale = dict(fp, feature_pipeline_version=(fp["feature_pipeline_version"] or 0) + 99)
    ok, issues = check_artifact_compatibility(stale, ["a", "b"])
    assert ok is False and any("feature-pipeline version" in i for i in issues)


def test_fingerprint_major_version_fatal_minor_warns():
    from backend.infra.model_registry import artifact_fingerprint, check_artifact_compatibility

    fp = artifact_fingerprint(["a"])
    cur = fp["package_versions"].get("numpy")
    if cur:
        major = int(str(cur).split(".")[0])
        fp_major = dict(fp)
        fp_major["package_versions"] = dict(fp["package_versions"], numpy=f"{major + 1}.0.0")
        ok, issues = check_artifact_compatibility(fp_major, ["a"])
        assert ok is False and any("numpy major" in i for i in issues)
        fp_minor = dict(fp)
        fp_minor["package_versions"] = dict(fp["package_versions"], numpy=f"{major}.999.0")
        ok2, issues2 = check_artifact_compatibility(fp_minor, ["a"])
        assert ok2 is True and issues2  # non-fatal warning recorded


# --------------------------------------------------------------------------- #
# shared transaction-cost module (#47 / list #2)
# --------------------------------------------------------------------------- #
def test_turnover_and_cost_arrays_and_dicts():
    from backend.portfolio.transaction_costs import transaction_cost, trade_cost, turnover

    assert turnover([0.5, 0.5, 0.0], [0.2, 0.3, 0.5]) == pytest.approx(1.0)
    assert turnover({"A": 0.5, "CASH": 0.5}, {"A": 0.2, "B": 0.3, "CASH": 0.5}) == pytest.approx(0.6)
    assert transaction_cost(0.4) == pytest.approx(TX_COST * 0.4)
    assert transaction_cost(0.4, 0.002) == pytest.approx(0.002 * 0.4)
    assert trade_cost({"A": 1.0}, {"A": 0.0}) == pytest.approx(TX_COST)


# --------------------------------------------------------------------------- #
# shared metrics module (#3 / list #3)
# --------------------------------------------------------------------------- #
def test_metrics_sharpe_and_drawdown():
    from backend.evaluation.metrics import annualized_sharpe, equity_curve, max_drawdown

    rets = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
    ann = TRADING_DAYS / 21
    expected = float((rets.mean() - RF_ANNUAL / ann) / (rets.std() + 1e-9) * np.sqrt(ann))
    assert annualized_sharpe(rets, ann) == pytest.approx(expected)
    assert annualized_sharpe([0.01], ann) == 0.0  # below min_periods
    curve = equity_curve([0.1, -0.5, 0.2])  # a clear drawdown
    assert max_drawdown(curve) < 0.0


# --------------------------------------------------------------------------- #
# accounting reconciliation (#4)
# --------------------------------------------------------------------------- #
def test_reconcile_allocation_identity():
    from backend.portfolio.accounting import reconcile_allocation

    ok, detail = reconcile_allocation(notional=800.0, fees=8.0, cash_remaining=192.0, budget=1000.0, weight_sum_pct=80.0)
    assert ok and detail["identity_residual"] == pytest.approx(0.0)
    bad, _ = reconcile_allocation(notional=900.0, fees=0.0, cash_remaining=200.0, budget=1000.0, weight_sum_pct=90.0)
    assert bad is False  # 900+0+200 != 1000


# --------------------------------------------------------------------------- #
# Markowitz solver failure + input contracts (#27, #7)
# --------------------------------------------------------------------------- #
def _cov(n=3, seed=0):
    x = np.random.default_rng(seed).normal(0, 0.01, (250, n))
    return np.cov(x.T) * TRADING_DAYS


def test_markowitz_solver_failure_is_unavailable_not_fabricated(monkeypatch):
    # Force the solver to report non-convergence on otherwise VALID finite inputs
    # (so the #7 input contract passes) and confirm #27: unavailable, not fabricated.
    from backend.portfolio import markowitz as mk
    from backend.portfolio.markowitz import MarkowitzOptimizer, OptimizationError

    class _Failed:
        success = False
        status = 4
        message = "forced non-convergence"
        nit = 3

        def __init__(self, n):
            self.x = np.full(n, 0.1)

    monkeypatch.setattr(mk.opt, "minimize", lambda fun, x0, **kw: _Failed(len(x0)))
    o = MarkowitzOptimizer()
    tickers = ["A", "B", "C"]
    mu = np.array([0.1, 0.08, 0.12])
    res = o.optimize(tickers, mu, _cov(), "moderate")
    assert res["available"] is False and res["solver_ok"] is False
    with pytest.raises(OptimizationError):
        o.optimize(tickers, mu, _cov(), "moderate", strict=True)


def test_markowitz_input_contracts():
    from backend.portfolio.markowitz import MarkowitzOptimizer

    o = MarkowitzOptimizer()
    with pytest.raises(ValueError, match="shape mismatch"):
        o.optimize(["A", "B"], np.array([0.1, 0.1, 0.1]), _cov(3), "moderate")
    asym = _cov(3)
    asym[0, 1] += 0.5  # break symmetry
    with pytest.raises(ValueError, match="symmetric"):
        o.optimize(["A", "B", "C"], np.array([0.1, 0.1, 0.1]), asym, "moderate")


def test_markowitz_cost_report_uses_actual_rate_not_penalty():
    from backend.portfolio.markowitz import MarkowitzOptimizer

    o = MarkowitzOptimizer()
    res = o.optimize(
        ["A", "B", "C"], np.array([0.1, 0.08, 0.12]), _cov(),
        "moderate", current_weights={"A": 0.3, "B": 0.3, "C": 0.0, "CASH": 0.4}, cost_aversion=5.0,
    )
    turn = res["turnover_from_current"]
    assert res["est_trade_cost_pct"] == pytest.approx(round(TX_COST * turn * 100, 4))
    assert "est_penalty_pct" in res  # penalty surfaced separately when cost_aversion != 1


# --------------------------------------------------------------------------- #
# point-in-time membership filter (#37 / list #? )
# --------------------------------------------------------------------------- #
def test_filter_eligible_rows_keeps_only_in_membership_rows(tmp_path):
    from backend.universe.universe_builder import UniverseBuilder

    ub = UniverseBuilder(data_dir=str(tmp_path))
    # AAA is a member from 2020-01, BBB only from 2020-03
    snap = pd.DataFrame(
        {"members": [frozenset({"AAA"}), frozenset({"AAA", "BBB"})]},
        index=pd.to_datetime(["2020-01-01", "2020-03-01"]),
    )
    ub._membership._snap = snap
    panel = pd.DataFrame(
        {"ticker": ["AAA", "BBB", "AAA", "BBB"], "x": [1, 2, 3, 4]},
        index=pd.to_datetime(["2020-02-01", "2020-02-01", "2020-04-01", "2020-04-01"]),
    )
    out = ub.filter_eligible_rows(panel, strict=False)
    # BBB on 2020-02-01 was NOT yet a member -> dropped; the other three kept
    kept = set(zip(out["ticker"], out.index.strftime("%Y-%m-%d")))
    assert ("BBB", "2020-02-01") not in kept
    assert ("AAA", "2020-02-01") in kept and ("BBB", "2020-04-01") in kept


# --------------------------------------------------------------------------- #
# optimizer-failure propagation (#1) and executable-price contract (#2)
# --------------------------------------------------------------------------- #
def _master(seed, tickers, n=300):
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    out = {}
    for i, t in enumerate(tickers):
        rng = np.random.default_rng(seed + i)
        px = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
        out[t] = pd.DataFrame(
            {"close": px, "exec_close": px, "close_unadj": px, "volume": 1e6,
             "imputed_flag": 0, "suspension_flag": 0},
            index=idx,
        )
    return out


class _FailedSolve:
    success = False
    status = 4
    message = "forced non-convergence"
    nit = 1

    def __init__(self, n):
        self.x = np.full(n, 0.1)


def test_build_portfolio_returns_unavailable_on_solver_failure(monkeypatch):
    from backend.portfolio import allocation, markowitz

    monkeypatch.setattr(markowitz.opt, "minimize", lambda fun, x0, **k: _FailedSolve(len(x0)))
    master = _master(1, ["AAA", "BBB"])
    res = allocation.build_markowitz_portfolio(master, ["AAA", "BBB"], 10000.0, "moderate")
    port = res["portfolio"]
    assert port["available"] is False
    assert port["holdings"] == []  # never fabricated into an all-cash target


def test_suggest_rebalance_makes_no_suggestions_when_optimisation_unavailable(monkeypatch):
    from backend.portfolio import markowitz
    from backend.portfolio.portfolio_manager import PortfolioManager

    monkeypatch.setattr(markowitz.opt, "minimize", lambda fun, x0, **k: _FailedSolve(len(x0)))
    master = _master(2, ["AAA", "BBB"])
    pm = PortfolioManager()
    sugg = pm.suggest_rebalance(
        [{"ticker": "AAA", "shares": 10}, {"ticker": "BBB", "shares": 5}],
        master, risk_profile="moderate", cash=1000.0,
    )
    assert sugg == []  # must NOT recommend reducing every holding


def test_executable_price_columns_are_not_forward_filled():
    from backend.data.cleaner import DataCleaner

    idx = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-06"])  # 2023-01-05 session missing
    df = pd.DataFrame(
        {"open": [10, 11, 12], "high": [10, 11, 12], "low": [10, 11, 12],
         "close": [10.0, 11.0, 12.0], "exec_close": [10.0, 11.0, 12.0],
         "close_unadj": [10.0, 11.0, 12.0], "volume": [100, 100, 100]},
        index=idx,
    )
    cleaned, audit = DataCleaner()._handle_gaps(df.copy(), "AAA")
    imputed = cleaned[cleaned["imputed_flag"] == 1]
    if len(imputed):  # a session was imputed
        # analytical close is carried, but executable prices stay NaN on imputed rows
        assert imputed["close"].notna().all()
        assert imputed["exec_close"].isna().all()
        assert imputed["close_unadj"].isna().all()
