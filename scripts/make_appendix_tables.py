"""Build Markdown appendix tables from the saved result JSONs and model metadata."""

import json
from datetime import datetime
from pathlib import Path

EXP = Path("data/experiments")
OUT = Path("reports/APPENDIX_RESULTS.md")


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def g(d, *ks, default="-"):
    for k in ks:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    lines = []
    A = lines.append
    A("# Appendix - Actual Results\n")
    A(
        f"_Auto-generated {datetime.now().isoformat(timespec='seconds')} from the result JSONs. "
        "Do not edit by hand; re-run `scripts/make_appendix_tables.py` after any retrain._\n"
    )

    # Provenance.
    prov = []
    for f in [
        "deployed_evaluation.json",
        "deployed_evaluation_h5.json",
        "deployed_evaluation_h1.json",
        "experiment_results.json",
        "portfolio_results.json",
    ]:
        d = load(EXP / f)
        prov.append([f, g(d, "generated_at"), g(d, "schema_version")])
    A("## A. Provenance\n")
    A(table(["File", "generated_at", "schema"], prov) + "\n")

    # Table 1: deployed classifier by horizon.
    A("## B. Deployed classifier by horizon (out-of-sample test)\n")
    rows = []
    for h, f in [
        (1, "deployed_evaluation_h1.json"),
        (5, "deployed_evaluation_h5.json"),
        (21, "deployed_evaluation.json"),
    ]:
        d = load(EXP / f)
        if not d:
            continue
        ms = g(d, "classifier_track", "models", default={})
        n_test = g(d, "classifier_track", "n_test")
        for name in ["majority_class", "logistic_technical", "xgb_technical", "xgb_technical_sentiment"]:
            m = ms.get(name, {})
            rows.append(
                [
                    h,
                    name,
                    g(m, "balanced_accuracy"),
                    fmt(g(m, "roc_auc", default=None)) if g(m, "roc_auc", default=None) not in (None, "-") else "-",
                    g(m, "lift_vs_random", "roc_auc_minus_0.5"),
                    n_test,
                ]
            )
    A(table(["Horizon", "Model", "Balanced acc", "ROC-AUC", "ROC-AUC - 0.5", "n_test"], rows) + "\n")
    A(
        "_Balanced accuracy 0.50 = no separation; ROC-AUC near 0.50 = little discrimination. "
        "Note the 1-day model shows slightly more lift than 21-day, consistent with a short-lived news effect._\n"
    )

    # Table 2: calibration and sentiment CI at h21.
    d = load(EXP / "deployed_evaluation.json")
    cal = g(d, "classifier_track", "calibration", default={})
    A("## C. Probability calibration (21-day, held-out block)\n")
    A(
        table(
            ["Metric", "Raw", "Isotonic-calibrated"],
            [
                ["Brier score", g(cal, "raw", "brier"), g(cal, "calibrated", "brier")],
                ["ECE (10-bin)", g(cal, "raw", "ece_10bin"), g(cal, "calibrated", "ece_10bin")],
            ],
        )
        + "\n"
    )
    ci = g(d, "classifier_track", "sentiment_contribution_paired_dir_hit_ci", default={})
    A("## D. Sentiment contribution (paired, date-block bootstrap, 21-day)\n")
    A(
        table(
            ["Metric", "Mean diff", "95% CI low", "95% CI high", "Significant"],
            [["Directional-hit improvement", g(ci, "mean_diff"), g(ci, "lo"), g(ci, "hi"), g(ci, "significant")]],
        )
        + "\n"
    )

    # Table 3: regression track at h21.
    reg = g(d, "regression_track", "models", default={})
    A("## E. Return regression (21-day, matched test rows)\n")
    rrows = [
        [n, g(reg, n, "mae"), g(reg, n, "rmse"), g(reg, n, "dir_acc")]
        for n in ["zero_return", "dev_hist_mean", "xgb_regressor", "lstm_actual"]
    ]
    A(table(["Model", "MAE", "RMSE", "Directional acc"], rrows) + "\n")

    # Table 4: research matrix per horizon.
    e = load(EXP / "experiment_results.json")
    A("## F. Research regressors by horizon (test set: balanced dir-acc, IC, net return)\n")
    mrows = []
    for h in ["1", "5", "21"]:
        hz = g(e, "horizons", h, default={})
        for key, label in [
            ("C_xgb_technical", "Technical"),
            ("D_xgb_sentiment", "Sentiment"),
            ("E_xgb_tech_sent", "Technical+Sentiment"),
        ]:
            blk = hz.get(key, {})
            mrows.append(
                [
                    h,
                    label,
                    g(blk, "prediction_test", "bal_dir_acc"),
                    g(blk, "prediction_test", "ic"),
                    g(blk, "prediction_test", "mae"),
                    g(blk, "trading", "net_return_pct"),
                ]
            )
    A(table(["Horizon", "Features", "Bal dir-acc", "IC", "MAE", "Trading net %"], mrows) + "\n")
    A(
        "_Trading net % is a top-K long strategy on the research regressor; it is not the deployed advisor and "
        "carries the stated benchmarking limitations._\n"
    )

    # Table 5: portfolio comparison.
    p = load(EXP / "portfolio_results.json")
    pc = g(p, "portfolio_comparison", default={})
    A(f"## G. Portfolio comparison (21-day horizon, shared split {g(p, 'shared_split_date')} to {g(p, 'test_end')})\n")
    prows = []
    for key, label in [
        ("equal_weight", "Equal-weight top-K"),
        ("historical_markowitz", "Markowitz"),
        ("historical_markowitz_cost_aware", "Markowitz (cost-aware)"),
        ("historical_markowitz_cost_aware_smart", "Markowitz (smart rebalance)"),
    ]:
        v = pc.get(key, {})
        if v:
            prows.append(
                [
                    label,
                    g(v, "net_return_pct"),
                    g(v, "sharpe"),
                    g(v, "max_drawdown_pct"),
                    g(v, "avg_turnover"),
                    g(v, "n_rebalances"),
                ]
            )
    ppo = g(p, "ppo", "summary", default={})
    prows.append(
        [
            "PPO (mean of seeds)",
            g(ppo, "mean_return_pct"),
            g(ppo, "mean_sharpe"),
            g(ppo, "worst_max_drawdown_pct"),
            "-",
            "-",
        ]
    )
    A(table(["Strategy", "Net return %", "Sharpe", "Max DD %", "Avg turnover", "Rebalances"], prows) + "\n")
    lim = g(p, "limitations", default=[])
    if isinstance(lim, list) and lim:
        A("**Stated limitations:** " + "; ".join(str(x) for x in lim) + "\n")

    # Table 6: model training metadata.
    A("## H. Deployed model training metadata\n")
    mrows = []
    for name, path in [
        ("XGBoost H1", "models/xgboost/h1/training_results.json"),
        ("XGBoost H5", "models/xgboost/h5/training_results.json"),
        ("XGBoost H21 (primary)", "models/xgboost/training_results.json"),
        ("LSTM H1", "models/lstm/h1/training_history.json"),
        ("LSTM H5", "models/lstm/h5/training_history.json"),
        ("LSTM H21 (primary)", "models/lstm/training_history.json"),
    ]:
        d = load(path)
        if not d:
            mrows.append([name, "MISSING", "-", "-", "-"])
            continue
        cfg = g(d, "config", default={}) or {}
        metric = (
            f"cv_auc={g(d, 'cv_auc_mean')}"
            if g(d, "cv_auc_mean", default=None) not in (None, "-")
            else f"best_epoch={g(d, 'best_epoch')}/{g(d, 'total_epochs')}"
        )
        cfgstr = (
            f"ep_max={cfg.get('epochs_max', '-')}, batch={cfg.get('batch_size', '-')}, "
            f"lr={cfg.get('learning_rate', '-')}, pat={cfg.get('patience', '-')}"
            if cfg
            else "fixed XGB hyperparams"
        )
        dr = g(d, "date_range", default=["-", "-"])
        mrows.append(
            [
                name,
                g(d, "n_samples", default=g(d, "n_sequences")),
                metric,
                cfgstr,
                f"{dr[0]}..{dr[1]}" if isinstance(dr, list) else "-",
            ]
        )
    A(table(["Model", "n_samples", "metric", "config", "train date range"], mrows) + "\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT} ({len(lines)} blocks)")


if __name__ == "__main__":
    main()
