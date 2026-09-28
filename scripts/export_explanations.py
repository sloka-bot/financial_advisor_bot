"""Save SHAP plots using the trained classifier's exact feature schema."""

import argparse
import json
import pickle
from pathlib import Path


def export(model_dir, features_dir, output, max_rows=500, end="2023-12-31"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import shap

    model_dir, features_dir, output = map(Path, (model_dir, features_dir, output))
    with (model_dir / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    with (model_dir / "scaler.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    with (model_dir / "feat_cols.pkl").open("rb") as handle:
        features = pickle.load(handle)
    rows, labels = [], []
    for path in sorted(features_dir.glob("*_master.csv")):
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        # Explain the latest observation inside the 2010-2023 report window.
        frame = frame.loc[frame.index <= pd.Timestamp(end)]
        if frame.empty or not set(features).issubset(frame.columns):
            continue
        valid = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not valid.empty:
            rows.append(valid.iloc[[-1]])
            labels.append({"ticker": path.stem.removesuffix("_master"), "date": str(valid.index[-1].date())})
        if len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError("No observations match the trained feature schema")
    raw = pd.concat(rows)
    explanation = shap.TreeExplainer(model)(scaler.transform(raw.to_numpy()))
    explanation.feature_names = features
    output.mkdir(parents=True, exist_ok=True)

    def _save(stem):
        # Save each figure as PNG and SVG.
        plt.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight")
        plt.savefig(output / f"{stem}.svg", bbox_inches="tight")
        plt.close()

    # Global bar: mean absolute SHAP value per feature.
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.title("Mean absolute SHAP contribution: latest observations")
    _save("shap_bar")

    # Beeswarm: SHAP value distribution per feature.
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("SHAP summary (beeswarm): feature effect distribution")
    _save("shap_summary")

    # Local waterfall for one observation.
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    plt.title(f"{labels[0]['ticker']}: {labels[0]['date']}")
    _save("shap_waterfall")
    metadata = {
        "model_directory": str(model_dir),
        "n_observations": len(raw),
        "features": features,
        "observations": labels,
        "local_observation": labels[0],
        "saved_plots": [
            "shap_bar.(png|svg)",
            "shap_summary.(png|svg)",
            "shap_waterfall.(png|svg)",
        ],
        "interpretation": (
            "Raw classifier margin (log odds), before isotonic calibration. "
            "Associations, not causal effects or estimated returns."
        ),
        "sampling": (
            "One latest complete observation per ticker. This explanation sample is not a performance evaluation."
        ),
    }
    (output / "shap_provenance.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/xgboost")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--output", default="reports/figures")
    parser.add_argument("--end", default="2023-12-31", help="latest master date to explain (matches the report window)")
    args = parser.parse_args()
    try:
        result = export(args.model_dir, args.features_dir, args.output, end=args.end)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"Explanations are not ready: {exc}\n")
    print(f"Saved explanations for {result['n_observations']} observations to {args.output}")


if __name__ == "__main__":
    main()
