"""Write the saved result JSONs to a Markdown table and a SHA-256 manifest."""

import argparse
import hashlib
import json
import math
from pathlib import Path

# Evidence files behind the report tables (relative to project root).
EVIDENCE = [
    "data/experiments/experiment_results.json",
    "data/experiments/deployed_evaluation.json",
    "data/experiments/deployed_evaluation_h1.json",
    "data/experiments/deployed_evaluation_h5.json",
    "data/experiments/portfolio_results.json",
    "data/experiments/explanations/shap_provenance.json",
    "models/xgboost/training_results.json",
    "models/xgboost/h1/training_results.json",
    "models/xgboost/h5/training_results.json",
    "models/lstm/training_history.json",
    "models/lstm/h1/training_history.json",
    "models/lstm/h5/training_history.json",
]


def scalar_rows(value, path=""):
    """Yield dotted-path and scalar pairs, skipping long arrays."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from scalar_rows(child, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        if value and all(isinstance(v, (int, float)) for v in value) and len(value) <= 12:
            yield path, ", ".join(str(v) for v in value)
        # Longer or nested lists are omitted.
    else:
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        yield path, value


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def sha256(fp: Path) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path, default=Path("reports/evidence"))
    args = ap.parse_args()
    root = args.root.resolve()
    out = (root / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    md = ["# Saved result tables", "", "Values are transcribed from saved records; no experiments were rerun.", ""]
    manifest = {}
    found = 0

    for rel in EVIDENCE:
        fp = root / rel
        if not fp.exists():
            continue
        found += 1
        manifest[rel] = {"sha256": sha256(fp), "bytes": fp.stat().st_size}
        try:
            data = json.loads(fp.read_text())
        except Exception as e:
            md += [f"## {rel}", f"_Unreadable: {e}_", ""]
            continue
        md += [f"## {rel}", "", "| Field | Value |", "|---|---|"]
        for pth, val in scalar_rows(data):
            md.append(f"| {cell(pth)} | {cell(val)} |")
        md.append("")

    (out / "saved_results.md").write_text("\n".join(md))
    (out / "package-sha256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote {out / 'saved_results.md'} and {out / 'package-sha256.json'} ({found} evidence files hashed)")


if __name__ == "__main__":
    main()
