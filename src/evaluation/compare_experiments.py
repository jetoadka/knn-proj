from __future__ import annotations

"""Compare results across downstream experiments using W&B API.

Pulls metrics from all completed runs in the `downstream-face-rec` project,
generates comparison tables, and optionally creates plots.

Usage:
    # Print comparison table for all runs
    python -m src.evaluation.compare_experiments

    # Save comparison as JSON
    python -m src.evaluation.compare_experiments --output results/comparison.json

    # Compare specific runs
    python -m src.evaluation.compare_experiments --runs E1-baseline E2-augmented E3-mixed
"""

import argparse
import json
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None
    print("ERROR: wandb not installed. Run: pip install wandb")

import numpy as np


ENTITY = "knn-proj"
PROJECT = "downstream-face-rec"

METRIC_KEYS = [
    "best_val_accuracy",
    "test/rank1_accuracy",
    "test/rank5_accuracy",
    "test/tar_at_far_1e4",
    "test/kfold_verification_accuracy",
]


def fetch_runs(entity: str, project: str, run_names: list[str] | None = None) -> list[dict]:
    """Fetch completed runs from W&B."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    results = []
    for run in runs:
        if run_names and run.name not in run_names:
            continue

        run_data = {
            "name": run.name,
            "state": run.state,
            "created_at": run.created_at,
        }

        # Get config
        config = run.config
        run_data["backbone"] = config.get("backbone", "?")
        run_data["loss"] = config.get("loss", "?")
        run_data["epochs"] = config.get("epochs", "?")
        run_data["train_dir"] = config.get("train_dir", "?")

        # Get summary metrics
        for key in METRIC_KEYS:
            val = run.summary.get(key)
            run_data[key] = float(val) if val is not None else None

        results.append(run_data)

    return sorted(results, key=lambda r: r["name"])


def print_comparison_table(runs: list[dict]):
    """Print a formatted comparison table."""
    if not runs:
        print("No runs found.")
        return

    print(f"\n{'='*100}")
    print(f"EXPERIMENT COMPARISON — {ENTITY}/{PROJECT}")
    print(f"{'='*100}")

    # Header
    print(f"\n{'Experiment':<25} {'Backbone':<16} {'Loss':<10} "
          f"{'Val Acc':>8} {'Rank-1':>8} {'Rank-5':>8} {'TAR@1e-4':>10} {'KFold':>8}")
    print("-" * 100)

    for run in runs:
        val_acc = run.get("best_val_accuracy")
        r1 = run.get("test/rank1_accuracy")
        r5 = run.get("test/rank5_accuracy")
        tar = run.get("test/tar_at_far_1e4")
        kfold = run.get("test/kfold_verification_accuracy")

        print(f"{run['name']:<25} {run['backbone']:<16} {run['loss']:<10} "
              f"{_fmt(val_acc):>8} {_fmt(r1):>8} {_fmt(r5):>8} {_fmt(tar):>10} {_fmt(kfold):>8}")

    print("-" * 100)

    # Find best for each metric
    for key in METRIC_KEYS:
        values = [(r["name"], r[key]) for r in runs if r.get(key) is not None]
        if values:
            best_name, best_val = max(values, key=lambda x: x[1])
            short_key = key.split("/")[-1]
            print(f"  Best {short_key}: {best_name} ({best_val:.4f})")

    print()


def _fmt(val) -> str:
    """Format a metric value."""
    if val is None:
        return "—"
    return f"{val:.4f}"


def generate_markdown_report(runs: list[dict]) -> str:
    """Generate a Markdown comparison table for documentation."""
    lines = [
        "# Experiment Comparison Results",
        "",
        f"Project: `{ENTITY}/{PROJECT}`",
        "",
        "| Experiment | Backbone | Loss | Val Acc | Rank-1 | Rank-5 | TAR@FAR=1e-4 | 10-fold |",
        "|------------|----------|------|---------|--------|--------|--------------|---------|",
    ]

    for run in runs:
        val_acc = _fmt(run.get("best_val_accuracy"))
        r1 = _fmt(run.get("test/rank1_accuracy"))
        r5 = _fmt(run.get("test/rank5_accuracy"))
        tar = _fmt(run.get("test/tar_at_far_1e4"))
        kfold = _fmt(run.get("test/kfold_verification_accuracy"))

        lines.append(
            f"| {run['name']} | {run['backbone']} | {run['loss']} | "
            f"{val_acc} | {r1} | {r5} | {tar} | {kfold} |"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        "| Metric | Best Experiment | Value |",
        "|--------|----------------|-------|",
    ])

    for key in METRIC_KEYS:
        values = [(r["name"], r[key]) for r in runs if r.get(key) is not None]
        if values:
            best_name, best_val = max(values, key=lambda x: x[1])
            short_key = key.split("/")[-1]
            lines.append(f"| {short_key} | {best_name} | {best_val:.4f} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare downstream experiment results")
    parser.add_argument("--runs", nargs="*", type=str, default=None,
                        help="Specific run names to compare (default: all)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save comparison as JSON")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Save comparison as Markdown")
    parser.add_argument("--entity", type=str, default=ENTITY)
    parser.add_argument("--project", type=str, default=PROJECT)

    args = parser.parse_args()

    if wandb is None:
        print("ERROR: wandb is required for this tool")
        return

    print(f"Fetching runs from {args.entity}/{args.project}...")
    runs = fetch_runs(args.entity, args.project, args.runs)
    print(f"Found {len(runs)} runs")

    print_comparison_table(runs)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Serialize
        serializable = []
        for r in runs:
            sr = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                  for k, v in r.items() if v is not None}
            serializable.append(sr)
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"JSON saved to {args.output}")

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        md = generate_markdown_report(runs)
        with open(args.markdown, "w") as f:
            f.write(md)
        print(f"Markdown saved to {args.markdown}")


if __name__ == "__main__":
    main()
