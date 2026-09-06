#!/usr/bin/env python3
"""
Generate the seed-level PALF-minus-no-prior Pearson delta plot
for one cognitive task.

The final LF1 experiment stores Working Memory and Fluid Intelligence
in separate task directories, so each seed_metrics.csv contains only:

    seed, model, pearson

Examples
--------

Working Memory:

python scripts_paper/plot_seed_deltas.py \
    --seed-metrics outputs/iclr/lf1_final_10x5/working_memory/seed_metrics.csv \
    --task "Working Memory" \
    --output figures/fig_seed_deltas_wm.pdf

Fluid Intelligence:

python scripts_paper/plot_seed_deltas.py \
    --seed-metrics outputs/iclr/lf1_final_10x5/fluid_intelligence/seed_metrics.csv \
    --task "Fluid Intelligence" \
    --output figures/fig_seed_deltas_fluid.pdf
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def pick_column(df, names):
    for name in names:
        if name in df.columns:
            return name

    raise KeyError(
        f"None of {names} found. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_model(name):
    """
    Map repository model names to paper-facing names.
    """

    s = str(name).strip().lower()

    # Final prior-aware late-fusion model
    if (
        s == "lf1"
        or "lf1" in s
        or "prior_substitution" in s
        or "prior-substitution" in s
    ):
        return "PALF"

    # Architecture-matched no-prior late-fusion baseline
    if (
        s == "lf0"
        or "lf0" in s
        or "no_prior_late" in s
        or "no-prior late" in s
    ):
        return "No-prior late fusion"

    return str(name)


def task_stem(task):
    s = task.lower()

    if "work" in s or "wm" in s:
        return "wm"

    if "fluid" in s or "pmat" in s:
        return "fluid"

    return (
        s.replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot seed-level Pearson deltas between PALF (LF1) "
            "and the architecture-matched no-prior late-fusion "
            "baseline (LF0)."
        )
    )

    parser.add_argument(
        "--seed-metrics",
        required=True,
        help="Path to one task-specific seed_metrics.csv",
    )

    parser.add_argument(
        "--task",
        required=True,
        help='Task label, e.g. "Working Memory" or "Fluid Intelligence"',
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PDF path. If omitted, a filename is generated "
            "inside --output-dir."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="figures_iclr",
        help="Output directory when --output is not supplied.",
    )

    args = parser.parse_args()

    csv_path = Path(args.seed_metrics)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"seed_metrics.csv not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    print(f"Loaded: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")

    seed_col = pick_column(df, ["seed"])
    model_col = pick_column(df, ["model", "method"])
    pearson_col = pick_column(
        df,
        [
            "pearson",
            "pearson_mean",
            "pearson_r",
            "r",
        ],
    )

    df = df.copy()

    df["paper_model"] = df[model_col].map(normalize_model)

    print("\nModels found:")
    for model in sorted(df[model_col].astype(str).unique()):
        print(f"  {model}")

    keep = df[
        df["paper_model"].isin(
            [
                "PALF",
                "No-prior late fusion",
            ]
        )
    ].copy()

    if keep.empty:
        raise ValueError(
            "Could not find LF0/LF1 rows in the supplied CSV.\n"
            f"Raw model values were:\n"
            f"{sorted(df[model_col].astype(str).unique())}"
        )

    found_models = set(keep["paper_model"])

    required_models = {
        "PALF",
        "No-prior late fusion",
    }

    missing = required_models - found_models

    if missing:
        raise ValueError(
            f"Missing required model(s): {sorted(missing)}\n"
            f"Detected models: {sorted(found_models)}"
        )

    pivot = keep.pivot_table(
        index=seed_col,
        columns="paper_model",
        values=pearson_col,
        aggfunc="mean",
    )

    pivot = pivot.dropna(
        subset=[
            "PALF",
            "No-prior late fusion",
        ]
    )

    if pivot.empty:
        raise ValueError(
            "No aligned LF0/LF1 seed pairs could be constructed."
        )

    pivot = pivot.sort_index()

    delta = (
        pivot["PALF"]
        - pivot["No-prior late fusion"]
    )

    print(f"\nTask: {args.task}")
    print(f"Aligned seeds: {len(delta)}")

    print("\nSeed-level deltas:")
    for seed, value in delta.items():
        print(
            f"  seed {seed}: "
            f"Δr = {value:+.6f}"
        )

    print("\nSummary:")
    print(
        f"  LF0 mean: "
        f"{pivot['No-prior late fusion'].mean():.6f}"
    )
    print(
        f"  PALF mean: "
        f"{pivot['PALF'].mean():.6f}"
    )
    print(
        f"  mean Δr: "
        f"{delta.mean():+.6f}"
    )
    print(
        f"  median Δr: "
        f"{delta.median():+.6f}"
    )
    print(
        f"  positive seeds: "
        f"{int((delta > 0).sum())}/{len(delta)}"
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(3.25, 2.05)
    )

    # Zero-reference line
    ax.axhline(
        0.0,
        linewidth=0.9,
        linestyle="--",
    )

    ax.scatter(
        delta.index,
        delta.values,
        s=30,
        zorder=3,
    )

    # Mean delta
    ax.axhline(
        delta.mean(),
        linewidth=1.1,
        linestyle=":",
    )

    ax.set_xlabel(
        "Seed"
    )

    ax.set_ylabel(
        r"PALF $-$ no-prior Pearson $\Delta r$"
    )

    ax.set_xticks(
        list(delta.index)
    )

    ax.margins(x=0.05)

    fig.tight_layout()

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_dir = Path(args.output_dir)
        output_path = (
            output_dir
            / f"fig_seed_deltas_{task_stem(args.task)}.pdf"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved figure: {output_path}"
    )


if __name__ == "__main__":
    main()