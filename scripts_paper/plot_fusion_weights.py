#!/usr/bin/env python3

"""
Plot the actual split-level LF1/PALF fusion weights from all_split_results.pkl.

The final LF1 experiment stores 50 OuterSplitResult objects per task:
    10 seeds x 5 outer folds.

Each result contains:
    result.seed
    result.outer_fold
    result.lf1_weights["FP"]
    result.lf1_weights["S"]

Examples
--------

Working Memory:

PYTHONPATH=src python scripts_paper/plot_fusion_weights.py \
  --pkl outputs/iclr/lf1_final_10x5/working_memory/all_split_results.pkl \
  --task "Working Memory" \
  --output figures/supp_fig_fusion_weights_wm.pdf \
  --export-csv figures/source_fusion_weights_wm.csv


Fluid Intelligence:

PYTHONPATH=src python scripts_paper/plot_fusion_weights.py \
  --pkl outputs/iclr/lf1_final_10x5/fluid_intelligence/all_split_results.pkl \
  --task "Fluid Intelligence" \
  --output figures/supp_fig_fusion_weights_fluid.pdf \
  --export-csv figures/source_fusion_weights_fluid.csv
"""

from pathlib import Path
import argparse
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Make repository modules importable during pickle loading.
# This is needed because pickle may reference repository classes
# such as OuterSplitResult and Level1Result.
# ------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if SRC_DIR.exists():
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


def load_results(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Pickle not found: {path}"
        )

    with path.open("rb") as f:
        results = pickle.load(f)

    if not isinstance(
        results,
        (list, tuple),
    ):
        raise TypeError(
            "Expected all_split_results.pkl "
            "to contain a list/tuple."
        )

    if len(results) == 0:
        raise ValueError(
            "Pickle contains no split results."
        )

    return results


def extract_weights(results):
    rows = []

    for i, result in enumerate(results):

        # ----------------------------------------------------
        # Required metadata
        # ----------------------------------------------------

        if not hasattr(
            result,
            "seed",
        ):
            raise AttributeError(
                f"Result {i} has no seed field."
            )

        if not hasattr(
            result,
            "outer_fold",
        ):
            raise AttributeError(
                f"Result {i} has no outer_fold field."
            )

        if not hasattr(
            result,
            "lf1_weights",
        ):
            raise AttributeError(
                f"Result {i} has no lf1_weights field."
            )

        weights = (
            result.lf1_weights
        )

        if not isinstance(
            weights,
            dict,
        ):
            raise TypeError(
                f"Result {i}: lf1_weights "
                "is not a dictionary."
            )

        if "FP" not in weights:
            raise KeyError(
                f"Result {i}: "
                "lf1_weights has no FP entry. "
                f"Available: {list(weights.keys())}"
            )

        if "S" not in weights:
            raise KeyError(
                f"Result {i}: "
                "lf1_weights has no S entry. "
                f"Available: {list(weights.keys())}"
            )

        w_fp = float(
            weights["FP"]
        )

        w_sc = float(
            weights["S"]
        )

        rows.append(
            {
                "seed":
                    int(result.seed),

                "outer_fold":
                    int(
                        result.outer_fold
                    ),

                "w_prior_fc":
                    w_fp,

                "w_sc":
                    w_sc,

                "weight_sum":
                    w_fp + w_sc,
            }
        )

    return pd.DataFrame(rows)


def validate_weights(df):

    # --------------------------------------------------------
    # Expected cardinality
    # --------------------------------------------------------

    print(
        f"Number of split records: "
        f"{len(df)}"
    )

    if len(df) != 50:
        print(
            "WARNING: expected 50 records "
            "(10 seeds x 5 folds)."
        )

    seeds = sorted(
        df["seed"].unique()
    )

    folds = sorted(
        df["outer_fold"].unique()
    )

    print(
        f"Seeds: {seeds}"
    )

    print(
        f"Outer folds: {folds}"
    )

    # --------------------------------------------------------
    # Duplicate split detection
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "seed",
            "outer_fold",
        ]
    )

    if duplicates.any():
        raise ValueError(
            "Duplicate seed/fold records found."
        )

    # --------------------------------------------------------
    # Range validation
    # --------------------------------------------------------

    for column in [
        "w_prior_fc",
        "w_sc",
    ]:

        if (
            (df[column] < -1e-10).any()
            or
            (df[column] > 1 + 1e-10).any()
        ):

            raise ValueError(
                f"{column} contains values "
                "outside [0,1]."
            )

    # --------------------------------------------------------
    # Simplex validation
    # --------------------------------------------------------

    max_sum_error = np.max(
        np.abs(
            df["weight_sum"]
            - 1.0
        )
    )

    print(
        "Maximum "
        "|w_prior_fc + w_sc - 1| "
        f"= {max_sum_error:.3e}"
    )

    if max_sum_error > 1e-8:
        raise ValueError(
            "LF1 fusion weights "
            "do not sum to one."
        )

    # --------------------------------------------------------
    # 0.05-grid validation
    # --------------------------------------------------------

    grid_error = np.abs(
        df["w_prior_fc"] / 0.05
        - np.round(
            df["w_prior_fc"] / 0.05
        )
    )

    max_grid_error = (
        grid_error.max()
    )

    print(
        "Maximum 0.05-grid error "
        f"= {max_grid_error:.3e}"
    )

    if max_grid_error > 1e-8:
        print(
            "WARNING: some weights "
            "are not on the expected "
            "0.05 grid."
        )


def print_summary(
    df,
    task,
):

    w = df[
        "w_prior_fc"
    ]

    print(
        "\n"
        + "=" * 60
    )

    print(task)

    print(
        "=" * 60
    )

    print(
        f"Mean prior-aware FC weight   : "
        f"{w.mean():.6f}"
    )

    print(
        f"Median prior-aware FC weight : "
        f"{w.median():.6f}"
    )

    print(
        f"SD                           : "
        f"{w.std(ddof=1):.6f}"
    )

    print(
        f"Min                          : "
        f"{w.min():.2f}"
    )

    print(
        f"Max                          : "
        f"{w.max():.2f}"
    )

    print(
        f"Fraction w_FP = 0            : "
        f"{(w == 0).mean():.3f}"
    )

    print(
        f"Fraction w_FP >= 0.10        : "
        f"{(w >= 0.10).mean():.3f}"
    )

    print(
        f"Fraction w_FP >= 0.25        : "
        f"{(w >= 0.25).mean():.3f}"
    )

    print(
        f"Fraction w_FP >= 0.50        : "
        f"{(w >= 0.50).mean():.3f}"
    )

    print(
        "\nMean by seed:"
    )

    seed_means = (
        df.groupby("seed")
        ["w_prior_fc"]
        .mean()
    )

    for seed, value in (
        seed_means.items()
    ):

        print(
            f"  seed {seed:2d}: "
            f"{value:.4f}"
        )


def plot_weights(
    df,
    task,
    output,
):

    w = df[
        "w_prior_fc"
    ]

    # Fusion search uses increments of 0.05.
    bins = np.arange(
        -0.025,
        1.025 + 1e-12,
        0.05,
    )

    fig, ax = plt.subplots(
        figsize=(3.15, 2.05)
    )

    ax.hist(
        w.to_numpy(),
        bins=bins,
    )

    ax.axvline(
        w.mean(),
        linestyle="--",
        linewidth=1.0,
        label=(
            f"mean = "
            f"{w.mean():.2f}"
        ),
    )

    ax.set_xlim(
        -0.025,
        1.025,
    )

    ax.set_xlabel(
        "Prior-aware FC fusion weight"
    )

    ax.set_ylabel(
        "Outer splits"
    )

    ax.legend(
        frameon=False,
        fontsize=7,
    )

    fig.tight_layout()

    output = Path(
        output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved figure: {output}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pkl",
        required=True,
    )

    parser.add_argument(
        "--task",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--export-csv",
        default=None,
    )

    args = parser.parse_args()

    results = load_results(
        args.pkl
    )

    df = extract_weights(
        results
    )

    validate_weights(df)

    print_summary(
        df,
        args.task,
    )

    if args.export_csv:

        csv_path = Path(
            args.export_csv
        )

        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            csv_path,
            index=False,
        )

        print(
            f"\nExported source data: "
            f"{csv_path}"
        )

    plot_weights(
        df,
        args.task,
        args.output,
    )


if __name__ == "__main__":
    main()