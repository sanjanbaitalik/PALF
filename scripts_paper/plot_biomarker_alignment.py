#!/usr/bin/env python3

"""
Plot matched-task-prior coefficient alignment for ONE cognitive task.

The final output hierarchy may store Working Memory and Fluid Intelligence
in separate directories, so a task/target column is NOT required.

Examples
--------

Working Memory:

python scripts_paper/plot_biomarker_alignment.py \
  --csv outputs/iclr/lf1_final_evidence_audit/working_memory/biomarker_seed_metrics.csv \
  --task "Working Memory" \
  --output figures/fig_biomarker_alignment_wm.pdf

Fluid Intelligence:

python scripts_paper/plot_biomarker_alignment.py \
  --csv outputs/iclr/lf1_final_evidence_audit/fluid_intelligence/biomarker_seed_metrics.csv \
  --task "Fluid Intelligence" \
  --output figures/fig_biomarker_alignment_fluid.pdf
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ORDER = [
    "No prior",
    "Matched",
    "Unrelated",
    "Shuffled",
    "Random",
]


def pick_column(df, names, required=True):
    for name in names:
        if name in df.columns:
            return name

    if required:
        raise KeyError(
            f"None of {names} found.\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def normalize_task(value):
    s = str(value).strip().lower()

    if (
        "work" in s
        or "list" in s
        or s == "wm"
    ):
        return "Working Memory"

    if (
        "fluid" in s
        or "pmat" in s
        or s == "fluid_intelligence"
    ):
        return "Fluid Intelligence"

    return str(value)


def normalize_condition(value):
    s = str(value).strip().lower()

    if "unrelated" in s:
        return "Unrelated"

    if "shuff" in s:
        return "Shuffled"

    if "random" in s:
        return "Random"

    if (
        "no_prior" in s
        or "no-prior" in s
        or "no prior" in s
        or s in {
            "lf0",
            "fc_ridge",
            "ridge",
            "no_prior",
        }
    ):
        return "No prior"

    if (
        "matched" in s
        or s in {
            "lf1",
            "fp",
            "palf",
        }
        or "prior_substitution" in s
        or "prior-substitution" in s
    ):
        return "Matched"

    return str(value)


def filter_task_if_present(df, task):
    """
    If the CSV contains task/target information, filter it.
    Otherwise assume the supplied file already belongs to one task.
    """

    task_col = pick_column(
        df,
        [
            "task",
            "target",
            "phenotype",
        ],
        required=False,
    )

    if task_col is None:
        return df

    wanted = normalize_task(task)

    tmp = df.copy()
    tmp["_paper_task"] = tmp[task_col].map(
        normalize_task
    )

    available = sorted(
        tmp["_paper_task"]
        .dropna()
        .unique()
    )

    out = tmp[
        tmp["_paper_task"] == wanted
    ].copy()

    if out.empty:
        raise ValueError(
            f'No rows found for task "{wanted}".\n'
            f"Tasks detected: {available}"
        )

    return out.drop(
        columns=["_paper_task"]
    )


def convert_to_long(df):
    """
    Return a DataFrame containing:

        condition
        alignment

    and seed when available.

    Supports both long and wide source tables.
    """

    seed_col = pick_column(
        df,
        ["seed"],
        required=False,
    )

    # ---------------------------------------------------------
    # Long-format attempt
    # ---------------------------------------------------------

    condition_col = pick_column(
        df,
        [
            "condition",
            "model",
            "method",
            "prior_condition",
            "prior_type",
        ],
        required=False,
    )

    alignment_col = pick_column(
        df,
        [
            "matched_task_prior_alignment",
            "alignment_matched_task_prior",
            "matched_alignment",
            "matched_task_alignment",
            "alignment",
        ],
        required=False,
    )

    if (
        condition_col is not None
        and alignment_col is not None
    ):

        out = pd.DataFrame()

        if seed_col is not None:
            out["seed"] = df[seed_col]

        out["condition"] = (
            df[condition_col]
            .map(normalize_condition)
        )

        out["alignment"] = pd.to_numeric(
            df[alignment_col],
            errors="coerce",
        )

        out = out[
            out["condition"].isin(ORDER)
        ].dropna(
            subset=["alignment"]
        )

        if out.empty:
            raise ValueError(
                "Long-format columns were detected, "
                "but none of the conditions could be "
                "mapped to No prior / Matched / "
                "Unrelated / Shuffled / Random.\n"
                f"Raw values in {condition_col}: "
                f"{sorted(df[condition_col].astype(str).unique())}"
            )

        return out

    # ---------------------------------------------------------
    # Wide-format attempt
    # ---------------------------------------------------------

    candidates = {
        "No prior": [
            "no_prior_alignment",
            "no_prior_matched_task_alignment",
            "alignment_no_prior",
            "fc_no_prior_alignment",
        ],

        "Matched": [
            "matched_alignment",
            "matched_task_prior_alignment",
            "alignment_matched",
            "fp_matched_alignment",
        ],

        "Unrelated": [
            "unrelated_alignment",
            "alignment_unrelated",
            "fp_unrelated_alignment",
        ],

        "Shuffled": [
            "shuffled_alignment",
            "alignment_shuffled",
            "fp_shuffled_alignment",
        ],

        "Random": [
            "random_alignment",
            "alignment_random",
            "fp_random_alignment",
        ],
    }

    found = {}

    for label, columns in candidates.items():
        col = pick_column(
            df,
            columns,
            required=False,
        )

        if col is not None:
            found[label] = col

    if len(found) < 2:
        raise KeyError(
            "Could not identify a supported biomarker "
            "alignment layout.\n"
            f"Available columns: {list(df.columns)}"
        )

    rows = []

    for label, col in found.items():

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        for idx, value in values.items():

            if pd.isna(value):
                continue

            row = {
                "condition": label,
                "alignment": float(value),
            }

            if seed_col is not None:
                row["seed"] = df.loc[
                    idx,
                    seed_col,
                ]

            rows.append(row)

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help=(
            "Task-specific or combined "
            "biomarker_seed_metrics.csv"
        ),
    )

    parser.add_argument(
        "--task",
        required=True,
        help=(
            'e.g. "Working Memory" '
            'or "Fluid Intelligence"'
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output PDF path",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    print(f"Loaded: {csv_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df)}")

    df = filter_task_if_present(
        df,
        args.task,
    )

    long_df = convert_to_long(df)

    print(
        f"\nTask: {args.task}"
    )

    print(
        "\nRecognized conditions:"
    )

    for condition in ORDER:

        values = long_df.loc[
            long_df["condition"] == condition,
            "alignment",
        ]

        if len(values) == 0:
            continue

        sd = (
            values.std(ddof=1)
            if len(values) > 1
            else float("nan")
        )

        print(
            f"  {condition:10s}: "
            f"n={len(values):2d}, "
            f"mean={values.mean():+.6f}, "
            f"sd={sd:.6f}"
        )

    stats = (
        long_df
        .groupby("condition")["alignment"]
        .agg(
            [
                "mean",
                "std",
                "count",
            ]
        )
        .reindex(ORDER)
        .dropna(
            subset=["mean"]
        )
    )

    stats["sem"] = (
        stats["std"]
        / np.sqrt(stats["count"])
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(3.45, 2.15)
    )

    x = np.arange(
        len(stats)
    )

    ax.bar(
        x,
        stats["mean"].to_numpy(),
        yerr=(
            stats["sem"]
            .fillna(0)
            .to_numpy()
        ),
        capsize=2,
    )

    ax.axhline(
        0.0,
        linewidth=0.8,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        stats.index,
        rotation=28,
        ha="right",
    )

    ax.set_ylabel(
        "Alignment with matched-task prior"
    )

    ax.margins(
        x=0.04
    )

    fig.tight_layout()

    output = Path(
        args.output
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
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()