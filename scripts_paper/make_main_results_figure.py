#!/usr/bin/env python3
"""
Create a publication-quality 2x2 composite figure for the main paper.

Panel (a): Working Memory - prediction delta r
Panel (b): Fluid Intelligence - prediction delta r
Panel (c): Working Memory - prior alignment
Panel (d): Fluid Intelligence - prior alignment

Replots directly from authoritative source CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# Data sources
WM_SEED_CSV = REPO_ROOT / "outputs/iclr/lf1_final_10x5/working_memory/seed_metrics.csv"
FI_SEED_CSV = REPO_ROOT / "outputs/iclr/lf1_final_10x5/fluid_intelligence/seed_metrics.csv"
WM_BIO_CSV = REPO_ROOT / "outputs/iclr/lf1_final_evidence_audit/biomarker_seed_metrics_working_memory.csv"
FI_BIO_CSV = REPO_ROOT / "outputs/iclr/lf1_final_evidence_audit/biomarker_seed_metrics_fluid_intelligence.csv"

OUTPUT_DIR = REPO_ROOT / "figures_iclr"

# Shared style constants
FONT_SIZE = 8
TICK_SIZE = 7
LABEL_SIZE = 8
MARKER_SIZE = 28

CONDITION_ORDER = ["No prior", "Matched", "Unrelated", "Shuffled", "Random"]
DISPLAY_LABELS = {
    "No prior": "No prior",
    "Matched": "Matched",
    "Unrelated": "Cross-task",
    "Shuffled": "Shuffled",
    "Random": "Random",
}


def normalize_model(name):
    s = str(name).strip().lower()
    if s == "lf1" or "lf1" in s or "prior_substitution" in s or "prior-substitution" in s:
        return "PALF"
    if s == "lf0" or "lf0" in s or "no_prior_late" in s or "no-prior late" in s:
        return "No-prior late fusion"
    return str(name)


def compute_deltas(csv_path: Path) -> pd.Series:
    """Compute PALF minus no-prior Pearson deltas per seed."""
    df = pd.read_csv(csv_path)
    df["paper_model"] = df["model"].map(normalize_model)
    keep = df[df["paper_model"].isin(["PALF", "No-prior late fusion"])].copy()
    pivot = keep.pivot_table(
        index="seed", columns="paper_model", values="pearson", aggfunc="mean",
    )
    pivot = pivot.dropna(subset=["PALF", "No-prior late fusion"])
    pivot = pivot.sort_index()
    delta = pivot["PALF"] - pivot["No-prior late fusion"]
    return delta


def load_biomarker_data(csv_path: Path) -> pd.DataFrame:
    """Load and process biomarker alignment data from evidence audit CSV."""
    df = pd.read_csv(csv_path)
    seed_col = "seed" if "seed" in df.columns else None

    # Evidence audit format: model column + alignment columns
    if "model" in df.columns and "matched_alignment" in df.columns:
        rows = []
        for _, row in df.iterrows():
            model = str(row["model"]).strip().lower()
            if "no_prior" in model or model == "fc_no_prior":
                cond = "No prior"
                val = row["matched_alignment"]
            elif "fp_matched" in model or "matched" in model:
                cond = "Matched"
                val = row["matched_alignment"]
            elif "fp_unrelated" in model or "unrelated" in model:
                cond = "Unrelated"
                val = row["matched_alignment"]
            elif "fp_shuffled" in model or "shuffled" in model:
                cond = "Shuffled"
                val = row["matched_alignment"]
            elif "fp_random" in model or "random" in model:
                cond = "Random"
                val = row["matched_alignment"]
            else:
                continue
            entry = {"condition": cond, "alignment": float(val)}
            if seed_col:
                entry["seed"] = row[seed_col]
            rows.append(entry)
        return pd.DataFrame(rows)

    raise ValueError(f"Unrecognized biomarker CSV format: {list(df.columns)}")


def panel_prediction(ax, delta: pd.Series, task_label: str, panel_letter: str):
    """Draw a prediction delta scatter panel."""
    ax.scatter(delta.index, delta.values, s=MARKER_SIZE, zorder=3, color="#1f77b4")
    ax.axhline(0.0, linewidth=0.7, linestyle="--", color="gray")
    ax.axhline(delta.mean(), linewidth=0.9, linestyle=":", color="#d62728")
    ax.set_xlabel("Seed", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"PALF $-$ no-prior $\Delta r$", fontsize=LABEL_SIZE)
    ax.set_xticks(list(delta.index))
    ax.set_xticklabels([str(int(s)) for s in delta.index], fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_title(f"({panel_letter}) {task_label}", fontsize=FONT_SIZE, fontweight="bold", pad=4)
    ax.set_ylim(-0.015, 0.060)
    ax.margins(x=0.05)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def panel_biomarker(ax, bio_df: pd.DataFrame, task_label: str, panel_letter: str):
    """Draw a biomarker alignment bar panel."""
    stats = (
        bio_df.groupby("condition")["alignment"]
        .agg(["mean", "std", "count"])
        .reindex(CONDITION_ORDER)
        .dropna(subset=["mean"])
    )
    stats["sem"] = stats["std"] / np.sqrt(stats["count"])

    x = np.arange(len(stats))
    bars = ax.bar(
        x, stats["mean"].to_numpy(),
        yerr=stats["sem"].fillna(0).to_numpy(),
        capsize=2, color="#1f77b4", edgecolor="black", linewidth=0.3,
    )
    ax.axhline(0.0, linewidth=0.7, color="gray")
    ax.set_xticks(x)
    display_names = [DISPLAY_LABELS.get(idx, idx) for idx in stats.index]
    ax.set_xticklabels(display_names, rotation=28, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel(r"ROI-prior alignment (Spearman $\rho$)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_title(f"({panel_letter}) {task_label}", fontsize=FONT_SIZE, fontweight="bold", pad=4)
    ax.set_ylim(-0.15, 0.85)
    ax.margins(x=0.04)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compute deltas
    wm_delta = compute_deltas(WM_SEED_CSV)
    fi_delta = compute_deltas(FI_SEED_CSV)

    # Load biomarker data
    wm_bio = load_biomarker_data(WM_BIO_CSV)
    fi_bio = load_biomarker_data(FI_BIO_CSV)

    # Print source files
    print("Source files used:")
    print(f"  {WM_SEED_CSV}")
    print(f"  {FI_SEED_CSV}")
    print(f"  {WM_BIO_CSV}")
    print(f"  {FI_BIO_CSV}")

    # Print sanity checks
    print(f"\nWM mean delta r: {wm_delta.mean():+.4f}")
    print(f"WM positive seeds: {(wm_delta > 0).sum()}/{len(wm_delta)}")
    print(f"FI mean delta r: {fi_delta.mean():+.4f}")
    print(f"FI positive seeds: {(fi_delta > 0).sum()}/{len(fi_delta)}")

    for cond in CONDITION_ORDER:
        vals = wm_bio.loc[wm_bio["condition"] == cond, "alignment"]
        if len(vals) > 0:
            print(f"  WM {cond:10s}: mean={vals.mean():.4f}")
    for cond in CONDITION_ORDER:
        vals = fi_bio.loc[fi_bio["condition"] == cond, "alignment"]
        if len(vals) > 0:
            print(f"  FI {cond:10s}: mean={vals.mean():.4f}")

    # Create 2x2 figure at ICLR text width (~7.0 in)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))

    panel_prediction(axes[0, 0], wm_delta, "Working Memory", "a")
    panel_prediction(axes[0, 1], fi_delta, "Fluid Intelligence", "b")
    panel_biomarker(axes[1, 0], wm_bio, "Working Memory", "c")
    panel_biomarker(axes[1, 1], fi_bio, "Fluid Intelligence", "d")

    fig.tight_layout(h_pad=1.5, w_pad=1.5)

    pdf_path = OUTPUT_DIR / "fig_main_results.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    print(f"\nSaved: {pdf_path}")

    png_path = OUTPUT_DIR / "fig_main_results.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {png_path}")

    plt.close(fig)
    print("\nMain results composite generated.")


if __name__ == "__main__":
    main()
