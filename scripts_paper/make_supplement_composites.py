#!/usr/bin/env python3
"""
Generate combined supplementary figures by replotting from source data.

Creates:
  supp_fig_priors_combined.pdf      - (a) WM prior, (b) FI prior
  supp_fig_fusion_weights_combined.pdf - (a) WM weights, (b) FI weights
  supp_fig_top_edges_combined.pdf   - (a) WM edges, (b) FI edges

Uses direct replotting from source data to maintain vector quality.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import nibabel as nib

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

OUTPUT_DIR = REPO_ROOT / "figures_iclr"

FONT_SIZE = 8
TICK_SIZE = 7

WM_PKL = REPO_ROOT / "outputs/iclr/lf1_final_10x5/working_memory/all_split_results.pkl"
FI_PKL = REPO_ROOT / "outputs/iclr/lf1_final_10x5/fluid_intelligence/all_split_results.pkl"

WM_PRIOR = REPO_ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv"
FI_PRIOR = REPO_ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv"

WM_STABLE_TOP = REPO_ROOT / "outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/working_memory/stable_top_edges.csv"
FI_STABLE_TOP = REPO_ROOT / "outputs/iclr/lf1_final_10x5/frozen_fp_coefficients/fluid_intelligence/stable_top_edges.csv"

ATLAS = REPO_ROOT / "inputs/atlases/AAL116.nii.gz"

N_ROI = 116
N_EDGE = N_ROI * (N_ROI - 1) // 2


def load_fusion_weights(pkl_path: Path) -> pd.DataFrame:
    with pkl_path.open("rb") as f:
        results = pickle.load(f)
    rows = []
    for r in results:
        w_fp = float(r.lf1_weights["FP"])
        w_sc = float(r.lf1_weights["S"])
        rows.append({
            "seed": int(r.seed),
            "outer_fold": int(r.outer_fold),
            "w_prior_fc": w_fp,
            "w_sc": w_sc,
        })
    return pd.DataFrame(rows)


def edge_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector).reshape(-1)
    matrix = np.zeros((N_ROI, N_ROI), dtype=float)
    iu = np.triu_indices(N_ROI, k=1)
    matrix[iu] = vector
    matrix[(iu[1], iu[0])] = vector
    return matrix


def panel_fusion_weights(ax, df: pd.DataFrame, task_label: str, panel_letter: str):
    w = df["w_prior_fc"]
    bins = np.arange(-0.025, 1.025 + 1e-12, 0.05)
    ax.hist(w.to_numpy(), bins=bins, color="#1f77b4", edgecolor="black", linewidth=0.3)
    ax.axvline(w.mean(), linestyle="--", linewidth=0.9, color="#d62728",
               label=f"mean = {w.mean():.2f}")
    ax.set_xlim(-0.025, 1.025)
    ax.set_xlabel("Prior-aware FC fusion weight", fontsize=7)
    ax.set_ylabel("Outer splits", fontsize=7)
    ax.set_title(f"({panel_letter}) {task_label}", fontsize=FONT_SIZE, fontweight="bold", pad=4)
    ax.legend(frameon=False, fontsize=6)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Fusion weights combined ----
    wm_wf = load_fusion_weights(WM_PKL)
    fi_wf = load_fusion_weights(FI_PKL)

    print("Fusion weights:")
    print(f"  WM n={len(wm_wf)}, mean w_FP={wm_wf['w_prior_fc'].mean():.4f}")
    print(f"  FI n={len(fi_wf)}, mean w_FP={fi_wf['w_prior_fc'].mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    panel_fusion_weights(axes[0], wm_wf, "Working Memory", "a")
    panel_fusion_weights(axes[1], fi_wf, "Fluid Intelligence", "b")
    fig.tight_layout(w_pad=2.0)
    out = OUTPUT_DIR / "supp_fig_fusion_weights_combined.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ---- Top edges combined (requires nilearn) ----
    try:
        from nilearn import plotting

        atlas_img = nib.load(str(ATLAS))
        coordinates = plotting.find_parcellation_cut_coords(atlas_img)

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

        for idx, (stable_top, label) in enumerate([
            (WM_STABLE_TOP, "Working Memory"),
            (FI_STABLE_TOP, "Fluid Intelligence"),
        ]):
            edge_df = pd.read_csv(stable_top)
            edge_df = edge_df.sort_values("mean_abs_coef", ascending=False).head(20)

            coefficients = np.zeros(N_EDGE)
            for _, row in edge_df.iterrows():
                i = int(row["roi_i_1based"]) - 1
                j = int(row["roi_j_1based"]) - 1
                iu = np.triu_indices(N_ROI, k=1)
                for ei in range(N_EDGE):
                    if iu[0][ei] == min(i, j) and iu[1][ei] == max(i, j):
                        coefficients[ei] = float(row["mean_signed_coef"])
                        break

            matrix = edge_vector_to_matrix(coefficients)

            # Use nilearn to plot on the axis
            display = plotting.plot_connectome(
                matrix, coordinates,
                axes=axes[idx], edge_threshold=0.0, node_size=6,
                title=f"({'a' if idx == 0 else 'b'}) {label}",
            )

        fig.tight_layout(w_pad=2.0)
        out = OUTPUT_DIR / "supp_fig_top_edges_combined.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    except ImportError:
        print("nilearn not available; skipping top edges combined PDF.")
        print("Use individual PDFs instead.")

    print("\nSupplementary composites generated.")


if __name__ == "__main__":
    main()
