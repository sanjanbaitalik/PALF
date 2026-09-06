#!/usr/bin/env python3
"""Plot top prior-aware FC coefficients from the frozen coefficient export.

Generates one figure per task showing the highest-magnitude edges from the
stable_top_edges.csv produced by export_frozen_fp_coefficients.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn import plotting

N_ROI = 116
N_EDGE = N_ROI * (N_ROI - 1) // 2


def edge_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector).reshape(-1)
    if vector.size != N_EDGE:
        raise ValueError(f"Expected {N_EDGE} edges, got {vector.size}")
    matrix = np.zeros((N_ROI, N_ROI), dtype=float)
    iu = np.triu_indices(N_ROI, k=1)
    matrix[iu] = vector
    matrix[(iu[1], iu[0])] = vector
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Plot top prior-aware FC coefficients.",
    )
    parser.add_argument(
        "--edge-summary",
        required=True,
        help="Path to stable_top_edges.csv from the coefficient export.",
    )
    parser.add_argument(
        "--atlas",
        required=True,
        help="Path to AAL116.nii.gz atlas.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top edges to display (default: 20).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the figure (PDF/PNG).",
    )
    parser.add_argument(
        "--task-label",
        default="",
        help="Optional task label for the figure title.",
    )
    args = parser.parse_args()

    if not (1 <= args.top_k <= N_EDGE):
        raise ValueError(f"--top-k must be between 1 and {N_EDGE}")

    edge_df = pd.read_csv(args.edge_summary)
    required_cols = {"mean_abs_coef", "mean_signed_coef", "roi_i_1based", "roi_j_1based"}
    missing = required_cols - set(edge_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    edge_df = edge_df.sort_values("mean_abs_coef", ascending=False).head(args.top_k)

    coefficients = np.zeros(N_EDGE)
    for _, row in edge_df.iterrows():
        i = int(row["roi_i_1based"]) - 1
        j = int(row["roi_j_1based"]) - 1
        iu = np.triu_indices(N_ROI, k=1)
        edge_idx = None
        for idx in range(N_EDGE):
            if iu[0][idx] == min(i, j) and iu[1][idx] == max(i, j):
                edge_idx = idx
                break
        if edge_idx is not None:
            coefficients[edge_idx] = float(row["mean_signed_coef"])

    matrix = edge_vector_to_matrix(coefficients)

    atlas_path = Path(args.atlas)
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas not found: {atlas_path}")

    atlas_img = nib.load(str(atlas_path))
    coordinates = plotting.find_parcellation_cut_coords(atlas_img)

    if coordinates.shape[0] != N_ROI:
        raise ValueError(
            f"Atlas yielded {coordinates.shape[0]} parcel coordinates; expected {N_ROI}."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    title = args.task_label or ""
    display = plotting.plot_connectome(
        matrix,
        coordinates,
        edge_threshold=0.0,
        node_size=8,
        title=title if title else None,
    )

    display.savefig(str(output))
    display.close()

    caption = (
        "Highest-magnitude prior-aware FC coefficients aggregated across "
        "the 50 frozen outer fits. The visualization is descriptive and "
        "does not imply causal connectivity."
    )
    print(f"\nCaption: {caption}")

    iu = np.triu_indices(N_ROI, k=1)
    rows = []
    for rank, (_, row) in enumerate(edge_df.iterrows(), start=1):
        roi_i = int(row["roi_i_1based"]) - 1
        roi_j = int(row["roi_j_1based"]) - 1
        rows.append({
            "rank": rank,
            "roi_i_1based": int(row["roi_i_1based"]),
            "roi_j_1based": int(row["roi_j_1based"]),
            "roi_i_name": str(row.get("roi_i_name", "")),
            "roi_j_name": str(row.get("roi_j_name", "")),
            "mean_abs_coef": float(row["mean_abs_coef"]),
            "mean_signed_coef": float(row["mean_signed_coef"]),
            "top20_frequency": float(row.get("top20_frequency", 0.0)),
            "sign_consistency": float(row.get("sign_consistency", 0.0)),
            "top10_frequency": float(row.get("top10_frequency", 0.0)),
        })

    sidecar = output.with_suffix(".top_edges.tsv")
    pd.DataFrame(rows).to_csv(sidecar, sep="\t", index=False)

    print(f"Saved figure: {output}")
    print(f"Saved edge list: {sidecar}")
    print(f"Top {args.top_k} edges plotted.")


if __name__ == "__main__":
    main()
