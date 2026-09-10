"""
AAL116 → Yeo 7-network + SUBCORTICAL + CEREBELLAR mapping.

Uses external canonical atlas (Yeo 2011) independent of HCP data.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import datasets

SYSTEM_NAMES = [
    "VIS", "SM", "DATN", "LIMBIC", "FPN", "DMN", "VAS",
    "SUBCORTICAL", "CEREBELLAR"
]
N_SYSTEMS = 9

YEO_NETWORK_NAMES = {
    1: "VIS",       # Visual
    2: "SM",        # Somatomotor
    3: "DATN",      # Dorsal Attention
    4: "LIMBIC",    # Limbic
    5: "FPN",       # Frontoparietal Control
    6: "DMN",       # Default Mode
    7: "VAS",       # Ventral Attention / Salience
}

# Subcortical AAL ROI prefixes (matching AAL116_labels.csv)
SUBCORTICAL_PREFIXES = [
    "Caudate", "Putamen", "Pallidum", "Thalamus",
    "Hippocampus", "ParaHippocampal", "Amygdala"
]

CEREBELLAR_PREFIXES = ["Cerebelum", "Vermis"]


def _is_subcortical(label: str) -> bool:
    return any(label.startswith(p) for p in SUBCORTICAL_PREFIXES)


def _is_cerebellar(label: str) -> bool:
    return any(label.startswith(p) for p in CEREBELLAR_PREFIXES)


def build_aal116_yeo7_mapping(
    aal_nii_path: str = "inputs/atlases/AAL116.nii.gz",
    labels_csv_path: str = "inputs/atlases/AAL116_labels.csv",
) -> pd.DataFrame:
    """Build AAL116 → Yeo7 mapping via atlas overlap.

    Returns DataFrame with columns:
        roi_index, roi_label, system, winning_network, winning_overlap_fraction,
        n_voxels, overlap_VIS..overlap_VAS
    """
    labels_df = pd.read_csv(labels_csv_path)
    aal_img = nib.load(aal_nii_path)
    aal_data = aal_img.get_fdata()

    yeo_atlas = datasets.fetch_atlas_yeo_2011()
    yeo_img = nib.load(yeo_atlas["maps"])
    yeo_resampled = resample_to_img(yeo_img, aal_img, interpolation="nearest")
    yeo_data = np.asarray(yeo_resampled.get_fdata())
    while yeo_data.ndim > 3:
        yeo_data = yeo_data[:, :, :, 0]

    records = []
    n_unresolved = 0

    for _, row in labels_df.iterrows():
        roi_idx = int(row["roi_index"])
        roi_label = row["roi_label"]

        if _is_subcortical(roi_label):
            records.append({
                "roi_index": roi_idx,
                "roi_label": roi_label,
                "system": "SUBCORTICAL",
                "winning_network": "SUBCORTICAL",
                "winning_overlap_fraction": 1.0,
                "n_voxels": int((aal_data == roi_idx).sum()),
                "overlap_VIS": 0, "overlap_SM": 0, "overlap_DATN": 0,
                "overlap_LIMBIC": 0, "overlap_FPN": 0, "overlap_DMN": 0,
                "overlap_VAS": 0,
            })
            continue

        if _is_cerebellar(roi_label):
            records.append({
                "roi_index": roi_idx,
                "roi_label": roi_label,
                "system": "CEREBELLAR",
                "winning_network": "CEREBELLAR",
                "winning_overlap_fraction": 1.0,
                "n_voxels": int((aal_data == roi_idx).sum()),
                "overlap_VIS": 0, "overlap_SM": 0, "overlap_DATN": 0,
                "overlap_LIMBIC": 0, "overlap_FPN": 0, "overlap_DMN": 0,
                "overlap_VAS": 0,
            })
            continue

        # Cortical ROI: compute Yeo overlap
        roi_mask = aal_data == roi_idx
        n_voxels = int(roi_mask.sum())
        overlaps = {}
        for net_id in range(1, 8):
            net_name = YEO_NETWORK_NAMES[net_id]
            overlaps[net_name] = int(((yeo_data == net_id) & roi_mask).sum())

        if n_voxels == 0:
            winning = "UNRESOLVED"
            frac = 0.0
            n_unresolved += 1
        else:
            winning_id = int(np.argmax([overlaps[YEO_NETWORK_NAMES[i]] for i in range(1, 8)])) + 1
            winning = YEO_NETWORK_NAMES[winning_id]
            frac = overlaps[winning] / n_voxels

        records.append({
            "roi_index": roi_idx,
            "roi_label": roi_label,
            "system": winning,
            "winning_network": winning,
            "winning_overlap_fraction": round(frac, 4),
            "n_voxels": n_voxels,
            "overlap_VIS": overlaps.get("VIS", 0),
            "overlap_SM": overlaps.get("SM", 0),
            "overlap_DATN": overlaps.get("DATN", 0),
            "overlap_LIMBIC": overlaps.get("LIMBIC", 0),
            "overlap_FPN": overlaps.get("FPN", 0),
            "overlap_DMN": overlaps.get("DMN", 0),
            "overlap_VAS": overlaps.get("VAS", 0),
        })

    df = pd.DataFrame(records)
    if n_unresolved > 0:
        print(f"WARNING: {n_unresolved} cortical ROIs have zero voxels")
    return df


def build_system_pair_list() -> pd.DataFrame:
    """Return all valid unordered system pairs (including within-system)."""
    pairs = []
    pair_idx = 0
    for i, s1 in enumerate(SYSTEM_NAMES):
        for j, s2 in enumerate(SYSTEM_NAMES):
            if j >= i:
                pairs.append({
                    "pair_index": pair_idx,
                    "system_a": s1,
                    "system_b": s2,
                    "pair_name": f"{s1}-{s2}" if s1 != s2 else s1,
                })
                pair_idx += 1
    return pd.DataFrame(pairs)


def roi_to_system_array(mapping_df: pd.DataFrame) -> np.ndarray:
    """Return (116,) system-index array for each ROI."""
    system_to_idx = {s: i for i, s in enumerate(SYSTEM_NAMES)}
    arr = np.zeros(116, dtype=int)
    for _, row in mapping_df.iterrows():
        arr[row["roi_index"] - 1] = system_to_idx[row["system"]]
    return arr


def edge_to_system_pair(
    fc_upper_indices: np.ndarray,
    sc_upper_indices: np.ndarray,
    roi_system: np.ndarray,
    system_pairs_df: pd.DataFrame,
) -> np.ndarray:
    """Map each edge to a system-pair index.

    Parameters
    ----------
    fc_upper_indices : (n_edges, 2) array of (i, j) ROI index pairs for FC upper triangle
    sc_upper_indices : (n_edges, 2) array of (i, j) ROI index pairs for SC upper triangle
    roi_system : (116,) system index per ROI
    system_pairs_df : DataFrame with pair_index, system_a, system_b

    Returns
    -------
    edge_pair_indices : (n_edges,) int array of pair indices
    """
    n_edges = fc_upper_indices.shape[0]
    system_to_idx = {s: i for i, s in enumerate(SYSTEM_NAMES)}
    pair_lookup = {}
    for _, row in system_pairs_df.iterrows():
        sa = system_to_idx[row["system_a"]]
        sb = system_to_idx[row["system_b"]]
        key = (min(sa, sb), max(sa, sb))
        pair_lookup[key] = row["pair_index"]

    edge_pair_indices = np.zeros(n_edges, dtype=int)
    for e in range(n_edges):
        i, j = fc_upper_indices[e]
        si = roi_system[i]
        sj = roi_system[j]
        key = (min(si, sj), max(si, sj))
        edge_pair_indices[e] = pair_lookup.get(key, -1)

    return edge_pair_indices


def save_mapping(
    mapping_df: pd.DataFrame,
    output_dir: str,
) -> dict:
    """Save mapping CSV and QC JSON. Return QC dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_df.to_csv(output_dir / "AAL116_to_Yeo7_plus_subcortical_cerebellar.csv", index=False)

    cortical = mapping_df[~mapping_df["system"].isin(["SUBCORTICAL", "CEREBELLAR"])]
    unresolved = cortical[cortical["system"] == "UNRESOLVED"]

    system_counts = mapping_df["system"].value_counts().to_dict()
    min_overlap = cortical["winning_overlap_fraction"].min() if len(cortical) > 0 else None

    qc = {
        "atlas_source": "AAL116 (inputs/atlases/AAL116.nii.gz)",
        "yeo_source": "Yeo 2011 7-network atlas via nilearn",
        "n_rois": 116,
        "n_cortical": int(len(cortical)),
        "n_subcortical": int((mapping_df["system"] == "SUBCORTICAL").sum()),
        "n_cerebellar": int((mapping_df["system"] == "CEREBELLAR").sum()),
        "n_unresolved": int(len(unresolved)),
        "system_counts": system_counts,
        "min_cortical_overlap_fraction": float(min_overlap) if min_overlap is not None else None,
        "unresolved_rois": unresolved["roi_index"].tolist() if len(unresolved) > 0 else [],
    }

    mapping_hash = hashlib.sha256(
        mapping_df.to_csv(index=False).encode()
    ).hexdigest()[:16]
    qc["mapping_csv_sha256_prefix"] = mapping_hash

    with open(output_dir / "mapping_qc.json", "w") as f:
        json.dump(qc, f, indent=2, default=str)

    return qc
