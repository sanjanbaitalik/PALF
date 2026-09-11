"""AAL116 → Yeo7 system mapping with correct Yeo 2011 labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import resample_to_img

# Correct Yeo 2011 7-network labels (from the original paper)
YEO_NETWORK_NAMES = {
    1: "VIS",        # Visual
    2: "SM",         # Somatomotor
    3: "DAN",        # Dorsal Attention Network
    4: "VAN",        # Ventral Attention / Salience Network
    5: "LIMBIC",     # Limbic
    6: "FPN",        # Frontoparietal Control Network
    7: "DMN",        # Default Mode Network
}

SYSTEM_NAMES = ["VIS", "SM", "DAN", "VAN", "LIMBIC", "FPN", "DMN", "SUBCORTICAL", "CEREBELLAR"]
N_SYSTEMS = len(SYSTEM_NAMES)

# Subcortical AAL ROI prefixes
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
    """Build AAL116 → Yeo7 + subcortical/cerebellar mapping from atlas overlap."""
    aal_img = nib.load(aal_nii_path)
    aal_data = np.asarray(aal_img.get_fdata())

    labels_df = pd.read_csv(labels_csv_path)

    # Load Yeo 7-network atlas
    yeo = datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick")
    yeo_img = nib.load(yeo["maps"])
    yeo_resampled = resample_to_img(yeo_img, aal_img, interpolation="nearest")
    yeo_data = np.asarray(yeo_resampled.get_fdata())
    while yeo_data.ndim > 3:
        yeo_data = yeo_data[:, :, :, 0]

    records = []
    unresolved = []
    min_overlap_frac = 1.0

    for _, row in labels_df.iterrows():
        roi_id = int(row.iloc[0])
        roi_label = str(row.iloc[1])

        roi_mask = aal_data == roi_id
        n_voxels = int(roi_mask.sum())

        if _is_subcortical(roi_label):
            system = "SUBCORTICAL"
            winning_id = -1
            overlap_frac = 1.0
        elif _is_cerebellar(roi_label):
            system = "CEREBELLAR"
            winning_id = -2
            overlap_frac = 1.0
        else:
            # Cortical: find max overlap with Yeo networks
            overlaps = {}
            for net_id in range(1, 8):
                net_name = YEO_NETWORK_NAMES[net_id]
                count = int(((yeo_data == net_id) & roi_mask).sum())
                overlaps[net_name] = count

            total_overlap = sum(overlaps.values())
            if total_overlap == 0:
                system = "UNRESOLVED"
                winning_id = 0
                overlap_frac = 0.0
                unresolved.append(roi_label)
            else:
                winning_net = max(overlaps, key=overlaps.get)
                winning_count = overlaps[winning_net]
                overlap_frac = winning_count / n_voxels if n_voxels > 0 else 0.0
                system = winning_net
                winning_id = [k for k, v in YEO_NETWORK_NAMES.items() if v == winning_net][0]

                if overlap_frac < min_overlap_frac:
                    min_overlap_frac = overlap_frac

        records.append({
            "roi_id": roi_id,
            "roi_label": roi_label,
            "system": system,
            "yeo_winning_id": winning_id,
            "n_voxels": n_voxels,
            "overlap_fraction": overlap_frac,
        })

    df = pd.DataFrame(records)

    # Build QC
    system_counts = df["system"].value_counts().to_dict()
    n_cortical = len(df[~df["system"].isin(["SUBCORTICAL", "CEREBELLAR", "UNRESOLVED"])])
    n_resolved = n_cortical - len(unresolved)

    qc = {
        "nilearn_version": str(datasets.__version__) if hasattr(datasets, "__version__") else "unknown",
        "atlas_path": aal_nii_path,
        "atlas_hash": hashlib.sha256(open(aal_nii_path, "rb").read()).hexdigest()[:16],
        "labels_path": labels_csv_path,
        "yeo_version": "Yeo 2011 7-network (thick)",
        "yeo_id_mapping": YEO_NETWORK_NAMES,
        "system_counts": system_counts,
        "n_total_rois": len(df),
        "n_cortical": n_cortical,
        "n_subcortical": int(system_counts.get("SUBCORTICAL", 0)),
        "n_cerebellar": int(system_counts.get("CEREBELLAR", 0)),
        "n_unresolved": len(unresolved),
        "unresolved_rois": unresolved,
        "min_cortical_overlap_fraction": float(min_overlap_frac),
        "n_remapped": 0,  # Will be updated after comparison
    }

    return df, qc


def save_mapping(df: pd.DataFrame, qc: dict, output_dir: Path) -> None:
    """Save mapping CSV and QC JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "AAL116_to_Yeo7_plus_subcortical_cerebellar.csv", index=False)
    with open(output_dir / "mapping_qc.json", "w") as f:
        json.dump(qc, f, indent=2)


def roi_to_system_array(df: pd.DataFrame) -> np.ndarray:
    """Convert mapping DataFrame to system name array indexed by ROI ID (1-based)."""
    max_id = int(df["roi_id"].max())
    arr = np.full(max_id + 1, "", dtype=object)
    for _, row in df.iterrows():
        arr[int(row["roi_id"])] = row["system"]
    return arr


def build_system_pair_list() -> pd.DataFrame:
    """Build the 45 unordered system pairs."""
    pairs = []
    for i, s1 in enumerate(SYSTEM_NAMES):
        for j, s2 in enumerate(SYSTEM_NAMES):
            if j >= i:
                pair_name = f"{s1}-{s2}" if s1 != s2 else s1
                pairs.append({"system_a": s1, "system_b": s2, "pair_name": pair_name})
    df = pd.DataFrame(pairs)
    assert len(df) == 45, f"Expected 45 pairs, got {len(df)}"
    return df


def edge_to_system_pair(
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    roi_system: np.ndarray,
    system_pairs_df: pd.DataFrame,
) -> np.ndarray:
    """Map edges to system pair indices. Returns -1 for unmapped edges."""
    n_edges = len(row_indices)
    pair_lookup = {}
    for idx, row in system_pairs_df.iterrows():
        a, b = sorted([row["system_a"], row["system_b"]])
        pair_lookup[(a, b)] = idx

    result = np.full(n_edges, -1, dtype=int)
    for e in range(n_edges):
        i = row_indices[e].item() if hasattr(row_indices[e], 'item') else int(row_indices[e])
        j = col_indices[e].item() if hasattr(col_indices[e], 'item') else int(col_indices[e])
        if i < len(roi_system) and j < len(roi_system):
            sys_i = roi_system[i]
            sys_j = roi_system[j]
            if sys_i and sys_j:
                key = tuple(sorted([sys_i, sys_j]))
                if key in pair_lookup:
                    result[e] = pair_lookup[key]

    return result
