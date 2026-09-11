"""SFC pair features with proper nested CV scalers."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def build_sfc_features_from_raw(
    X_fc_raw: np.ndarray,
    X_sc_raw: np.ndarray,
    analysis_idx: np.ndarray,
    validation_idx: np.ndarray,
    edge_pair_idx: np.ndarray,
    n_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build SFC pair features for analysis and validation sets.

    FC/SC scalers are fit ONLY on analysis_idx subjects.
    This prevents leakage across nested CV folds.

    Parameters
    ----------
    X_fc_raw : (n_subjects, n_edges) raw FC edge values
    X_sc_raw : (n_subjects, n_edges) raw SC edge values
    analysis_idx : indices of analysis (training) subjects
    validation_idx : indices of validation (held-out) subjects
    edge_pair_idx : (n_edges,) mapping each edge to a system pair index (-1 = unmapped)
    n_pairs : number of system pairs (45)

    Returns
    -------
    X_sfc_analysis : (len(analysis_idx), n_pairs) pair features for analysis
    X_sfc_validation : (len(validation_idx), n_pairs) pair features for validation
    """
    # Fit scalers on analysis subjects only
    fc_scaler = StandardScaler()
    sc_scaler = StandardScaler()

    fc_analysis = X_fc_raw[analysis_idx]
    sc_analysis = X_sc_raw[analysis_idx]

    fc_scaler.fit(fc_analysis)
    sc_scaler.fit(sc_analysis)

    # Transform analysis set
    fc_analysis_z = fc_scaler.transform(fc_analysis)
    sc_analysis_z = sc_scaler.transform(sc_analysis)

    # Handle empty validation set
    has_validation = len(validation_idx) > 0
    if has_validation:
        fc_validation_z = fc_scaler.transform(X_fc_raw[validation_idx])
        sc_validation_z = sc_scaler.transform(X_sc_raw[validation_idx])
    else:
        fc_validation_z = np.zeros((0, X_fc_raw.shape[1]))
        sc_validation_z = np.zeros((0, X_sc_raw.shape[1]))

    # Replace NaN/Inf
    fc_analysis_z = np.nan_to_num(fc_analysis_z, nan=0.0, posinf=0.0, neginf=0.0)
    sc_analysis_z = np.nan_to_num(sc_analysis_z, nan=0.0, posinf=0.0, neginf=0.0)
    fc_validation_z = np.nan_to_num(fc_validation_z, nan=0.0, posinf=0.0, neginf=0.0)
    sc_validation_z = np.nan_to_num(sc_validation_z, nan=0.0, posinf=0.0, neginf=0.0)

    # Interaction: h = z_FC * z_SC
    h_analysis = fc_analysis_z * sc_analysis_z
    h_validation = fc_validation_z * sc_validation_z

    # Aggregate by system pair
    valid_mask = edge_pair_idx >= 0
    valid_pairs = edge_pair_idx[valid_mask]

    X_sfc_analysis = np.zeros((len(analysis_idx), n_pairs))
    X_sfc_validation = np.zeros((len(validation_idx), n_pairs))

    for p in range(n_pairs):
        edge_mask = valid_pairs == p
        if edge_mask.sum() == 0:
            continue
        X_sfc_analysis[:, p] = h_analysis[:, valid_mask][:, edge_mask].mean(axis=1)
        X_sfc_validation[:, p] = h_validation[:, valid_mask][:, edge_mask].mean(axis=1)

    return X_sfc_analysis, X_sfc_validation
