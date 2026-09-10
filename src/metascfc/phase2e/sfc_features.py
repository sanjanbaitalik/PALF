"""
Structure-Function Coupling (SFC) feature computation.

Computes elementwise FC×SC interaction features aggregated by system pairs.
"""
import numpy as np
import pandas as pd
from typing import Optional


def compute_sfc_pair_features(
    fc_upper: np.ndarray,
    sc_upper: np.ndarray,
    edge_pair_indices: np.ndarray,
    n_pairs: int,
    fc_scaler_mean: Optional[np.ndarray] = None,
    fc_scaler_std: Optional[np.ndarray] = None,
    sc_scaler_mean: Optional[np.ndarray] = None,
    sc_scaler_std: Optional[np.ndarray] = None,
    fit_scalers: bool = False,
) -> tuple:
    """Compute system-pair aggregated FC×SC interaction features.

    Parameters
    ----------
    fc_upper : (n_subjects, n_edges) raw FC upper triangle
    sc_upper : (n_subjects, n_edges) raw SC upper triangle
    edge_pair_indices : (n_edges,) system-pair index per edge (-1 = unmapped)
    n_pairs : total number of system pairs
    fc_scaler_mean, fc_scaler_std : FC standardization params (fit on train)
    sc_scaler_mean, sc_scaler_std : SC standardization params (fit on train)
    fit_scalers : if True, compute scalers from fc_upper/sc_upper

    Returns
    -------
    X_sfc : (n_subjects, n_pairs) pair-aggregated coupling features
    fc_mean, fc_std, sc_mean, sc_std : scalers used (for test-time reuse)
    """
    n_subj = fc_upper.shape[0]
    n_edges = fc_upper.shape[1]

    if fit_scalers:
        fc_scaler_mean = fc_upper.mean(axis=0)
        fc_scaler_std = fc_upper.std(axis=0)
        fc_scaler_std[fc_scaler_std < 1e-10] = 1.0
        sc_scaler_mean = sc_upper.mean(axis=0)
        sc_scaler_std = sc_upper.std(axis=0)
        sc_scaler_std[sc_scaler_std < 1e-10] = 1.0

    fc_z = (fc_upper - fc_scaler_mean) / fc_scaler_std
    sc_z = (sc_upper - sc_scaler_mean) / sc_scaler_std

    h = fc_z * sc_z  # elementwise interaction

    X_sfc = np.zeros((n_subj, n_pairs))
    valid_mask = edge_pair_indices >= 0
    valid_edges = np.where(valid_mask)[0]
    valid_pairs = edge_pair_indices[valid_edges]

    for p in range(n_pairs):
        mask_p = valid_pairs == p
        if mask_p.sum() > 0:
            X_sfc[:, p] = h[:, valid_edges[mask_p]].mean(axis=1)

    return X_sfc, fc_scaler_mean, fc_scaler_std, sc_scaler_mean, sc_scaler_std


def compute_sfc_pair_features_for_split(
    fc_train: np.ndarray,
    sc_train: np.ndarray,
    fc_test: np.ndarray,
    sc_test: np.ndarray,
    edge_pair_indices: np.ndarray,
    n_pairs: int,
) -> tuple:
    """Compute SFC features with training-fit scalers applied to both train and test.

    Returns
    -------
    X_sfc_train, X_sfc_test : (n_train, n_pairs), (n_test, n_pairs)
    """
    X_sfc_train, fc_mean, fc_std, sc_mean, sc_std = compute_sfc_pair_features(
        fc_train, sc_train, edge_pair_indices, n_pairs, fit_scalers=True
    )
    X_sfc_test, _, _, _, _ = compute_sfc_pair_features(
        fc_test, sc_test, edge_pair_indices, n_pairs,
        fc_scaler_mean=fc_mean, fc_scaler_std=fc_std,
        sc_scaler_mean=sc_mean, sc_scaler_std=sc_std,
        fit_scalers=False,
    )
    return X_sfc_train, X_sfc_test


def get_valid_pair_mask(edge_pair_indices: np.ndarray) -> np.ndarray:
    """Return boolean mask of edges that map to a valid system pair."""
    return edge_pair_indices >= 0


def summarize_pair_features(
    X_sfc: np.ndarray,
    pair_names: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return mean/std of each pair feature, sorted by mean absolute value."""
    means = np.abs(X_sfc).mean(axis=0)
    stds = X_sfc.std(axis=0)
    df = pd.DataFrame({
        "pair_name": pair_names,
        "mean_abs": means,
        "std": stds,
    }).sort_values("mean_abs", ascending=False)
    return df.head(top_k)
