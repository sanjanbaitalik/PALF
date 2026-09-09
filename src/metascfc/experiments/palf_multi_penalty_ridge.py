"""Multi-Penalty Ridge (MPR) solver for Phase 2B.

Joint FC+SC regression with separate penalties for:
- generic FC edges (group G)
- semantic-priority FC edges (group A)
- SC edges (group S)

Objective:
    min ||y - X_G β_G - X_A β_A - X_S β_S||²
      + λ (||β_G||² + r_A ||β_A||² + r_S ||β_S||²)

Subject-space kernel formulation for efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler


N_ROI = 116
N_EDGE = N_ROI * (N_ROI - 1) // 2  # 6670


@dataclass
class EdgeGrouping:
    """Disjoint edge groups for one target."""
    semantic_indices: np.ndarray  # group A
    generic_indices: np.ndarray   # group G
    top_rois: np.ndarray          # top-K ROI indices
    top_roi_names: List[str]      # top-K ROI names
    n_semantic: int
    n_generic: int

    def validate(self) -> None:
        assert self.n_semantic + self.n_generic == N_EDGE
        assert len(set(self.semantic_indices.tolist()) & set(self.generic_indices.tolist())) == 0
        assert len(self.semantic_indices) == self.n_semantic
        assert len(self.generic_indices) == self.n_generic


def build_edge_grouping(
    roi_prior: np.ndarray,
    roi_names: List[str],
    top_k: int = 10,
    n_rois: int = N_ROI,
) -> EdgeGrouping:
    """Build disjoint semantic/generic FC edge groups from ROI prior.

    Parameters
    ----------
    roi_prior : (n_rois,) prior scores per ROI.
    roi_names : list of ROI names.
    top_k : number of top ROIs for semantic group.
    n_rois : atlas size.

    Returns
    -------
    EdgeGrouping with validated disjoint/exhaustive groups.
    """
    roi_prior = np.asarray(roi_prior, dtype=np.float64).ravel()
    assert len(roi_prior) == n_rois

    # Sort by prior score descending
    sorted_idx = np.argsort(roi_prior)[::-1]
    top_rois = sorted(sorted_idx[:top_k].tolist())
    top_roi_names = [roi_names[i] for i in top_rois]
    top_set = set(top_rois)

    # Build all upper-triangle edge indices
    iu = np.triu_indices(n_rois, k=1)
    all_i, all_j = iu[0], iu[1]

    # Semantic group: edges where at least one endpoint is in top-K
    sem_mask = np.array([i in top_set or j in top_set
                         for i, j in zip(all_i, all_j)])
    semantic_indices = np.where(sem_mask)[0]
    generic_indices = np.where(~sem_mask)[0]

    grouping = EdgeGrouping(
        semantic_indices=semantic_indices,
        generic_indices=generic_indices,
        top_rois=np.array(top_rois),
        top_roi_names=top_roi_names,
        n_semantic=len(semantic_indices),
        n_generic=len(generic_indices),
    )
    grouping.validate()
    return grouping


def solve_mpr_kernel(
    X_G: np.ndarray,
    X_A: np.ndarray,
    X_S: np.ndarray,
    y: np.ndarray,
    lam: float,
    r_A: float,
    r_S: float,
    fit_intercept: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Solve multi-penalty Ridge in subject space using kernel formulation.

    Parameters
    ----------
    X_G : (n, p_G) standardized generic FC features.
    X_A : (n, p_A) standardized semantic FC features.
    X_S : (n, p_S) standardized SC features.
    y : (n,) target values (already centered if fit_intercept).
    lam : overall regularization.
    r_A : semantic penalty ratio.
    r_S : SC penalty ratio.
    fit_intercept : if True, y is assumed already centered.

    Returns
    -------
    train_pred : (n,) in-sample predictions (original scale if intercept added).
    info : dict with kernel matrices and coefficients.
    """
    n = len(y)
    y = np.asarray(y, dtype=np.float64).ravel()

    # Compute group Gram matrices
    K_G = X_G @ X_G.T
    K_A = X_A @ X_A.T
    K_S = X_S @ X_S.T

    # Combined kernel
    K_0 = K_G + (1.0 / r_A) * K_A + (1.0 / r_S) * K_S

    # Solve: alpha = (K_0 + lambda * I)^{-1} y
    K_reg = K_0 + lam * np.eye(n)
    alpha = np.linalg.solve(K_reg, y)

    # Training predictions
    train_pred = K_0 @ alpha

    # Recover primal coefficients
    # beta = X^T alpha / scaling, but for standardized features:
    # beta_G = X_G^T alpha, beta_A = X_A^T alpha / r_A, beta_S = X_S^T alpha / r_S
    beta_G = X_G.T @ alpha
    beta_A = X_A.T @ alpha / r_A
    beta_S = X_S.T @ alpha / r_S

    return train_pred, {
        "alpha": alpha,
        "K_0": K_0,
        "beta_G": beta_G,
        "beta_A": beta_A,
        "beta_S": beta_S,
    }


def predict_mpr_kernel(
    X_G_test: np.ndarray,
    X_A_test: np.ndarray,
    X_S_test: np.ndarray,
    X_G_train: np.ndarray,
    X_A_train: np.ndarray,
    X_S_train: np.ndarray,
    alpha: np.ndarray,
    r_A: float,
    r_S: float,
) -> np.ndarray:
    """Predict on new data using dual coefficients.

    Uses cross-kernels with the same training standardization.
    """
    # Cross-kernels
    K_G_cross = X_G_test @ X_G_train.T
    K_A_cross = X_A_test @ X_A_train.T
    K_S_cross = X_S_test @ X_S_train.T

    K_cross = K_G_cross + (1.0 / r_A) * K_A_cross + (1.0 / r_S) * K_S_cross
    return K_cross @ alpha


def compute_beta_fc_full(
    beta_G: np.ndarray,
    beta_A: np.ndarray,
    generic_indices: np.ndarray,
    semantic_indices: np.ndarray,
    n_edges: int = N_EDGE,
) -> np.ndarray:
    """Reconstruct full 6670-dimensional FC coefficient vector."""
    beta_full = np.zeros(n_edges, dtype=np.float64)
    beta_full[generic_indices] = beta_G
    beta_full[semantic_indices] = beta_A
    return beta_full


def validate_primal_reconstruction(
    X_G: np.ndarray,
    X_A: np.ndarray,
    X_S: np.ndarray,
    y: np.ndarray,
    beta_G: np.ndarray,
    beta_A: np.ndarray,
    beta_S: np.ndarray,
    train_pred: np.ndarray,
    y_mean: float = 0.0,
    tol: float = 1e-8,
) -> float:
    """Validate that primal coefficients reproduce kernel predictions.

    Returns max absolute error.
    """
    pred_primal = (
        X_G @ beta_G + X_A @ beta_A + X_S @ beta_S + y_mean
    )
    return float(np.max(np.abs(pred_primal - train_pred)))
