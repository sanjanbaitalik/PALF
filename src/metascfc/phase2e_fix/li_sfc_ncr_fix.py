"""LI-SFC-NCR expert with correct dual prediction."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge


def build_pair_laplacian(
    prior: np.ndarray,
    selected: np.ndarray,
    system_names: np.ndarray,
) -> Optional[np.ndarray]:
    """Build pair-graph Laplacian for NCR regularization."""
    n = len(selected)
    if n < 2:
        return None

    # Build adjacency matrix from prior
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            # Connectivity based on shared systems
            si = system_names[selected[i]]
            sj = system_names[selected[j]]

            # Check if pairs share a system
            parts_i = set(si.split("-")) if "-" in si else {si}
            parts_j = set(sj.split("-")) if "-" in sj else {sj}
            shared = parts_i & parts_j

            if shared:
                # Weight by geometric mean of prior scores
                w = np.sqrt(prior[selected[i]] * prior[selected[j]])
                W[i, j] = w
                W[j, i] = w

    # Degree matrix
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Symmetric normalization
    d_inv_sqrt = np.zeros(n)
    nonzero = np.diag(D) > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(np.diag(D)[nonzero])
    L_norm = np.diag(d_inv_sqrt) @ L @ np.diag(d_inv_sqrt)

    return L_norm


def ridge_expert_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Standard Ridge prediction."""
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def ncr_expert_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    ratio: float,
    eigvals_L: np.ndarray,
    eigvecs_L: np.ndarray,
) -> np.ndarray:
    """NCR prediction using dual formulation with correct broadcasting."""
    lambda_L = ratio * alpha
    n = len(y_train)

    # Compute dual coefficients
    quad = np.zeros((n, n))
    for j in range(eigvecs_L.shape[1]):
        Zj = X_train @ eigvecs_L[:, j]
        denom = alpha + lambda_L * eigvals_L[j]
        if denom < 1e-12:
            continue
        quad += np.outer(Zj, Zj) / denom

    try:
        alpha_dual = np.linalg.solve(np.eye(n) + quad, y_train)
    except np.linalg.LinAlgError:
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    # Predict: beta_j = (Z_train_j @ alpha_dual) / denom
    pred = np.zeros(len(X_test))
    for j in range(eigvecs_L.shape[1]):
        Z_train_j = X_train @ eigvecs_L[:, j]
        Z_test_j = X_test @ eigvecs_L[:, j]
        denom = alpha + lambda_L * eigvals_L[j]
        if denom < 1e-12:
            continue
        beta_j = (Z_train_j @ alpha_dual) / denom
        pred += Z_test_j * beta_j

    return pred


def hierarchical_fusion_weights(
    y: np.ndarray,
    oof_baseline: np.ndarray,
    oof_expert: np.ndarray,
    eta_grid: np.ndarray = None,
) -> Tuple[float, float]:
    """Find optimal fusion weights for baseline and expert."""
    if eta_grid is None:
        eta_grid = np.arange(0, 1.05, 0.05)

    best_eta = 0.0
    best_r = -np.inf

    for eta in eta_grid:
        fused = (1 - eta) * oof_baseline + eta * oof_expert
        r = np.corrcoef(y, fused)[0, 1]
        if r > best_r:
            best_r = r
            best_eta = eta

    return best_eta, best_r
