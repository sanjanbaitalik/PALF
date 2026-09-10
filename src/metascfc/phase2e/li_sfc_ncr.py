"""
LI-SFC-NCR: LLM Interaction-Prior Structure-Function Coupling NCR.

Ridge and NCR experts on system-pair coupling features with LLM prior guidance.
"""
import numpy as np
import warnings
from typing import Optional, NamedTuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class PairExpertResult(NamedTuple):
    beta_standardized: np.ndarray
    beta_original: np.ndarray
    intercept: float
    lambda_R: float
    lambda_L: float
    alpha: float  # fusion weight (not Ridge alpha)
    selected_pairs: np.ndarray
    selected_indices: np.ndarray
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    target_mean: float
    target_std: float


def build_pair_laplacian(
    pair_prior: np.ndarray,
    selected_indices: np.ndarray,
    system_names: np.ndarray,
) -> Optional[np.ndarray]:
    """Build symmetric-normalized Laplacian for selected pair features.

    Two pair-features are adjacent if they share at least one system.
    Edge weights: w_uv = sqrt(q_u * q_v).

    Returns None if no valid Laplacian can be built.
    """
    n = len(selected_indices)
    if n <= 1:
        return None

    # Parse system names for selected pairs
    selected_names = [system_names[i] for i in selected_indices]

    def _parse_systems(pair_name):
        if "-" in pair_name:
            return set(pair_name.split("-"))
        return {pair_name}

    adj = np.zeros((n, n))
    for i in range(n):
        sys_i = _parse_systems(selected_names[i])
        for j in range(i + 1, n):
            sys_j = _parse_systems(selected_names[j])
            if sys_i & sys_j:  # share at least one system
                qi = pair_prior[selected_indices[i]]
                qj = pair_prior[selected_indices[j]]
                w = np.sqrt(max(qi, 0) * max(qj, 0))
                adj[i, j] = w
                adj[j, i] = w

    # Check if graph is connected enough
    deg = adj.sum(axis=1)
    if (deg == 0).all():
        return None

    # Symmetric normalization: L = I - D^{-1/2} A D^{-1/2}
    deg_sqrt = np.sqrt(deg + 1e-10)
    D_inv_sqrt = np.diag(1.0 / deg_sqrt)
    L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt

    # Make symmetric PSD
    L = 0.5 * (L + L.T)
    eigvals = np.linalg.eigvalsh(L)
    if eigvals.min() < -1e-10:
        L += (-eigvals.min() + 1e-6) * np.eye(n)

    return L


def fit_sfc_ridge_expert(
    X_sfc: np.ndarray,
    y: np.ndarray,
    alpha_grid: Optional[np.ndarray] = None,
    n_inner_folds: int = 3,
    selected_indices: Optional[np.ndarray] = None,
) -> PairExpertResult:
    """Fit Ridge expert on system-pair coupling features.

    If selected_indices is None, uses all valid (nonzero) pairs.
    """
    if alpha_grid is None:
        alpha_grid = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

    if selected_indices is None:
        nonzero_mask = np.abs(X_sfc).sum(axis=0) > 0
        selected_indices = np.where(nonzero_mask)[0]

    X_sel = X_sfc[:, selected_indices]
    n_pairs = X_sel.shape[1]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    # Replace NaN/inf
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std

    # Inner CV for alpha selection
    from sklearn.model_selection import KFold
    best_alpha = alpha_grid[0]
    best_score = -np.inf

    kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
    for alpha in alpha_grid:
        scores = []
        for train_idx, val_idx in kf.split(X_scaled):
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X_scaled[train_idx], y_scaled[train_idx])
            pred = model.predict(X_scaled[val_idx])
            corr = np.corrcoef(pred, y_scaled[val_idx])[0, 1]
            if np.isfinite(corr):
                scores.append(corr)
        if scores:
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha

    # Final fit
    model = Ridge(alpha=best_alpha, fit_intercept=False)
    model.fit(X_scaled, y_scaled)
    beta_scaled = model.coef_.copy()

    # Map back to original scale
    beta_original = beta_scaled / (scaler.scale_ + 1e-10) * y_std

    return PairExpertResult(
        beta_standardized=beta_scaled,
        beta_original=beta_original,
        intercept=0.0,
        lambda_R=best_alpha,
        lambda_L=0.0,
        alpha=0.0,
        selected_pairs=np.array([f"pair_{i}" for i in selected_indices]),
        selected_indices=selected_indices,
        scaler_mean=scaler.mean_,
        scaler_std=scaler.scale_,
        target_mean=y_mean,
        target_std=y_std,
    )


def fit_sfc_ncr_expert(
    X_sfc: np.ndarray,
    y: np.ndarray,
    pair_prior: np.ndarray,
    system_names: np.ndarray,
    alpha_grid: Optional[np.ndarray] = None,
    ratio_grid: Optional[np.ndarray] = None,
    n_inner_folds: int = 3,
    selected_indices: Optional[np.ndarray] = None,
) -> PairExpertResult:
    """Fit NCR expert on system-pair coupling features with LLM prior Laplacian.

    lambda_L = ratio * lambda_R
    """
    if alpha_grid is None:
        alpha_grid = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    if ratio_grid is None:
        ratio_grid = np.array([0.0, 0.1, 0.3, 1.0])

    if selected_indices is None:
        nonzero_mask = np.abs(X_sfc).sum(axis=0) > 0
        selected_indices = np.where(nonzero_mask)[0]

    X_sel = X_sfc[:, selected_indices]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std

    # Build Laplacian
    L = build_pair_laplacian(pair_prior, selected_indices, system_names)

    if L is None:
        return fit_sfc_ridge_expert(X_sfc, y, alpha_grid, n_inner_folds, selected_indices)

    # Eigendecompose Laplacian once
    eigvals_L, eigvecs_L = np.linalg.eigh(L)
    eigvals_L = np.maximum(eigvals_L, 0.0)

    # Whitened features: Z = X @ U
    Z = X_scaled @ eigvecs_L

    from sklearn.model_selection import KFold
    best_score = -np.inf
    best_alpha = alpha_grid[0]
    best_ratio = 0.0

    kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=42)

    for alpha in alpha_grid:
        for ratio in ratio_grid:
            lambda_L = ratio * alpha
            scores = []
            for train_idx, val_idx in kf.split(X_scaled):
                Z_train, Z_val = Z[train_idx], Z[val_idx]
                y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]

                # Dual solution: alpha_dual = (I + Q)^{-1} y
                # Q = sum_j z_j z_j^T / (alpha + lambda_L * mu_j) + lambda_L * I * (project)
                n_tr = len(train_idx)
                quad = np.zeros((n_tr, n_tr))
                for j in range(Z.shape[1]):
                    denom = alpha + lambda_L * eigvals_L[j]
                    if denom < 1e-12:
                        continue
                    quad += np.outer(Z_train[:, j], Z_train[:, j]) / denom

                try:
                    alpha_dual = np.linalg.solve(np.eye(n_tr) + quad, y_train)
                except np.linalg.LinAlgError:
                    continue

                # Predict on val
                pred_val = np.zeros(len(val_idx))
                for j in range(Z.shape[1]):
                    denom = alpha + lambda_L * eigvals_L[j]
                    if denom < 1e-12:
                        continue
                    pred_val += (Z_val[:, j] @ Z_train[:, j]) * alpha_dual / denom

                corr = np.corrcoef(pred_val, y_val)[0, 1]
                if np.isfinite(corr):
                    scores.append(corr)

            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
                    best_ratio = ratio

    # Final fit with best params
    lambda_L = best_ratio * best_alpha
    n = len(X_scaled)
    quad = np.zeros((n, n))
    for j in range(Z.shape[1]):
        denom = best_alpha + lambda_L * eigvals_L[j]
        if denom < 1e-12:
            continue
        quad += np.outer(Z[:, j], Z[:, j]) / denom

    try:
        alpha_dual = np.linalg.solve(np.eye(n) + quad, y_scaled)
    except np.linalg.LinAlgError:
        alpha_dual = np.zeros(n)

    # Recover primal weights
    beta_whitened = np.zeros(Z.shape[1])
    for j in range(Z.shape[1]):
        denom = best_alpha + lambda_L * eigvals_L[j]
        if denom < 1e-12:
            continue
        beta_whitened[j] = (Z[:, j] @ alpha_dual) / denom

    beta_scaled = eigvecs_L @ beta_whitened
    beta_original = beta_scaled / (scaler.scale_ + 1e-10) * y_std

    return PairExpertResult(
        beta_standardized=beta_scaled,
        beta_original=beta_original,
        intercept=0.0,
        lambda_R=best_alpha,
        lambda_L=lambda_L,
        alpha=best_ratio,
        selected_pairs=np.array([f"pair_{i}" for i in selected_indices]),
        selected_indices=selected_indices,
        scaler_mean=scaler.mean_,
        scaler_std=scaler.scale_,
        target_mean=y_mean,
        target_std=y_std,
    )


def predict_sfc_expert(
    X_sfc_test: np.ndarray,
    expert: PairExpertResult,
) -> np.ndarray:
    """Predict using a fitted SFC expert."""
    X_sel = X_sfc_test[:, expert.selected_indices]
    X_scaled = (X_sel - expert.scaler_mean) / (expert.scaler_std + 1e-10)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    pred_scaled = X_scaled @ expert.beta_standardized
    return pred_scaled * expert.target_std + expert.target_mean


def hierarchical_fusion(
    y: np.ndarray,
    base_oof: np.ndarray,
    expert_oof: np.ndarray,
    base_test: np.ndarray,
    expert_test: np.ndarray,
    eta_grid: Optional[np.ndarray] = None,
) -> dict:
    """Select eta and fuse baseline + expert predictions.

    final = (1 - eta) * base + eta * expert
    """
    if eta_grid is None:
        eta_grid = np.arange(0, 1.01, 0.05)

    best_eta = 0.0
    best_corr = -np.inf
    for eta in eta_grid:
        fused = (1 - eta) * base_oof + eta * expert_oof
        corr = np.corrcoef(fused, y)[0, 1]
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_eta = eta

    fused_test = (1 - best_eta) * base_test + best_eta * expert_test
    fused_oof = (1 - best_eta) * base_oof + best_eta * expert_oof

    return {
        "eta": best_eta,
        "fused_oof": fused_oof,
        "fused_test": fused_test,
        "oof_corr": best_corr,
    }


def search_fusion_weights(
    y: np.ndarray,
    base_oof: np.ndarray,
    expert_oof: np.ndarray,
) -> np.ndarray:
    """Search convex weight for FC/SC expert fusion."""
    best_w = 0.5
    best_corr = -np.inf
    for w in np.arange(0, 1.01, 0.05):
        fused = w * base_oof + (1 - w) * expert_oof
        corr = np.corrcoef(fused, y)[0, 1]
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_w = w
    return best_w
