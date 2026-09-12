"""OpenChallenge: CKE (Calibrated Kernel Ensemble) kernel machinery.

Locked method (MODEL_PROPOSAL_LOCK.md sha daff28b3...):
RBF kernel ridge regression on standardized FC+SC edges, late-fused with the
frozen corrected R0 baseline by an inner-CV-calibrated convex weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

N_ROI = 116
N_EDGE = N_ROI * (N_ROI - 1) // 2  # 6670
N_FEAT = 2 * N_EDGE                # 13340
IU = np.triu_indices(N_ROI, k=1)


@dataclass
class KRRFit:
    kernel: str          # "rbf" | "linear"
    lam: float
    sigma: float         # bandwidth (rbf only; nan for linear)
    y_mean: float
    alpha: np.ndarray    # dual coefficients
    X_train: np.ndarray  # training features (kept for kernel evals / gradients)


def median_heuristic(X: np.ndarray, max_pairs: int = 200) -> float:
    """Median pairwise distance on a deterministic subset."""
    n = X.shape[0]
    idx = np.arange(min(n, max_pairs))
    G = X[idx] @ X[idx].T
    d2 = np.maximum(np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G, 0.0)
    iu = np.triu_indices(len(idx), k=1)
    d = np.sqrt(d2[iu[0], iu[1]])
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if len(d) else 1.0


def _dist2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    G = A @ B.T
    d2 = (np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * G, 0.0))
    return d2


def fit_krr(Xtr: np.ndarray, ytr: np.ndarray, kernel: str, lam: float,
            mult: float = 1.0, sigma_base: Optional[float] = None) -> KRRFit:
    """Fit kernel ridge. sigma = mult * median heuristic (train scope only)."""
    ym = float(ytr.mean())
    yc = ytr - ym
    if kernel == "rbf":
        base = sigma_base if sigma_base is not None else median_heuristic(Xtr)
        sigma = max(mult * base, 1e-8)
        K = np.exp(-_dist2(Xtr, Xtr) / (2 * sigma ** 2))
    elif kernel == "linear":
        sigma = float("nan")
        K = Xtr @ Xtr.T
    else:
        raise ValueError(kernel)
    n = len(ytr)
    alpha = np.linalg.solve(K + lam * np.eye(n), yc)
    return KRRFit(kernel=kernel, lam=float(lam), sigma=sigma, y_mean=ym,
                  alpha=alpha, X_train=Xtr.copy())


def krr_predict(fit: KRRFit, X: np.ndarray) -> np.ndarray:
    if fit.kernel == "rbf":
        K = np.exp(-_dist2(X, fit.X_train) / (2 * fit.sigma ** 2))
    else:
        K = X @ fit.X_train.T
    return K @ fit.alpha + fit.y_mean


def krr_gradient(fit: KRRFit, X: np.ndarray) -> np.ndarray:
    """Analytic gradient of the kernel prediction w.r.t. features.

    Returns an array aligned with X: (n_subjects, n_features).
    g_s = d f(x_s) / d x  (row-wise)
    """
    Xtr = fit.X_train
    if fit.kernel == "rbf":
        K = np.exp(-_dist2(X, Xtr) / (2 * fit.sigma ** 2))
        c = K @ fit.alpha                                   # (n,)
        M1 = (K * fit.alpha[None, :]) @ Xtr                  # (n, p)
        g = -(X * c[:, None] - M1) / (fit.sigma ** 2)
    else:
        w = fit.X_train.T @ fit.alpha                        # (p,)
        g = np.tile(w[None, :], (X.shape[0], 1))
    return g


def attribution_maps(fit: KRRFit, Xtr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-fit attribution: importance (mean |grad| over train) and signed mean."""
    G = krr_gradient(fit, Xtr)
    return G.__abs__().mean(axis=0), G.mean(axis=0)


def roi_scores(importance: np.ndarray) -> np.ndarray:
    """ROI score = sum of |importance| over incident edges (both blocks)."""
    out = np.zeros(N_ROI)
    for block in (0, 1):
        imp = importance[block * N_EDGE:(block + 1) * N_EDGE]
        np.add.at(out, IU[0], imp)
        np.add.at(out, IU[1], imp)
    return out


def edge_mask_for_rois(roi_set) -> np.ndarray:
    """Boolean mask over the 13,340-dim feature vector for edges incident to ROIs."""
    roi_set = list(int(r) for r in roi_set)
    m_fc = np.isin(IU[0], roi_set) | np.isin(IU[1], roi_set)
    return np.concatenate([m_fc, m_fc])
