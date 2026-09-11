"""PG-MT-BCR: Prior-Guided Multi-Task Bilinear Connectome Regression.

Low-rank bilinear model with LLM ROI prior weighting.
Shared rank=2, task-specific rank=1 per modality.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class BCRFitResult:
    """Result of fitting PG-MT-BCR or its ablations."""
    alpha_WM: float
    alpha_FI: float
    train_loss: float
    converged: bool
    n_steps: int
    U_fc_shared: np.ndarray   # (116, 2)
    U_fc_WM: np.ndarray       # (116, 1)
    U_fc_FI: np.ndarray       # (116, 1)
    U_sc_shared: np.ndarray   # (116, 2)
    U_sc_WM: np.ndarray       # (116, 1)
    U_sc_FI: np.ndarray       # (116, 1)
    amp_fc_shared: np.ndarray  # (2,)
    amp_fc_WM: float
    amp_fc_FI: float
    amp_sc_shared: np.ndarray  # (2,)
    amp_sc_WM: float
    amp_sc_FI: float
    intercept_WM: float
    intercept_FI: float
    residual_std_WM: float
    residual_std_FI: float


class PGMTBCR(nn.Module):
    """Prior-Guided Multi-Task Bilinear Connectome Regression."""

    def __init__(
        self,
        n_rois: int = 116,
        shared_rank: int = 2,
        specific_rank: int = 1,
        prior_shared: Optional[torch.Tensor] = None,
        prior_WM: Optional[torch.Tensor] = None,
        prior_FI: Optional[torch.Tensor] = None,
        lambda_amp: float = 1.0,
        lambda_prior: float = 0.1,
        epsilon: float = 1e-3,
        gamma: float = 0.5,
        use_prior: bool = True,
    ):
        super().__init__()
        self.n_rois = n_rois
        self.shared_rank = shared_rank
        self.specific_rank = specific_rank
        self.lambda_amp = lambda_amp
        self.lambda_prior = lambda_prior
        self.epsilon = epsilon
        self.gamma = gamma
        self.use_prior = use_prior

        # Factor matrices: (n_rois, rank)
        self.U_fc_shared = nn.Parameter(torch.randn(n_rois, shared_rank) * 0.01)
        self.U_fc_WM = nn.Parameter(torch.randn(n_rois, specific_rank) * 0.01)
        self.U_fc_FI = nn.Parameter(torch.randn(n_rois, specific_rank) * 0.01)
        self.U_sc_shared = nn.Parameter(torch.randn(n_rois, shared_rank) * 0.01)
        self.U_sc_WM = nn.Parameter(torch.randn(n_rois, specific_rank) * 0.01)
        self.U_sc_FI = nn.Parameter(torch.randn(n_rois, specific_rank) * 0.01)

        # Amplitude coefficients
        self.amp_fc_shared = nn.Parameter(torch.zeros(shared_rank))
        self.amp_fc_WM = nn.Parameter(torch.tensor(0.0))
        self.amp_fc_FI = nn.Parameter(torch.tensor(0.0))
        self.amp_sc_shared = nn.Parameter(torch.zeros(shared_rank))
        self.amp_sc_WM = nn.Parameter(torch.tensor(0.0))
        self.amp_sc_FI = nn.Parameter(torch.tensor(0.0))

        # Intercepts (unpenalized)
        self.intercept_WM = nn.Parameter(torch.tensor(0.0))
        self.intercept_FI = nn.Parameter(torch.tensor(0.0))

        # Priors (not learnable)
        if prior_shared is not None:
            self.register_buffer('prior_shared', prior_shared)
        else:
            self.register_buffer('prior_shared', torch.ones(n_rois) / n_rois)
        if prior_WM is not None:
            self.register_buffer('prior_WM', prior_WM)
        else:
            self.register_buffer('prior_WM', torch.ones(n_rois) / n_rois)
        if prior_FI is not None:
            self.register_buffer('prior_FI', prior_FI)
        else:
            self.register_buffer('prior_FI', torch.ones(n_rois) / n_rois)

        # Compute D matrices
        self._compute_D_matrices()

    def _compute_D_matrices(self):
        """Compute prior-weighted diagonal matrices D(p)."""
        def _compute_D(p):
            raw = (self.epsilon + p) ** (-self.gamma)
            D = raw / raw.mean()
            return torch.diag(D)
        self.D_shared = _compute_D(self.prior_shared)
        self.D_WM = _compute_D(self.prior_WM)
        self.D_FI = _compute_D(self.prior_FI)

    def _normalize_factors(self):
        """Normalize factor columns to unit norm, then orthogonalize shared."""
        with torch.no_grad():
            for name in ['U_fc_shared', 'U_fc_WM', 'U_fc_FI',
                         'U_sc_shared', 'U_sc_WM', 'U_sc_FI']:
                param = getattr(self, name)
                norms = torch.norm(param, dim=0, keepdim=True).clamp(min=1e-12)
                param.div_(norms)

            # Gram-Schmidt on shared factors
            for prefix in ['U_fc_shared', 'U_sc_shared']:
                U = getattr(self, prefix)
                # Column 0 is already unit norm
                # Column 1: subtract projection onto column 0
                u0 = U[:, 0:1]
                u1 = U[:, 1:2]
                u1 = u1 - u0 * (u0.t() @ u1)
                u1 = u1 / torch.norm(u1).clamp(min=1e-12)
                U[:, 0:1] = u0
                U[:, 1:2] = u1

    def component_score(self, U: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Compute u^T M u for each column of U.

        U: (n_rois, rank)
        M: (batch, n_rois, n_rois)
        Returns: (batch, rank)
        """
        # M @ U: (batch, n_rois, rank)
        MU = torch.bmm(M, U.unsqueeze(0).expand(M.shape[0], -1, -1))
        # U^T @ (M @ U): (batch, rank, rank) -> take diagonal
        # For each sample, compute u_k^T M u_k
        scores = torch.sum(U.unsqueeze(0) * MU, dim=1)  # (batch, rank)
        return scores

    def forward(self, X_fc: torch.Tensor, X_sc: torch.Tensor):
        """Forward pass.

        X_fc: (batch, 116, 116) standardized FC
        X_sc: (batch, 116, 116) standardized SC
        Returns: (batch, 2) predictions [WM, FI]
        """
        self._normalize_factors()

        # Shared component scores
        g_fc_shared = self.component_score(self.U_fc_shared, X_fc)  # (batch, 2)
        g_sc_shared = self.component_score(self.U_sc_shared, X_sc)  # (batch, 2)

        # Task-specific component scores
        g_fc_WM = self.component_score(self.U_fc_WM, X_fc)  # (batch, 1)
        g_fc_FI = self.component_score(self.U_fc_FI, X_fc)  # (batch, 1)
        g_sc_WM = self.component_score(self.U_sc_WM, X_sc)  # (batch, 1)
        g_sc_FI = self.component_score(self.U_sc_FI, X_sc)  # (batch, 1)

        # WM prediction
        r_WM = (self.intercept_WM
                + (g_fc_shared @ self.amp_fc_shared)
                + g_fc_WM.squeeze(-1) * self.amp_fc_WM
                + (g_sc_shared @ self.amp_sc_shared)
                + g_sc_WM.squeeze(-1) * self.amp_sc_WM)

        # FI prediction
        r_FI = (self.intercept_FI
                + (g_fc_shared @ self.amp_fc_shared)
                + g_fc_FI.squeeze(-1) * self.amp_fc_FI
                + (g_sc_shared @ self.amp_sc_shared)
                + g_sc_FI.squeeze(-1) * self.amp_sc_FI)

        return torch.stack([r_WM, r_FI], dim=-1)

    def compute_loss(self, X_fc, X_sc, r_WM, r_FI):
        """Compute training loss.

        r_WM, r_FI: (batch,) residual targets
        """
        preds = self.forward(X_fc, X_sc)
        r_WM_pred = preds[:, 0]
        r_FI_pred = preds[:, 1]

        # Data loss
        loss_WM = torch.mean((r_WM - r_WM_pred) ** 2)
        loss_FI = torch.mean((r_FI - r_FI_pred) ** 2)
        loss_data = loss_WM + loss_FI

        # Amplitude penalty
        loss_amp = (torch.sum(self.amp_fc_shared ** 2)
                    + self.amp_fc_WM ** 2 + self.amp_fc_FI ** 2
                    + torch.sum(self.amp_sc_shared ** 2)
                    + self.amp_sc_WM ** 2 + self.amp_sc_FI ** 2)

        # Prior penalty
        loss_prior = 0.0
        if self.use_prior and self.lambda_prior > 0:
            # Shared factors
            for U in [self.U_fc_shared, self.U_sc_shared]:
                for k in range(self.shared_rank):
                    uk = U[:, k]
                    loss_prior = loss_prior + uk @ self.D_shared @ uk
            # WM-specific
            for U in [self.U_fc_WM, self.U_sc_WM]:
                uk = U[:, 0]
                loss_prior = loss_prior + uk @ self.D_WM @ uk
            # FI-specific
            for U in [self.U_fc_FI, self.U_sc_FI]:
                uk = U[:, 0]
                loss_prior = loss_prior + uk @ self.D_FI @ uk

        total = loss_data + self.lambda_amp * loss_amp + self.lambda_prior * loss_prior
        return total, loss_data, loss_amp, loss_prior


def fit_bcr(
    X_fc_train: np.ndarray,
    X_sc_train: np.ndarray,
    r_WM: np.ndarray,
    r_FI: np.ndarray,
    prior_shared: np.ndarray,
    prior_WM: np.ndarray,
    prior_FI: np.ndarray,
    lambda_amp: float = 1.0,
    lambda_prior: float = 0.0,
    lr: float = 0.01,
    max_steps: int = 1500,
    min_steps: int = 200,
    patience: int = 75,
    rel_tol: float = 1e-7,
    grad_clip: float = 5.0,
    n_restarts: int = 2,
    seed: int = 0,
    device: str = 'cpu',
) -> BCRFitResult:
    """Fit PG-MT-BCR (or MT-BCR-NP if lambda_prior=0)."""
    n = X_fc_train.shape[0]
    use_prior = lambda_prior > 0

    # Convert to torch
    X_fc_t = torch.tensor(X_fc_train, dtype=torch.float64, device=device)
    X_sc_t = torch.tensor(X_sc_train, dtype=torch.float64, device=device)
    r_WM_t = torch.tensor(r_WM, dtype=torch.float64, device=device)
    r_FI_t = torch.tensor(r_FI, dtype=torch.float64, device=device)
    p_shared = torch.tensor(prior_shared, dtype=torch.float64, device=device)
    p_WM = torch.tensor(prior_WM, dtype=torch.float64, device=device)
    p_FI = torch.tensor(prior_FI, dtype=torch.float64, device=device)

    best_loss = float('inf')
    best_result = None

    for restart in range(n_restarts):
        torch.manual_seed(seed + restart * 1000)
        model = PGMTBCR(
            n_rois=116, shared_rank=2, specific_rank=1,
            prior_shared=p_shared, prior_WM=p_WM, prior_FI=p_FI,
            lambda_amp=lambda_amp, lambda_prior=lambda_prior,
            use_prior=use_prior,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        prev_loss = float('inf')
        no_improve = 0
        converged = False
        n_steps = 0

        for step in range(max_steps):
            optimizer.zero_grad()
            loss, _, _, _ = model.compute_loss(X_fc_t, X_sc_t, r_WM_t, r_FI_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            cur_loss = loss.item()
            n_steps = step + 1

            if step >= min_steps:
                if abs(prev_loss - cur_loss) / (abs(prev_loss) + 1e-12) < rel_tol:
                    no_improve += 1
                    if no_improve >= patience:
                        converged = True
                        break
                else:
                    no_improve = 0
            prev_loss = cur_loss

        # Extract results
        with torch.no_grad():
            model._normalize_factors()
            final_loss, loss_data, loss_amp, loss_prior = model.compute_loss(
                X_fc_t, X_sc_t, r_WM_t, r_FI_t
            )

        if final_loss.item() < best_loss:
            best_loss = final_loss.item()
            best_result = BCRFitResult(
                alpha_WM=0.0, alpha_FI=0.0,
                train_loss=final_loss.item(),
                converged=converged,
                n_steps=n_steps,
                U_fc_shared=model.U_fc_shared.cpu().numpy().copy(),
                U_fc_WM=model.U_fc_WM.cpu().numpy().copy(),
                U_fc_FI=model.U_fc_FI.cpu().numpy().copy(),
                U_sc_shared=model.U_sc_shared.cpu().numpy().copy(),
                U_sc_WM=model.U_sc_WM.cpu().numpy().copy(),
                U_sc_FI=model.U_sc_FI.cpu().numpy().copy(),
                amp_fc_shared=model.amp_fc_shared.cpu().numpy().copy(),
                amp_fc_WM=model.amp_fc_WM.item(),
                amp_fc_FI=model.amp_fc_FI.item(),
                amp_sc_shared=model.amp_sc_shared.cpu().numpy().copy(),
                amp_sc_WM=model.amp_sc_WM.item(),
                amp_sc_FI=model.amp_sc_FI.item(),
                intercept_WM=model.intercept_WM.item(),
                intercept_FI=model.intercept_FI.item(),
                residual_std_WM=float(r_WM.std()),
                residual_std_FI=float(r_FI.std()),
            )

    return best_result


def predict_bcr(
    fit: BCRFitResult,
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    alpha_WM: float = 0.0,
    alpha_FI: float = 0.0,
) -> tuple:
    """Predict using fitted BCR model.

    Returns: (pred_WM, pred_FI) - final predictions incorporating R0 residuals.
    """
    n = X_fc.shape[0]
    # Build U matrices
    U_fc_shared = fit.U_fc_shared  # (116, 2)
    U_fc_WM = fit.U_fc_WM          # (116, 1)
    U_fc_FI = fit.U_fc_FI          # (116, 1)
    U_sc_shared = fit.U_sc_shared
    U_sc_WM = fit.U_sc_WM
    U_sc_FI = fit.U_sc_FI

    def component_score(U, M):
        # U: (116, rank), M: (n, 116, 116)
        # M @ U: (n, 116, rank)
        MU = np.einsum('nij,jk->nik', M, U)
        # u_k^T M u_k = sum_j U[j,k] * MU[n,j,k]
        return np.sum(U[np.newaxis, :, :] * MU, axis=1)  # (n, rank)

    g_fc_shared = component_score(U_fc_shared, X_fc)
    g_sc_shared = component_score(U_sc_shared, X_sc)
    g_fc_WM = component_score(U_fc_WM, X_fc)
    g_fc_FI = component_score(U_fc_FI, X_fc)
    g_sc_WM = component_score(U_sc_WM, X_sc)
    g_sc_FI = component_score(U_sc_FI, X_sc)

    r_WM = (fit.intercept_WM
            + g_fc_shared @ fit.amp_fc_shared
            + g_fc_WM[:, 0] * fit.amp_fc_WM
            + g_sc_shared @ fit.amp_sc_shared
            + g_sc_WM[:, 0] * fit.amp_sc_WM)

    r_FI = (fit.intercept_FI
            + g_fc_shared @ fit.amp_fc_shared
            + g_fc_FI[:, 0] * fit.amp_fc_FI
            + g_sc_shared @ fit.amp_sc_shared
            + g_sc_FI[:, 0] * fit.amp_sc_FI)

    return r_WM * alpha_WM, r_FI * alpha_FI
