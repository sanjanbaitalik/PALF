"""Phase 3A-FIX: PG-MT-BCR and ST-BCR pytorch implementation.

Correct multi-task architecture: shared spatial factors with task-specific
shared amplitudes. Frozen full-batch Adam optimizer. Factor projection after
every optimizer step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

N_ROI = 116
N_EDGE = N_ROI * (N_ROI - 1) // 2
SHARED_RANK = 2
SPECIFIC_RANK = 1

_IU = np.triu_indices(N_ROI, k=1)
_EDGE_I = torch.tensor(_IU[0], dtype=torch.long)
_EDGE_J = torch.tensor(_IU[1], dtype=torch.long)


def to_edge(X: np.ndarray) -> np.ndarray:
    """Extract upper-triangle edges from symmetric (n,116,116) matrices."""
    X = np.asarray(X)
    if X.ndim == 2:
        return X
    return X[:, _IU[0], _IU[1]].astype(np.float64)

# Frozen optimizer settings
LR = 0.01
MAX_STEPS = 1500
MIN_STEPS = 200
PATIENCE = 75
REL_TOL = 1e-7
GRAD_CLIP = 5.0
N_RESTARTS = 2

# Prior penalty
EPSILON = 1e-3
GAMMA = 0.5


@dataclass
class BCRResult:
    kind: str
    state: dict
    train_loss: float
    converged: bool
    steps: int
    restart_used: int
    restart_records: List[dict] = field(default_factory=list)
    runtime_s: float = 0.0


def prior_D(p: np.ndarray) -> np.ndarray:
    """d_i(p) = (eps+p_i)^-gamma / mean_j (eps+p_j)^-gamma (diagonal)."""
    raw = (EPSILON + np.asarray(p, dtype=np.float64)) ** (-GAMMA)
    return raw / raw.mean()


def shared_prior(p_wm: np.ndarray, p_fi: np.ndarray) -> np.ndarray:
    x = np.sqrt((np.asarray(p_wm) + 1e-6) * (np.asarray(p_fi) + 1e-6))
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


class _BCRNet(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        if kind == "MT":
            self.U_fc_shared = nn.Parameter(torch.randn(N_ROI, SHARED_RANK) * 0.01)
            self.U_sc_shared = nn.Parameter(torch.randn(N_ROI, SHARED_RANK) * 0.01)
            self.U_fc_WM = nn.Parameter(torch.randn(N_ROI, SPECIFIC_RANK) * 0.01)
            self.U_fc_FI = nn.Parameter(torch.randn(N_ROI, SPECIFIC_RANK) * 0.01)
            self.U_sc_WM = nn.Parameter(torch.randn(N_ROI, SPECIFIC_RANK) * 0.01)
            self.U_sc_FI = nn.Parameter(torch.randn(N_ROI, SPECIFIC_RANK) * 0.01)
            self.amp_fc_shared_WM = nn.Parameter(torch.zeros(SHARED_RANK))
            self.amp_fc_shared_FI = nn.Parameter(torch.zeros(SHARED_RANK))
            self.amp_sc_shared_WM = nn.Parameter(torch.zeros(SHARED_RANK))
            self.amp_sc_shared_FI = nn.Parameter(torch.zeros(SHARED_RANK))
            self.amp_fc_WM = nn.Parameter(torch.tensor(0.0))
            self.amp_fc_FI = nn.Parameter(torch.tensor(0.0))
            self.amp_sc_WM = nn.Parameter(torch.tensor(0.0))
            self.amp_sc_FI = nn.Parameter(torch.tensor(0.0))
        elif kind == "ST_WM":
            self.U_fc_WM = nn.Parameter(torch.randn(N_ROI, 3) * 0.01)
            self.U_sc_WM = nn.Parameter(torch.randn(N_ROI, 3) * 0.01)
            self.amp_fc_WM = nn.Parameter(torch.zeros(3))
            self.amp_sc_WM = nn.Parameter(torch.zeros(3))
        elif kind == "ST_FI":
            self.U_fc_FI = nn.Parameter(torch.randn(N_ROI, 3) * 0.01)
            self.U_sc_FI = nn.Parameter(torch.randn(N_ROI, 3) * 0.01)
            self.amp_fc_FI = nn.Parameter(torch.zeros(3))
            self.amp_sc_FI = nn.Parameter(torch.zeros(3))
        else:
            raise ValueError(kind)
        self.b_WM = nn.Parameter(torch.tensor(0.0))
        self.b_FI = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def scores(U: torch.Tensor, Xe: torch.Tensor) -> torch.Tensor:
        """u^T M u for symmetric zero-diagonal M, via edges (factor 2)."""
        P = U[_EDGE_I] * U[_EDGE_J]  # (6670, r)
        return 2.0 * (Xe @ P)        # (n, r)

    def forward(self, Xefc, Xesc):
        if self.kind == "MT":
            Ufc = torch.cat([self.U_fc_shared, self.U_fc_WM, self.U_fc_FI], dim=1)
            Usc = torch.cat([self.U_sc_shared, self.U_sc_WM, self.U_sc_FI], dim=1)
            gfc = self.scores(Ufc, Xefc)  # (n, 4)
            gsc = self.scores(Usc, Xesc)
            gfs = gfc[:, 0:2]; gfw = gfc[:, 2]; gff = gfc[:, 3]
            gss = gsc[:, 0:2]; gsw = gsc[:, 2]; gsf = gsc[:, 3]
            rWM = (self.b_WM + gfs @ self.amp_fc_shared_WM + gfw * self.amp_fc_WM
                   + gss @ self.amp_sc_shared_WM + gsw * self.amp_sc_WM)
            rFI = (self.b_FI + gfs @ self.amp_fc_shared_FI + gff * self.amp_fc_FI
                   + gss @ self.amp_sc_shared_FI + gsf * self.amp_sc_FI)
            return rWM, rFI
        elif self.kind == "ST_WM":
            gfc = self.scores(self.U_fc_WM, Xefc)
            gsc = self.scores(self.U_sc_WM, Xesc)
            return self.b_WM + gfc @ self.amp_fc_WM + gsc @ self.amp_sc_WM, None
        else:
            gfc = self.scores(self.U_fc_FI, Xefc)
            gsc = self.scores(self.U_sc_FI, Xesc)
            return None, self.b_FI + gfc @ self.amp_fc_FI + gsc @ self.amp_sc_FI


def _project_factors(model: _BCRNet) -> None:
    """L2-normalize factor columns; Gram-Schmidt then renormalize shared."""
    with torch.no_grad():
        if model.kind == "MT":
            for name in ["U_fc_shared", "U_sc_shared", "U_fc_WM", "U_fc_FI",
                         "U_sc_WM", "U_sc_FI"]:
                U = getattr(model, name)
                U.div_(torch.norm(U, dim=0, keepdim=True).clamp(min=1e-12))
            for name in ["U_fc_shared", "U_sc_shared"]:
                U = getattr(model, name)
                u0, u1 = U[:, 0:1], U[:, 1:2]
                u1 = u1 - u0 * (u0.t() @ u1)
                u1 = u1 / torch.norm(u1).clamp(min=1e-12)
                U[:, 0:1] = u0
                U[:, 1:2] = u1
        elif model.kind == "ST_WM":
            for name in ["U_fc_WM", "U_sc_WM"]:
                U = getattr(model, name)
                U.div_(torch.norm(U, dim=0, keepdim=True).clamp(min=1e-12))
        else:
            for name in ["U_fc_FI", "U_sc_FI"]:
                U = getattr(model, name)
                U.div_(torch.norm(U, dim=0, keepdim=True).clamp(min=1e-12))


def _amp_list(model: _BCRNet) -> List[torch.Tensor]:
    if model.kind == "MT":
        return [model.amp_fc_shared_WM, model.amp_fc_shared_FI,
                model.amp_sc_shared_WM, model.amp_sc_shared_FI,
                model.amp_fc_WM, model.amp_fc_FI, model.amp_sc_WM, model.amp_sc_FI]
    elif model.kind == "ST_WM":
        return [model.amp_fc_WM, model.amp_sc_WM]
    else:
        return [model.amp_fc_FI, model.amp_sc_FI]


def _amp_penalty(model: _BCRNet) -> torch.Tensor:
    total = torch.tensor(0.0, dtype=torch.float64)
    for a in _amp_list(model):
        total = total + torch.sum(a ** 2)
    return total


def _prior_penalty(model: _BCRNet, d_fc_sh, d_sc_sh, d_fc_wm, d_sc_wm,
                   d_fc_fi, d_sc_fi) -> torch.Tensor:
    total = torch.tensor(0.0, dtype=torch.float64)
    if model.kind == "MT":
        for k in range(SHARED_RANK):
            total = total + torch.sum(d_fc_sh * model.U_fc_shared[:, k] ** 2)
            total = total + torch.sum(d_sc_sh * model.U_sc_shared[:, k] ** 2)
        total = total + torch.sum(d_fc_wm * model.U_fc_WM[:, 0] ** 2)
        total = total + torch.sum(d_sc_wm * model.U_sc_WM[:, 0] ** 2)
        total = total + torch.sum(d_fc_fi * model.U_fc_FI[:, 0] ** 2)
        total = total + torch.sum(d_sc_fi * model.U_sc_FI[:, 0] ** 2)
    elif model.kind == "ST_WM":
        for k in range(3):
            total = total + torch.sum(d_fc_wm * model.U_fc_WM[:, k] ** 2)
            total = total + torch.sum(d_sc_wm * model.U_sc_WM[:, k] ** 2)
    else:
        for k in range(3):
            total = total + torch.sum(d_fc_fi * model.U_fc_FI[:, k] ** 2)
            total = total + torch.sum(d_sc_fi * model.U_sc_FI[:, k] ** 2)
    return total


def train_bcr(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    z_r_WM: Optional[np.ndarray],
    z_r_FI: Optional[np.ndarray],
    prior_sh: np.ndarray,
    prior_wm: np.ndarray,
    prior_fi: np.ndarray,
    lambda_amp: float,
    lambda_prior: float,
    kind: str = "MT",
    seed: int = 0,
    diagnostics: Optional[List[dict]] = None,
    diag_context: Optional[dict] = None,
) -> BCRResult:
    """Train one BCR model with frozen Adam + projection + 2 restarts."""
    t_start = time.time()
    Xfc = torch.tensor(to_edge(X_fc), dtype=torch.float64)
    Xsc = torch.tensor(to_edge(X_sc), dtype=torch.float64)
    zWM = (torch.tensor(np.asarray(z_r_WM, dtype=np.float64), dtype=torch.float64)
           if z_r_WM is not None else None)
    zFI = (torch.tensor(np.asarray(z_r_FI, dtype=np.float64), dtype=torch.float64)
           if z_r_FI is not None else None)

    # D matrices as vectors
    d_sh = torch.tensor(prior_D(prior_sh), dtype=torch.float64)
    d_wm = torch.tensor(prior_D(prior_wm), dtype=torch.float64)
    d_fi = torch.tensor(prior_D(prior_fi), dtype=torch.float64)
    d_zero = torch.zeros(N_ROI, dtype=torch.float64)

    best_loss = float("inf")
    best_state = None
    best_conv = False
    best_steps = 0
    best_restart = 0
    records: List[dict] = []

    for restart in range(N_RESTARTS):
        torch.manual_seed(int(seed) * 100003 + restart * 7919)
        model = _BCRNet(kind).to(dtype=torch.float64)
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        prev_best = float("inf")
        no_improve = 0
        converged = False
        steps = 0
        loss_val = float("inf")

        for step in range(MAX_STEPS):
            opt.zero_grad()
            rWM, rFI = model.forward(Xfc, Xsc)
            loss = torch.tensor(0.0, dtype=torch.float64)
            if kind in ("MT", "ST_WM"):
                loss = loss + torch.mean((zWM - rWM) ** 2)
            if kind in ("MT", "ST_FI"):
                loss = loss + torch.mean((zFI - rFI) ** 2)
            loss = loss + float(lambda_amp) * _amp_penalty(model)
            if lambda_prior > 0:
                if kind == "MT":
                    loss = loss + float(lambda_prior) * _prior_penalty(
                        model, d_sh, d_sh, d_wm, d_wm, d_fi, d_fi)
                elif kind == "ST_WM":
                    loss = loss + float(lambda_prior) * _prior_penalty(
                        model, d_zero, d_zero, d_wm, d_wm, d_zero, d_zero)
                else:
                    loss = loss + float(lambda_prior) * _prior_penalty(
                        model, d_zero, d_zero, d_zero, d_zero, d_fi, d_fi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            _project_factors(model)

            loss_val = float(loss.item())
            steps = step + 1
            if loss_val < prev_best - 1e-12:
                prev_best = loss_val
                no_improve = 0
            else:
                no_improve += 1
            if step + 1 >= MIN_STEPS and no_improve >= PATIENCE:
                converged = True
                break

        # Recompute final loss on projected parameters
        with torch.no_grad():
            rWM, rFI = model.forward(Xfc, Xsc)
            final_loss = torch.tensor(0.0, dtype=torch.float64)
            if kind in ("MT", "ST_WM"):
                final_loss = final_loss + torch.mean((zWM - rWM) ** 2)
            if kind in ("MT", "ST_FI"):
                final_loss = final_loss + torch.mean((zFI - rFI) ** 2)
            final_loss = final_loss + float(lambda_amp) * _amp_penalty(model)
            if lambda_prior > 0:
                if kind == "MT":
                    final_loss = final_loss + float(lambda_prior) * _prior_penalty(
                        model, d_sh, d_sh, d_wm, d_wm, d_fi, d_fi)
                elif kind == "ST_WM":
                    final_loss = final_loss + float(lambda_prior) * _prior_penalty(
                        model, d_zero, d_zero, d_wm, d_wm, d_zero, d_zero)
                else:
                    final_loss = final_loss + float(lambda_prior) * _prior_penalty(
                        model, d_zero, d_zero, d_zero, d_zero, d_fi, d_fi)
            fval = float(final_loss.item())

        rec = dict(diag_context or {})
        rec.update({"kind": kind, "restart": restart, "final_loss": fval,
                    "steps": steps, "converged": bool(converged),
                    "lambda_amp": float(lambda_amp),
                    "lambda_prior": float(lambda_prior)})
        records.append(rec)

        if fval < best_loss:
            best_loss = fval
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_conv = converged
            best_steps = steps
            best_restart = restart

    for rec in records:
        rec["chosen"] = (rec["restart"] == best_restart)
    if diagnostics is not None:
        diagnostics.extend(records)

    return BCRResult(kind=kind, state=best_state, train_loss=best_loss,
                     converged=best_conv, steps=best_steps,
                     restart_used=best_restart, restart_records=records,
                     runtime_s=time.time() - t_start)


def predict_bcr(result: BCRResult, X_fc: np.ndarray, X_sc: np.ndarray):
    """Predict standardized residuals. Returns (rWM, rFI) possibly None."""
    model = _BCRNet(result.kind).to(dtype=torch.float64)
    model.load_state_dict(result.state)
    model.eval()
    Xfc = torch.tensor(to_edge(X_fc), dtype=torch.float64)
    Xsc = torch.tensor(to_edge(X_sc), dtype=torch.float64)
    with torch.no_grad():
        rWM, rFI = model.forward(Xfc, Xsc)
    return (rWM.numpy() if rWM is not None else None,
            rFI.numpy() if rFI is not None else None)


def coefficient_matrices(result: BCRResult, task: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (B_shared, B_specific) for given task, per modality.

    Returns dicts keyed by modality 'FC'/'SC'.
    """
    s = result.state
    out_shared, out_specific = {}, {}
    if result.kind == "MT":
        if task == "WM":
            a_fc = s["amp_fc_shared_WM"].numpy()
            a_sc = s["amp_sc_shared_WM"].numpy()
            b_fc = float(s["amp_fc_WM"]); b_sc = float(s["amp_sc_WM"])
            Ufc_spec = s["U_fc_WM"][:, 0]; Usc_spec = s["U_sc_WM"][:, 0]
        else:
            a_fc = s["amp_fc_shared_FI"].numpy()
            a_sc = s["amp_sc_shared_FI"].numpy()
            b_fc = float(s["amp_fc_FI"]); b_sc = float(s["amp_sc_FI"])
            Ufc_spec = s["U_fc_FI"][:, 0]; Usc_spec = s["U_sc_FI"][:, 0]
        Ufc_sh = s["U_fc_shared"].numpy(); Usc_sh = s["U_sc_shared"].numpy()
        Bfc_sh = sum(a_fc[k] * np.outer(Ufc_sh[:, k], Ufc_sh[:, k]) for k in range(SHARED_RANK))
        Bsc_sh = sum(a_sc[k] * np.outer(Usc_sh[:, k], Usc_sh[:, k]) for k in range(SHARED_RANK))
        Bfc_sp = b_fc * np.outer(Ufc_spec, Ufc_spec)
        Bsc_sp = b_sc * np.outer(Usc_spec, Usc_spec)
    else:
        if task == "WM":
            Ufc = s["U_fc_WM"].numpy(); Usc = s["U_sc_WM"].numpy()
            a_fc = s["amp_fc_WM"].numpy(); a_sc = s["amp_sc_WM"].numpy()
        else:
            Ufc = s["U_fc_FI"].numpy(); Usc = s["U_sc_FI"].numpy()
            a_fc = s["amp_fc_FI"].numpy(); a_sc = s["amp_sc_FI"].numpy()
        Bfc_sh = np.zeros((N_ROI, N_ROI)); Bsc_sh = np.zeros((N_ROI, N_ROI))
        Bfc_sp = sum(a_fc[k] * np.outer(Ufc[:, k], Ufc[:, k]) for k in range(3))
        Bsc_sp = sum(a_sc[k] * np.outer(Usc[:, k], Usc[:, k]) for k in range(3))
    out_shared["FC"] = Bfc_sh; out_shared["SC"] = Bsc_sh
    out_specific["FC"] = Bfc_sp; out_specific["SC"] = Bsc_sp
    return out_shared, out_specific


def reconstruct_residual(result: BCRResult, task: str,
                         Xe_fc: np.ndarray, Xe_sc: np.ndarray) -> np.ndarray:
    """Reconstruct residual prediction from coefficient matrices.

    Returns b_t + sum_m sum_{ij} B_{m,t,ij} M_{m,ij}.
    """
    shared, specific = coefficient_matrices(result, task)
    out = np.zeros(Xe_fc.shape[0])
    for mod, Xe in [("FC", Xe_fc), ("SC", Xe_sc)]:
        B = shared[mod] + specific[mod]
        out = out + 2.0 * (Xe @ B[_IU[0], _IU[1]])
    b = float(result.state["b_WM" if task == "WM" else "b_FI"])
    return out + b


def roi_importance(result: BCRResult, task: str, alpha: float) -> np.ndarray:
    """s_i = sum_j |alpha * B_{ij}| summed over both modalities."""
    shared, specific = coefficient_matrices(result, task)
    B = shared["FC"] + specific["FC"] + shared["SC"] + specific["SC"]
    B = alpha * B
    np.fill_diagonal(B, 0.0)
    return np.abs(B).sum(axis=1)
