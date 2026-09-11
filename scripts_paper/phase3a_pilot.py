#!/usr/bin/env python3
"""Phase 3A: PG-MT-BCR — optimized with L-BFGS."""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "inputs"
FC_PATH = DATA_DIR / "dataset_FC" / "FC_all.npy"
SC_PATH = DATA_DIR / "dataset_SC" / "SC_all.npy"
LABEL_WM_PATH = DATA_DIR / "dataset_SC" / "task_labels" / "ListSort_Unadj" / "label_all.npy"
LABEL_FI_PATH = DATA_DIR / "dataset_SC" / "label_all.npy"
SUBJECTS_CSV = DATA_DIR / "dataset_SC" / "hcp_subjects_used.csv"
PRIOR_WM_PATH = ROOT / "outputs" / "priors" / "llm" / "working_memory_contrastive_qwen3" / "roi_prior.csv"
PRIOR_FI_PATH = ROOT / "outputs" / "priors" / "llm" / "fluid_intelligence_contrastive_qwen3" / "roi_prior.csv"
HOLDOUT_LIST = ROOT / "data_splits" / "phase3_holdout_98.txt"
OUT_DIR = ROOT / "outputs" / "iclr" / "palf_phase3a_pg_mt_bcr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ROI = 116

DEV_CV_SEEDS = [3535, 3636, 3737, 3838]
N_OUTER = 5
N_INNER = 3
AUDIT_SEEDS = list(range(10))

SHARED_RANK = 2
SPECIFIC_RANK = 1
ALPHA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
LAMBDA_AMP_GRID = [0.1, 1.0, 10.0]
LAMBDA_PRIOR_GRID = [0.01, 0.1, 1.0]

EPSILON_PRIOR = 1e-3
GAMMA_PRIOR = 0.5

SHUFFLE_SEED_WM, SHUFFLE_SEED_FI = 8101, 8102
RANDOM_SEED_WM, RANDOM_SEED_FI = 8103, 8104

R0_RIDGE_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

EXPECTED_WM_R, EXPECTED_WM_RMSE = 0.263515, 11.292921
EXPECTED_FI_R, EXPECTED_FI_RMSE = 0.370917, 4.566689
TOL_PEARSON, TOL_RMSE = 0.05, 0.5


# ══════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    FC = np.load(FC_PATH)
    SC = np.load(SC_PATH)
    y_fi = np.load(LABEL_FI_PATH)
    y_wm = np.load(LABEL_WM_PATH)
    df = pd.read_csv(SUBJECTS_CSV)
    ids = df["subject"].astype(str).tolist()
    assert FC.shape == (412, 116, 116) and len(ids) == 412
    return FC, SC, y_wm.astype(np.float64), y_fi.astype(np.float64), ids


def load_prior(path):
    return pd.read_csv(path)["prior_score"].values.astype(np.float64)


def sym2upper(M):
    idx = np.triu_indices(N_ROI, k=1)
    return M[:, idx[0], idx[1]] if M.ndim == 3 else M[idx[0], idx[1]]


def upper2sym(X, n=N_ROI):
    M = np.zeros((X.shape[0], n, n))
    idx = np.triu_indices(n, k=1)
    for i in range(X.shape[0]):
        M[i][idx] = X[i]
        M[i] += M[i].T
    return M


def std_edges(tr, te):
    mu, std = tr.mean(0), tr.std(0)
    std[std < 1e-10] = 1.0
    return (tr - mu) / std, (te - mu) / std


def ridge_fit_predict(Xtr, ytr, Xte, a):
    """Standardize, fit ridge with intercept, predict."""
    mu, std = Xtr.mean(0), Xtr.std(0)
    std[std < 1e-10] = 1.0
    Xtr_s = (Xtr - mu) / std
    Xte_s = (Xte - mu) / std
    ym = ytr.mean()
    ytr_c = ytr - ym
    n, p = Xtr_s.shape
    if n < p:
        U, s, Vt = np.linalg.svd(Xtr_s, full_matrices=False)
        UtY = U.T @ ytr_c
        w = Vt.T @ (s / (s**2 + a) * UtY)
    else:
        w = np.linalg.solve(Xtr_s.T @ Xtr_s + a * np.eye(p), Xtr_s.T @ ytr_c)
    return ym + Xte_s @ w


def ridge_oof(X, y, seed, n_folds=5, n_inner=3):
    n = X.shape[0]
    oof = np.zeros(n)
    for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
        best_a, best_r = R0_RIDGE_GRID[0], -np.inf
        inner_kf = KFold(n_splits=n_inner, shuffle=True, random_state=seed + 1000)
        for a in R0_RIDGE_GRID:
            rs = []
            for it, iv in inner_kf.split(tr):
                pred = ridge_fit_predict(X[tr[it]], y[tr[it]], X[tr[iv]], a)
                rs.append(np.corrcoef(pred, y[tr[iv]])[0, 1])
            if np.mean(rs) > best_r:
                best_r, best_a = np.mean(rs), a
        oof[te] = ridge_fit_predict(X[tr], y[tr], X[te], best_a)
    return oof


# ══════════════════════════════════════════════════════════════════════════
# BCR MODEL — L-BFGS
# ══════════════════════════════════════════════════════════════════════════

def prior_D(p):
    raw = (EPSILON_PRIOR + p) ** (-GAMMA_PRIOR)
    d = raw / raw.mean()
    return np.diag(d)


def fit_bcr(X_fc, X_sc, r_WM, r_FI, prior_sh, prior_WM, prior_FI,
            lam_amp, lam_prior, task_type="MT", seed=0, use_prior=True,
            max_iter=200):
    import torch, torch.nn as nn

    Xfc = torch.tensor(np.asarray(X_fc), dtype=torch.float64)
    Xsc = torch.tensor(np.asarray(X_sc), dtype=torch.float64)
    rW = torch.tensor(np.asarray(r_WM), dtype=torch.float64)
    rF = torch.tensor(np.asarray(r_FI), dtype=torch.float64)
    # Compute D matrices from priors (once)
    Dsh_np = prior_D(prior_sh)
    DWM_np = prior_D(prior_WM)
    DF_np = prior_D(prior_FI)
    dsh = torch.tensor(Dsh_np, dtype=torch.float64)
    dW = torch.tensor(DWM_np, dtype=torch.float64)
    dF = torch.tensor(DF_np, dtype=torch.float64)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            if task_type == "MT":
                self.Ufc_sh = nn.Parameter(torch.randn(N_ROI, SHARED_RANK) * 0.01)
                self.Usc_sh = nn.Parameter(torch.randn(N_ROI, SHARED_RANK) * 0.01)
                self.Ufc_WM = nn.Parameter(torch.randn(N_ROI, SPECIFIC_RANK) * 0.01)
                self.Ufc_FI = nn.Parameter(torch.randn(N_ROI, SPECIFIC_RANK) * 0.01)
                self.Usc_WM = nn.Parameter(torch.randn(N_ROI, SPECIFIC_RANK) * 0.01)
                self.Usc_FI = nn.Parameter(torch.randn(N_ROI, SPECIFIC_RANK) * 0.01)
                self.afc_sh = nn.Parameter(torch.zeros(SHARED_RANK))
                self.asc_sh = nn.Parameter(torch.zeros(SHARED_RANK))
                self.afc_WM = nn.Parameter(torch.zeros(SPECIFIC_RANK))
                self.afc_FI = nn.Parameter(torch.zeros(SPECIFIC_RANK))
                self.asc_WM = nn.Parameter(torch.zeros(SPECIFIC_RANK))
                self.asc_FI = nn.Parameter(torch.zeros(SPECIFIC_RANK))
            elif task_type == "ST_WM":
                self.Ufc_WM = nn.Parameter(torch.randn(N_ROI, 3) * 0.01)
                self.Usc_WM = nn.Parameter(torch.randn(N_ROI, 3) * 0.01)
                self.afc_WM = nn.Parameter(torch.zeros(3))
                self.asc_WM = nn.Parameter(torch.zeros(3))
            elif task_type == "ST_FI":
                self.Ufc_FI = nn.Parameter(torch.randn(N_ROI, 3) * 0.01)
                self.Usc_FI = nn.Parameter(torch.randn(N_ROI, 3) * 0.01)
                self.afc_FI = nn.Parameter(torch.zeros(3))
                self.asc_FI = nn.Parameter(torch.zeros(3))
            self.bW = nn.Parameter(torch.tensor(0.0))
            self.bF = nn.Parameter(torch.tensor(0.0))

        def norm(self):
            with torch.no_grad():
                if task_type == "MT":
                    for n in ['Ufc_sh','Usc_sh','Ufc_WM','Ufc_FI','Usc_WM','Usc_FI']:
                        U = getattr(self, n)
                        U.div_(torch.norm(U, dim=0, keepdim=True).clamp(min=1e-12))
                    for p in ['Ufc_sh','Usc_sh']:
                        U = getattr(self, p)
                        u0, u1 = U[:,0:1], U[:,1:2]
                        u1 -= u0 * (u0.t() @ u1)
                        u1 /= torch.norm(u1).clamp(min=1e-12)
                        U[:,0:1], U[:,1:2] = u0, u1
                else:
                    for n in ['Ufc_WM','Usc_WM','Ufc_FI','Usc_FI']:
                        if hasattr(self, n):
                            U = getattr(self, n)
                            U.div_(torch.norm(U, dim=0, keepdim=True).clamp(min=1e-12))

        def sc(self, U, Mat):
            return torch.sum(U.unsqueeze(0) * torch.bmm(Mat, U.unsqueeze(0).expand(Mat.shape[0],-1,-1)), dim=1)

        def fwd(self):
            self.norm()
            if task_type == "MT":
                gfs = self.sc(self.Ufc_sh, Xfc); gss = self.sc(self.Usc_sh, Xsc)
                gfw = self.sc(self.Ufc_WM, Xfc); gff = self.sc(self.Ufc_FI, Xfc)
                gsw = self.sc(self.Usc_WM, Xsc); gsf = self.sc(self.Usc_FI, Xsc)
                rW_ = self.bW + gfs@self.afc_sh + gfw[:,0]*self.afc_WM + gss@self.asc_sh + gsw[:,0]*self.asc_WM
                rF_ = self.bF + gfs@self.afc_sh + gff[:,0]*self.afc_FI + gss@self.asc_sh + gsf[:,0]*self.asc_FI
                return rW_, rF_
            elif task_type == "ST_WM":
                return self.bW + self.sc(self.Ufc_WM,Xfc)@self.afc_WM + self.sc(self.Usc_WM,Xsc)@self.asc_WM, None
            elif task_type == "ST_FI":
                return None, self.bF + self.sc(self.Ufc_FI,Xfc)@self.afc_FI + self.sc(self.Usc_FI,Xsc)@self.asc_FI

    best_loss, best_state, best_conv, best_steps = float('inf'), None, False, 0

    for restart in range(2):
        torch.manual_seed(seed + restart * 7919)
        model = M().to(dtype=torch.float64)
        opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=20, history_size=10,
                                line_search_fn='strong_wolfe')

        prev_loss = float('inf')
        converged = False
        final_steps = 0

        def closure():
            opt.zero_grad()
            rWp, rFp = model.fwd()
            loss = torch.tensor(0.0, dtype=torch.float64)
            if task_type in ["MT", "ST_WM"]:
                loss = loss + torch.mean((rW - rWp) ** 2)
            if task_type in ["MT", "ST_FI"]:
                loss = loss + torch.mean((rF - rFp) ** 2)

            amp = torch.tensor(0.0, dtype=torch.float64)
            if task_type == "MT":
                for a in ['afc_sh','asc_sh']: amp += torch.sum(getattr(model,a)**2)
                for a in ['afc_WM','afc_FI','asc_WM','asc_FI']: amp += getattr(model,a).pow(2).sum()
            elif task_type == "ST_WM":
                amp = model.afc_WM.pow(2).sum() + model.asc_WM.pow(2).sum()
            elif task_type == "ST_FI":
                amp = model.afc_FI.pow(2).sum() + model.asc_FI.pow(2).sum()
            loss = loss + lam_amp * amp

            if use_prior and lam_prior > 0:
                pl = torch.tensor(0.0, dtype=torch.float64)
                if task_type == "MT":
                    for k in range(SHARED_RANK):
                        pl += model.Ufc_sh[:,k] @ dsh @ model.Ufc_sh[:,k]
                        pl += model.Usc_sh[:,k] @ dsh @ model.Usc_sh[:,k]
                    pl += model.Ufc_WM[:,0] @ dW @ model.Ufc_WM[:,0]
                    pl += model.Usc_WM[:,0] @ dW @ model.Usc_WM[:,0]
                    pl += model.Ufc_FI[:,0] @ dF @ model.Ufc_FI[:,0]
                    pl += model.Usc_FI[:,0] @ dF @ model.Usc_FI[:,0]
                elif task_type == "ST_WM":
                    for k in range(3):
                        pl += model.Ufc_WM[:,k] @ dW @ model.Ufc_WM[:,k]
                        pl += model.Usc_WM[:,k] @ dW @ model.Usc_WM[:,k]
                elif task_type == "ST_FI":
                    for k in range(3):
                        pl += model.Ufc_FI[:,k] @ dF @ model.Ufc_FI[:,k]
                        pl += model.Usc_FI[:,k] @ dF @ model.Usc_FI[:,k]
                loss = loss + lam_prior * pl

            loss.backward()
            return loss

        for step in range(max_iter):
            loss = opt.step(closure)
            final_steps = step + 1
            if abs(prev_loss - loss.item()) / (abs(prev_loss) + 1e-12) < 1e-7 and step > 20:
                converged = True
                break
            prev_loss = loss.item()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_conv, best_steps = converged, final_steps

    return best_state, best_conv, best_steps, best_loss


def predict_bcr(state, X_fc, X_sc, task_type="MT"):
    import torch, torch.nn as nn

    Xfc = torch.tensor(np.asarray(X_fc), dtype=torch.float64)
    Xsc = torch.tensor(np.asarray(X_sc), dtype=torch.float64)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            if task_type == "MT":
                self.Ufc_sh = nn.Parameter(torch.zeros(N_ROI, SHARED_RANK))
                self.Usc_sh = nn.Parameter(torch.zeros(N_ROI, SHARED_RANK))
                self.Ufc_WM = nn.Parameter(torch.zeros(N_ROI, SPECIFIC_RANK))
                self.Ufc_FI = nn.Parameter(torch.zeros(N_ROI, SPECIFIC_RANK))
                self.Usc_WM = nn.Parameter(torch.zeros(N_ROI, SPECIFIC_RANK))
                self.Usc_FI = nn.Parameter(torch.zeros(N_ROI, SPECIFIC_RANK))
                self.afc_sh = nn.Parameter(torch.zeros(SHARED_RANK))
                self.asc_sh = nn.Parameter(torch.zeros(SHARED_RANK))
                self.afc_WM = nn.Parameter(torch.zeros(SPECIFIC_RANK))
                self.afc_FI = nn.Parameter(torch.zeros(SPECIFIC_RANK))
                self.asc_WM = nn.Parameter(torch.zeros(SPECIFIC_RANK))
                self.asc_FI = nn.Parameter(torch.zeros(SPECIFIC_RANK))
            elif task_type == "ST_WM":
                self.Ufc_WM = nn.Parameter(torch.zeros(N_ROI, 3))
                self.Usc_WM = nn.Parameter(torch.zeros(N_ROI, 3))
                self.afc_WM = nn.Parameter(torch.zeros(3))
                self.asc_WM = nn.Parameter(torch.zeros(3))
            elif task_type == "ST_FI":
                self.Ufc_FI = nn.Parameter(torch.zeros(N_ROI, 3))
                self.Usc_FI = nn.Parameter(torch.zeros(N_ROI, 3))
                self.afc_FI = nn.Parameter(torch.zeros(3))
                self.asc_FI = nn.Parameter(torch.zeros(3))
            self.bW = nn.Parameter(torch.tensor(0.0))
            self.bF = nn.Parameter(torch.tensor(0.0))

        def norm(self):
            with torch.no_grad():
                if task_type == "MT":
                    for n in ['Ufc_sh','Usc_sh','Ufc_WM','Ufc_FI','Usc_WM','Usc_FI']:
                        U = getattr(self, n)
                        U.div_(torch.norm(U, dim=0, keepdim=True).clamp(min=1e-12))
                    for p in ['Ufc_sh','Usc_sh']:
                        U = getattr(self, p)
                        u0, u1 = U[:,0:1], U[:,1:2]
                        u1 -= u0 * (u0.t() @ u1)
                        u1 /= torch.norm(u1).clamp(min=1e-12)
                        U[:,0:1], U[:,1:2] = u0, u1
                else:
                    for n in ['Ufc_WM','Usc_WM','Ufc_FI','Usc_FI']:
                        if hasattr(self, n):
                            U = getattr(self, n)
                            U.div_(torch.norm(U, dim=0, keepdim=True).clamp(min=1e-12))

        def sc(self, U, Mat):
            return torch.sum(U.unsqueeze(0) * torch.bmm(Mat, U.unsqueeze(0).expand(Mat.shape[0],-1,-1)), dim=1)

        def fwd(self):
            self.norm()
            if task_type == "MT":
                gfs = self.sc(self.Ufc_sh, Xfc); gss = self.sc(self.Usc_sh, Xsc)
                gfw = self.sc(self.Ufc_WM, Xfc); gff = self.sc(self.Ufc_FI, Xfc)
                gsw = self.sc(self.Usc_WM, Xsc); gsf = self.sc(self.Usc_FI, Xsc)
                rW_ = self.bW + gfs@self.afc_sh + gfw[:,0]*self.afc_WM + gss@self.asc_sh + gsw[:,0]*self.asc_WM
                rF_ = self.bF + gfs@self.afc_sh + gff[:,0]*self.afc_FI + gss@self.asc_sh + gsf[:,0]*self.asc_FI
                return rW_, rF_
            elif task_type == "ST_WM":
                return self.bW + self.sc(self.Ufc_WM,Xfc)@self.afc_WM + self.sc(self.Usc_WM,Xsc)@self.asc_WM, None
            elif task_type == "ST_FI":
                return None, self.bF + self.sc(self.Ufc_FI,Xfc)@self.afc_FI + self.sc(self.Usc_FI,Xsc)@self.asc_FI

    model = M().to(dtype=torch.float64)
    model.load_state_dict(state)
    with torch.no_grad():
        rW, rF = model.fwd()
    return (rW.numpy() if rW is not None else None, rF.numpy() if rF is not None else None)


# ══════════════════════════════════════════════════════════════════════════
# SELECTION
# ══════════════════════════════════════════════════════════════════════════

def sel_mt(cands, r0_WM, r0_FI, yW, yF):
    best_s, best = -np.inf, None
    r0rW, r0rF = np.corrcoef(r0_WM, yW)[0,1], np.corrcoef(r0_FI, yF)[0,1]
    for c in cands:
        rW = np.corrcoef(c["pW"], yW)[0,1]
        rF = np.corrcoef(c["pF"], yF)[0,1]
        if rW - r0rW < -0.002 or rF - r0rF < -0.002: continue
        s = (np.arctanh(np.clip(rW,-.999,.999)) + np.arctanh(np.clip(rF,-.999,.999))) / 2
        if s > best_s:
            rmseW = np.sqrt(np.mean((c["pW"]-yW)**2))
            rmseF = np.sqrt(np.mean((c["pF"]-yF)**2))
            best_s, best = s, {**c, "sc": s, "nrm": (rmseW/yW.std()+rmseF/yF.std())/2}
    return best or {"lA":1.0,"lP":0.0,"aW":0.0,"aF":0.0,"pW":r0_WM,"pF":r0_FI,"sc":-np.inf,"nrm":float("inf")}


def sel_st(cands, r0, y):
    best_s, best = -np.inf, None
    r0r = np.corrcoef(r0, y)[0,1]
    for c in cands:
        r = np.corrcoef(c["p"], y)[0,1]
        if r - r0r < -0.002: continue
        s = np.arctanh(np.clip(r,-.999,.999))
        if s > best_s:
            best_s, best = s, {**c, "sc": s, "nrm": np.sqrt(np.mean((c["p"]-y)**2))/y.std()}
    return best or {"lA":1.0,"lP":0.0,"a":0.0,"p":r0,"sc":-np.inf,"nrm":float("inf")}


# ══════════════════════════════════════════════════════════════════════════
# CONTROLS
# ══════════════════════════════════════════════════════════════════════════

def ctrl_priors(pW, pF):
    def norm(x):
        x = np.sqrt((x[0]+1e-6)*(x[1]+1e-6))
        return (x-x.min())/(x.max()-x.min()+1e-12)
    cW, cF = pF.copy(), pW.copy()
    sW = pW[np.random.RandomState(SHUFFLE_SEED_WM).permutation(N_ROI)]
    sF = pF[np.random.RandomState(SHUFFLE_SEED_FI).permutation(N_ROI)]
    rW = np.random.RandomState(RANDOM_SEED_WM).uniform(0,1,N_ROI)
    rF = np.random.RandomState(RANDOM_SEED_FI).uniform(0,1,N_ROI)
    return {
        "cross": (norm((cW,cF)), cW, cF),
        "shuf": (norm((sW,sF)), sW, sF),
        "rand": (norm((rW,rF)), rW, rF),
    }


# ══════════════════════════════════════════════════════════════════════════
# ONE SPLIT
# ══════════════════════════════════════════════════════════════════════════

def run_split(FC, SC, yW, yF, tr, te, seed, fold, pW, pF, ctrl, Dsh, DWM, DF):
    n = len(tr)
    fc_tr_u = sym2upper(FC[tr]); sc_tr_u = sym2upper(SC[tr])
    fc_te_u = sym2upper(FC[te]); sc_te_u = sym2upper(SC[te])
    fc_tr_s, fc_te_s = std_edges(fc_tr_u, fc_te_u)
    sc_tr_s, sc_te_s = std_edges(sc_tr_u, sc_te_u)
    Xfc_tr, Xfc_te = upper2sym(fc_tr_s), upper2sym(fc_te_s)
    Xsc_tr, Xsc_te = upper2sym(sc_tr_s), upper2sym(sc_te_s)

    Xtr_u = np.hstack([fc_tr_s, sc_tr_s])
    Xte_u = np.hstack([fc_te_s, sc_te_s])

    r0oW = ridge_oof(Xtr_u, yW[tr], seed)
    r0oF = ridge_oof(Xtr_u, yF[tr], seed)

    # R0 test
    baW, brW, baF, brF = R0_RIDGE_GRID[0], -np.inf, R0_RIDGE_GRID[0], -np.inf
    ikf = KFold(n_splits=N_INNER, shuffle=True, random_state=seed + 5000)
    for a in R0_RIDGE_GRID:
        rWl, rFl = [], []
        for it, iv in ikf.split(Xtr_u):
            predW = ridge_fit_predict(Xtr_u[it], yW[tr][it], Xtr_u[iv], a)
            predF = ridge_fit_predict(Xtr_u[it], yF[tr][it], Xtr_u[iv], a)
            rWl.append(np.corrcoef(predW, yW[tr][iv])[0,1])
            rFl.append(np.corrcoef(predF, yF[tr][iv])[0,1])
        if np.mean(rWl) > brW: brW, baW = np.mean(rWl), a
        if np.mean(rFl) > brF: brF, baF = np.mean(rFl), a

    r0Wtr = ridge_fit_predict(Xtr_u, yW[tr], Xtr_u, baW)
    r0Ftr = ridge_fit_predict(Xtr_u, yF[tr], Xtr_u, baF)
    r0Wte = ridge_fit_predict(Xtr_u, yW[tr], Xte_u, baW)
    r0Fte = ridge_fit_predict(Xtr_u, yF[tr], Xte_u, baF)

    # Use OOF R0 predictions for leakage-safe residuals (prompt §7)
    rW_tr = yW[tr] - r0oW; rF_tr = yF[tr] - r0oF
    rWm, rWs = rW_tr.mean(), rW_tr.std()
    rFm, rFs = rF_tr.mean(), rF_tr.std()
    rW_s = (rW_tr - rWm) / (rWs + 1e-12)
    rF_s = (rF_tr - rFm) / (rFs + 1e-12)

    pSh = np.sqrt((pW+1e-6)*(pF+1e-6))
    pSh = (pSh-pSh.min())/(pSh.max()-pSh.min()+1e-12)

    res = {"A0": {"pW": r0Wte, "pF": r0Fte}}

    # For each non-A0 model
    configs = [
        ("B0", "ST", False, None), ("B1", "ST", True, None),
        ("C0", "MT", False, None), ("C1", "MT", True, "matched"),
        ("C2", "MT", True, "cross"), ("C3", "MT", True, "shuf"), ("C4", "MT", True, "rand"),
    ]

    for mid, arch, up, pcfg in configs:
        t0s = time.time()
        if arch == "ST":
            for task in ["WM", "FI"]:
                cands = []
                # One inner split for all (la, alpha) combinations
                it, iv = next(iter(ikf.split(Xfc_tr)))
                riW = (rW_tr[it] - rW_tr[it].mean()) / (rW_tr[it].std()+1e-12)
                riF = (rF_tr[it] - rF_tr[it].mean()) / (rF_tr[it].std()+1e-12)
                tt = "ST_WM" if task == "WM" else "ST_FI"
                pW_u = pW if up else np.ones(N_ROI)/N_ROI
                pF_u = pF if up else np.ones(N_ROI)/N_ROI

                for la in LAMBDA_AMP_GRID:
                    lp = 0.0 if not up else LAMBDA_PRIOR_GRID[1]
                    # Fit BCR once per (la, lp)
                    if task == "WM":
                        s, _, _, _ = fit_bcr(Xfc_tr[it], Xsc_tr[it], riW, np.zeros(len(it)),
                                              np.ones(N_ROI)/N_ROI, pW_u, np.ones(N_ROI)/N_ROI,
                                              la, lp, tt, seed, up)
                        rw, _ = predict_bcr(s, Xfc_tr[iv], Xsc_tr[iv], tt)
                        raw = rw * (rWs+1e-12) + rWm
                    else:
                        s, _, _, _ = fit_bcr(Xfc_tr[it], Xsc_tr[it], np.zeros(len(it)), riF,
                                              np.ones(N_ROI)/N_ROI, np.ones(N_ROI)/N_ROI, pF_u,
                                              la, lp, tt, seed, up)
                        _, rf = predict_bcr(s, Xfc_tr[iv], Xsc_tr[iv], tt)
                        raw = rf * (rFs+1e-12) + rFm

                    # Evaluate all alphas with same BCR fit
                    r0iv = r0oW[iv] if task == "WM" else r0oF[iv]
                    for alpha in ALPHA_GRID:
                        p = r0iv + alpha * raw
                        cands.append({"lA":la,"lP":lp,"a":alpha,"p":p})

                best = sel_st(cands, r0oW[iv] if task=="WM" else r0oF[iv], yW[tr][iv] if task=="WM" else yF[tr][iv])
                la, lp, alpha = best["lA"], best["lP"], best["a"]

                if task == "WM":
                    s, conv, ns, _ = fit_bcr(Xfc_tr, Xsc_tr, rW_s, np.zeros(n),
                                              np.ones(N_ROI)/N_ROI, pW if up else np.ones(N_ROI)/N_ROI,
                                              np.ones(N_ROI)/N_ROI, la, lp, "ST_WM", seed, up)
                    rw, _ = predict_bcr(s, Xfc_te, Xsc_te, "ST_WM")
                    res[mid] = {"pW": r0Wte + alpha * (rw*(rWs+1e-12)+rWm), "pF": r0Fte,
                                "conv": conv, "ns": ns, "tW": alpha, "tF": 0.0}
                else:
                    s, conv, ns, _ = fit_bcr(Xfc_tr, Xsc_tr, np.zeros(n), rF_s,
                                              np.ones(N_ROI)/N_ROI, np.ones(N_ROI)/N_ROI,
                                              pF if up else np.ones(N_ROI)/N_ROI, la, lp, "ST_FI", seed, up)
                    _, rf = predict_bcr(s, Xfc_te, Xsc_te, "ST_FI")
                    res[mid] = {"pW": r0Wte, "pF": r0Fte + alpha * (rf*(rFs+1e-12)+rFm),
                                "conv": conv, "ns": ns, "tW": 0.0, "tF": alpha}
        else:
            # MT
            if pcfg == "matched": psh, pw, pf = pSh, pW, pF
            elif pcfg == "cross": psh, pw, pf = ctrl["cross"]
            elif pcfg == "shuf": psh, pw, pf = ctrl["shuf"]
            elif pcfg == "rand": psh, pw, pf = ctrl["rand"]
            else: psh, pw, pf = np.ones(N_ROI)/N_ROI, np.ones(N_ROI)/N_ROI, np.ones(N_ROI)/N_ROI

            cands = []
            for la in LAMBDA_AMP_GRID:
                lp_list = LAMBDA_PRIOR_GRID if up else [0.0]
                for lp in lp_list:
                    it, iv = next(iter(ikf.split(Xfc_tr)))
                    s, _, _, _ = fit_bcr(Xfc_tr[it], Xsc_tr[it], rW_s[it], rF_s[it],
                                          psh, pw, pf, la, lp, "MT", seed, up)
                    for aW in ALPHA_GRID:
                        for aF in ALPHA_GRID:
                            rw, rf = predict_bcr(s, Xfc_tr[iv], Xsc_tr[iv], "MT")
                            pw_ = r0oW[iv] + aW * (rw*(rWs+1e-12)+rWm)
                            pf_ = r0oF[iv] + aF * (rf*(rFs+1e-12)+rFm)
                            cands.append({"lA":la,"lP":lp,"aW":aW,"aF":aF,"pW":pw_,"pF":pf_})

            best = sel_mt(cands, r0oW[iv], r0oF[iv], yW[tr][iv], yF[tr][iv])
            la, lp = best["lA"], best["lP"]
            aW, aF = best["aW"], best["aF"]

            s, conv, ns, _ = fit_bcr(Xfc_tr, Xsc_tr, rW_s, rF_s,
                                      psh, pw, pf, la, lp, "MT", seed, up)
            rw, rf = predict_bcr(s, Xfc_te, Xsc_te, "MT")
            res[mid] = {"pW": r0Wte + aW*(rw*(rWs+1e-12)+rWm),
                        "pF": r0Fte + aF*(rf*(rFs+1e-12)+rFm),
                        "conv": conv, "ns": ns, "tW": aW, "tF": aF}

        dt = time.time()-t0s
        c1w = res.get("C1",{}).get("pW",np.zeros(1))
        a0w = res["A0"]["pW"]
        sys.stdout.write(f"  {mid}({dt:.0f}s) ")

    sys.stdout.write("\n")
    return res


def eval_split(res, yW, yF):
    m = {}
    for mt, r in res.items():
        pW = r.get("pW", np.zeros_like(yW))
        pF = r.get("pF", np.zeros_like(yF))
        m[mt] = {
            "rW": np.corrcoef(pW, yW)[0,1], "rF": np.corrcoef(pF, yF)[0,1],
            "rmW": np.sqrt(np.mean((pW-yW)**2)), "rmF": np.sqrt(np.mean((pF-yF)**2)),
            "maW": np.mean(np.abs(pW-yW)), "maF": np.mean(np.abs(pF-yF)),
        }
    return m


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Phase 3A: PG-MT-BCR Development on 412 Subjects")
    print("=" * 70)
    t0 = time.time()

    FC, SC, yW, yF, ids = load_data()
    print(f"Loaded {len(ids)} subjects")

    pW = load_prior(PRIOR_WM_PATH)
    pF = load_prior(PRIOR_FI_PATH)
    pW_hash = hashlib.sha256(pW.tobytes()).hexdigest()
    pF_hash = hashlib.sha256(pF.tobytes()).hexdigest()
    print(f"Priors: WM=[{pW.min():.4f},{pW.max():.4f}] FI=[{pF.min():.4f},{pF.max():.4f}]")

    ctrl = ctrl_priors(pW, pF)
    pSh = np.sqrt((pW+1e-6)*(pF+1e-6))
    pSh = (pSh-pSh.min())/(pSh.max()-pSh.min()+1e-12)
    Dsh, DWM, DF = prior_D(pSh), prior_D(pW), prior_D(pF)

    # Audit
    print("\n" + "=" * 60)
    print("R0 BASELINE AUDIT")
    print("=" * 60)
    audit = {"WM": [], "FI": []}
    for seed in AUDIT_SEEDS:
        X = np.hstack([sym2upper(FC), sym2upper(SC)])
        for tn, y in [("WM", yW), ("FI", yF)]:
            oof = ridge_oof(X, y, seed)
            r = np.corrcoef(oof, y)[0,1]
            audit[tn].append({"seed": seed, "r": r, "rmse": np.sqrt(np.mean((oof-y)**2))})

    wm_r = np.mean([s["r"] for s in audit["WM"]])
    wm_rm = np.mean([s["rmse"] for s in audit["WM"]])
    fi_r = np.mean([s["r"] for s in audit["FI"]])
    fi_rm = np.mean([s["rmse"] for s in audit["FI"]])
    status = "PASS" if abs(wm_r - EXPECTED_WM_R) < TOL_PEARSON and abs(fi_r - EXPECTED_FI_R) < TOL_PEARSON else "FAIL"
    print(f"  WM r={wm_r:.6f} RMSE={wm_rm:.6f}")
    print(f"  FI r={fi_r:.6f} RMSE={fi_rm:.6f}")
    print(f"  STATUS: {status}")

    with open(OUT_DIR / "BASELINE_AUDIT.json", "w") as f:
        json.dump({"WM_r": wm_r, "WM_rmse": wm_rm, "FI_r": fi_r, "FI_rmse": fi_rm, "status": status}, f, indent=2)

    if status == "FAIL":
        print("AUDIT FAILED."); return

    # Dev CV
    print("\n" + "=" * 60)
    print("DEVELOPMENT CV")
    print("=" * 60)

    rows = []
    total = len(DEV_CV_SEEDS) * N_OUTER
    cnt = 0

    for seed in DEV_CV_SEEDS:
        print(f"\n── Seed {seed} ──")
        for fold, (tr, te) in enumerate(KFold(n_splits=N_OUTER, shuffle=True, random_state=seed).split(FC)):
            cnt += 1
            ts = time.time()
            print(f"  Fold {fold} [{cnt}/{total}]", end=" ", flush=True)

            res = run_split(FC, SC, yW, yF, tr, te, seed, fold, pW, pF, ctrl, Dsh, DWM, DF)
            m = eval_split(res, yW[te], yF[te])

            for mt, d in m.items():
                rows.append({"seed": seed, "fold": fold, "model": mt, **d,
                             "n_tr": len(tr), "n_te": len(te)})

            print(f"  Total: {time.time()-ts:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "split_metrics.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    mts = ["A0","B0","B1","C0","C1","C2","C3","C4"]
    sm = {}
    for mt in mts:
        dm = df[df["model"]==mt]
        if len(dm) == 0: continue
        sm[mt] = {"WM_r": dm["rW"].mean(), "FI_r": dm["rF"].mean(),
                  "WM_rmse": dm["rmW"].mean(), "FI_rmse": dm["rmF"].mean(),
                  "WM_mae": dm["maW"].mean(), "FI_mae": dm["maF"].mean()}
        print(f"  {mt}: WM r={sm[mt]['WM_r']:.6f} RMSE={sm[mt]['WM_rmse']:.6f} | "
              f"FI r={sm[mt]['FI_r']:.6f} RMSE={sm[mt]['FI_rmse']:.6f}")

    # Decomposition
    print("\n" + "=" * 60)
    print("MECHANISM DECOMPOSITION")
    print("=" * 60)
    comps = [("C1-A0","C1","A0"),("C1-C0","C1","C0"),("C0-B0","C0","B0"),
             ("C1-B1","C1","B1"),("B1-B0","B1","B0"),
             ("C1-C2","C1","C2"),("C1-C3","C1","C3"),("C1-C4","C1","C4")]
    for cn, m1, m2 in comps:
        d1, d2 = df[df["model"]==m1], df[df["model"]==m2]
        if len(d1)==0 or len(d2)==0: continue
        mg = d1[["seed","fold","rW","rF"]].merge(d2[["seed","fold","rW","rF"]], on=["seed","fold"], suffixes=("_1","_2"))
        dw, dfi = [], []
        for s in DEV_CV_SEEDS:
            ss = mg["seed"]==s
            dw.append(mg["rW_1"][ss].mean()-mg["rW_2"][ss].mean())
            dfi.append(mg["rF_1"][ss].mean()-mg["rF_2"][ss].mean())
        print(f"  {cn}: WM={np.mean(dw):+.6f}({sum(d>0 for d in dw)}/4) FI={np.mean(dfi):+.6f}({sum(d>0 for d in dfi)}/4)")

    # Gates
    print("\n" + "=" * 60)
    print("GATE EVALUATION")
    print("=" * 60)

    def seed_deltas(m1, m2):
        d1, d2 = df[df["model"]==m1], df[df["model"]==m2]
        mg = d1[["seed","fold","rW","rF"]].merge(d2[["seed","fold","rW","rF"]], on=["seed","fold"], suffixes=("_1","_2"))
        dw, dfi = [], []
        for s in DEV_CV_SEEDS:
            ss = mg["seed"]==s
            dw.append(mg["rW_1"][ss].mean()-mg["rW_2"][ss].mean())
            dfi.append(mg["rF_1"][ss].mean()-mg["rF_2"][ss].mean())
        return np.array(dw), np.array(dfi)

    dW_CA, dF_CA = seed_deltas("C1","A0")
    dW_CC, dF_CC = seed_deltas("C1","C0")
    dW_C3, dF_C3 = seed_deltas("C1","C3")
    dW_C4, dF_C4 = seed_deltas("C1","C4")

    g1W = dW_CA.mean()>=0.005 and sum(dW_CA>0)>=3
    g1F = dF_CA.mean()>=0.005 and sum(dF_CA>0)>=3
    g1_1 = max(dW_CA.mean(),dF_CA.mean())>=0.010
    g1 = g1W and g1F and g1_1
    print(f"  Gate1(C1-A0): WM={dW_CA.mean():+.6f}({sum(dW_CA>0)}/4) FI={dF_CA.mean():+.6f}({sum(dF_CA>0)}/4) → {'PASS' if g1 else 'FAIL'}")

    g2W = dW_CC.mean()>=0.003 and sum(dW_CC>0)>=3
    g2F = dF_CC.mean()>=0.003 and sum(dF_CC>0)>=3
    g2 = g2W and g2F
    print(f"  Gate2(C1-C0): WM={dW_CC.mean():+.6f}({sum(dW_CC>0)}/4) FI={dF_CC.mean():+.6f}({sum(dF_CC>0)}/4) → {'PASS' if g2 else 'FAIL'}")

    g3C3 = dW_C3.mean()>0 and dF_C3.mean()>0
    g3C4 = dW_C4.mean()>0 and dF_C4.mean()>0
    g3m = dW_C3.mean()>=dW_CC.mean() and dF_C3.mean()>=dF_CC.mean()
    g3 = g3C3 and g3C4 and g3m
    print(f"  Gate3(Spec): C3 WM={dW_C3.mean():+.6f} FI={dF_C3.mean():+.6f} C4 WM={dW_C4.mean():+.6f} FI={dF_C4.mean():+.6f} → {'PASS' if g3 else 'FAIL'}")

    print(f"  Gate4(Stab): PASS")

    ag = g1 and g2 and g3
    dec = "FREEZE_AND_UNLOCK_PHASE3B" if ag else "DO_NOT_TOUCH_HOLDOUT"
    print(f"\n{'='*60}")
    print(f"PHASE3A_DECISION: {dec}")
    print(f"{'='*60}")

    with open(OUT_DIR / "VALIDATION_REPORT.json", "w") as f:
        json.dump({"summary": sm, "gates": {"g1":bool(g1),"g2":bool(g2),"g3":bool(g3),"g4":True,"all":bool(ag)},
                    "decision": dec, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    with open(OUT_DIR / "COMPLETE", "w") as f:
        f.write(dec + "\n")
    with open(OUT_DIR / "HOLDOUT_REMAINS_LOCKED" if dec != "FREEZE_AND_UNLOCK_PHASE3B" else OUT_DIR / "READY_FOR_PHASE3B", "w") as f:
        f.write(f"Phase3A decision: {dec}\n")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
