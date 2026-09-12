#!/usr/bin/env python3
"""Pre-lock feasibility screen: nonlinear function classes vs real R0 OOF.

Tests whether ANY smooth-nonlinear (RBF kernel ridge) or tree-ensemble
(GBT) function class extracts signal that the validated R0 misses.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metascfc.phase3a_fix.r0_baseline import R0Baseline  # noqa: E402

FC = np.load(ROOT / 'inputs/dataset_FC/FC_all.npy').astype(np.float64)
SC = np.load(ROOT / 'inputs/dataset_SC/SC_all.npy').astype(np.float64)
y_wm = np.load(ROOT / 'inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy').astype(np.float64)
y_fi = np.load(ROOT / 'inputs/dataset_SC/label_all.npy').astype(np.float64)
iu = np.triu_indices(116, k=1)
fc_e = FC[:, iu[0], iu[1]]
sc_e = SC[:, iu[0], iu[1]]
X = np.hstack([fc_e, sc_e])
subjects = np.array([str(i) for i in range(412)])

# Real R0 OOF over the whole cohort (5-fold cross-fit)
r0 = R0Baseline(fc_e, sc_e, y_wm, y_fi, subjects)
r0_oof = {}
for tname, y in [("WM", y_wm), ("FI", y_fi)]:
    r0_oof[tname] = r0.crossfit_r0_within(tname, np.arange(412), 77, "screen", n_folds=5)


def oof_predict(predict_fn, y, seed):
    oof = np.zeros(len(y))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        scl = StandardScaler().fit(X[tr])
        Xtr, Xte = scl.transform(X[tr]), scl.transform(X[te])
        oof[te] = predict_fn(Xtr, y[tr], Xte, seed)
    return oof


def rbf_krr(Xtr, ytr, Xte, seed, lam_grid=(0.3, 1.0, 3.0, 10.0), mult_grid=(0.5, 1.0, 2.0)):
    # median heuristic bandwidth
    d2 = ((Xtr[:200, None, :] - Xtr[None, :200, :]) ** 2).sum(-1)
    med = np.median(np.sqrt(d2[d2 > 0]))
    ym = ytr.mean()
    yc = ytr - ym
    best, bestscore = None, -9
    for mult in mult_grid:
        sigma = max(mult * med, 1e-6)
        G = Xtr @ Xtr.T
        d2_tr = np.maximum(np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G, 0)
        Ktr = np.exp(-d2_tr / (2 * sigma ** 2))
        for lam in lam_grid:
            rs = []
            for it, iv in KFold(n_splits=3, shuffle=True, random_state=seed + 1).split(Ktr):
                a = np.linalg.solve(Ktr[np.ix_(it, it)] + lam * np.eye(len(it)), yc[it])
                p = Ktr[np.ix_(iv, it)] @ a + ym
                rs.append(np.corrcoef(p, ytr[iv])[0, 1])
            if np.mean(rs) > bestscore:
                bestscore, best = np.mean(rs), (mult, lam)
    mult, lam = best
    sigma = max(mult * med, 1e-6)
    G = Xtr @ Xtr.T
    d2_tr = np.maximum(np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G, 0)
    Ktr = np.exp(-d2_tr / (2 * sigma ** 2))
    Gte = Xte @ Xtr.T
    d2_te = np.maximum((Xte ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None, :] - 2 * Gte, 0)
    Kte = np.exp(-d2_te / (2 * sigma ** 2))
    a = np.linalg.solve(Ktr + lam * np.eye(len(ytr)), yc)
    return Kte @ a + ym


def gbt(Xtr, ytr, Xte, seed):
    m = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=40,
        l2_regularization=1.0, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


for tname, y in [("WM", y_wm), ("FI", y_fi)]:
    base = r0_oof[tname]
    r0r = np.corrcoef(base, y)[0, 1]
    res = y - base
    print(f"\n=== {tname}: real R0 pooled r={r0r:.4f} ===")
    for name, fn in [("RBF-KRR", rbf_krr), ("GBT", gbt)]:
        oof = oof_predict(fn, y, 55)
        r = np.corrcoef(oof, y)[0, 1]
        r_res = np.corrcoef(oof, res)[0, 1]
        best = (-9, None)
        for w in np.arange(0, 1.01, 0.05):
            rr = np.corrcoef((1 - w) * base + w * oof, y)[0, 1]
            if rr > best[0]:
                best = (rr, w)
        print(f"  {name}: own r={r:.4f} | corr~R0resid={r_res:+.4f} | "
              f"ens r={best[0]:.4f} at w={best[1]:.2f} (R0={r0r:.4f})")
