#!/usr/bin/env python3
"""Pre-lock feasibility screen: multi-view (raw-edge + node-strength profile)."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

FC = np.load('inputs/dataset_FC/FC_all.npy').astype(np.float64)
SC = np.load('inputs/dataset_SC/SC_all.npy').astype(np.float64)
y_wm = np.load('inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy').astype(np.float64)
y_fi = np.load('inputs/dataset_SC/label_all.npy').astype(np.float64)
iu = np.triu_indices(116, k=1)
fc_e = FC[:, iu[0], iu[1]]
sc_e = SC[:, iu[0], iu[1]]

def node_profile(M):
    """Per-ROI connection profiles = rows of the symmetrized matrix."""
    out = []
    for m in M:
        W = np.abs(np.asarray(m, dtype=np.float64))
        np.fill_diagonal(W, 0)
        out.append(W)
    return np.array(out)          # 412 x 116 x 116

def rowsum_feats(M):
    """Node strength profiles (both signs preserved for FC)."""
    return M[:, :, :].sum(axis=2)  # 412 x 116

X_str = np.hstack([rowsum_feats(FC), rowsum_feats(SC)])   # 232 features
GRID = [0.1, 1.0, 10.0, 100.0]

def oof_ridge(X, y, seed):
    oof = np.zeros(len(y))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        scl = StandardScaler().fit(X[tr])
        Xtr, Xte = scl.transform(X[tr]), scl.transform(X[te])
        best, bestr = GRID[0], -9
        for a in GRID:
            rs = []
            for it, iv in KFold(n_splits=3, shuffle=True, random_state=seed + 1).split(Xtr):
                m = Ridge(alpha=a).fit(Xtr[it], y[tr][it])
                rs.append(np.corrcoef(m.predict(Xtr[iv]), y[tr][iv])[0, 1])
            if np.mean(rs) > bestr:
                bestr, best = np.mean(rs), a
        oof[te] = Ridge(alpha=best).fit(Xtr, y[tr]).predict(Xte)
    return oof

for tname, y in [("WM", y_wm), ("FI", y_fi)]:
    r0 = oof_ridge(np.hstack([fc_e, sc_e]), y, 41)
    st = oof_ridge(X_str, y, 41)
    r_res = np.corrcoef(st, y - r0)[0, 1]
    best = (-9, None)
    for w in np.arange(0, 1.01, 0.05):
        r = np.corrcoef((1 - w) * r0 + w * st, y)[0, 1]
        if r > best[0]:
            best = (r, w)
    print(f"{tname}: R0r={np.corrcoef(r0,y)[0,1]:.4f} strength_r={np.corrcoef(st,y)[0,1]:.4f} "
          f"~R0resid r={r_res:+.4f} | ens r={best[0]:.4f} at w={best[1]:.2f}")
