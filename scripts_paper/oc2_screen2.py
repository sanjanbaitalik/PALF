#!/usr/bin/env python3
"""OpenChallenge screen #2: cross-modal residual, population basis, rank-norm,
local kNN, RBF kernel — honest nested blending with R0."""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metascfc.phase3a_fix.r0_baseline import R0Baseline  # noqa: E402

FC = np.load(ROOT / "inputs/dataset_FC/FC_all.npy").astype(np.float64)
SC = np.load(ROOT / "inputs/dataset_SC/SC_all.npy").astype(np.float64)
Y = {
    "WM": np.load(ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy").astype(np.float64),
    "FI": np.load(ROOT / "inputs/dataset_SC/label_all.npy").astype(np.float64),
}
N, IU = 412, np.triu_indices(116, 1)
FC_E, SC_E = FC[:, IU[0], IU[1]], SC[:, IU[0], IU[1]]
TASKS = ("WM", "FI")


def partition(scope, seed, tag, k=3):
    h = int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()[:8], 16)
    perm = np.random.RandomState(h).permutation(len(scope))
    sizes = np.full(k, len(scope) // k)
    sizes[: len(scope) % k] += 1
    folds, cur = [], 0
    for i in range(k):
        a, b = cur, cur + sizes[i]
        folds.append((scope[np.concatenate([perm[:a], perm[b:]])], scope[perm[a:b]]))
        cur = b
    return folds


def ranknorm(A):
    """Per-subject rank -> normal scores (no cross-subject information)."""
    from scipy.stats import norm
    n, p = A.shape
    R = np.argsort(np.argsort(A, axis=1), axis=1).astype(np.float64)
    return norm.ppf((R + 0.5) / p)


def group_basis_feats(tr, te, which):
    """Project modality matrices into population Laplacian eigenbasis of SC."""
    W = np.clip(SC[tr].mean(0), 0, None)
    np.fill_diagonal(W, 0)
    d = W.sum(1)
    d[d <= 1e-12] = 1e-12
    L = np.eye(116) - np.diag(1 / np.sqrt(d)) @ W @ np.diag(1 / np.sqrt(d))
    _, U = np.linalg.eigh(L)
    src = FC if which == "FC" else SC
    out_tr = np.stack([U.T @ src[s] @ U for s in tr])
    out_te = np.stack([U.T @ src[s] @ U for s in te])
    return out_tr[:, IU[0], IU[1]], out_te[:, IU[0], IU[1]]


def pred_ridge(Xtr, ytr, Xte, alpha=1000.0):
    s = StandardScaler().fit(Xtr)
    return Ridge(alpha=alpha).fit(s.transform(Xtr), ytr).predict(s.transform(Xte))


def pred_crossresid_ridge(tr, te, ytr, use_sc=True, alpha=1000.0):
    """Residual of FC after linear prediction from SC (fit on train only)."""
    sf = StandardScaler().fit(FC_E[tr])
    ss = StandardScaler().fit(SC_E[tr])
    A = sf.transform(FC_E[tr]); B = ss.transform(SC_E[tr])
    At = sf.transform(FC_E[te]); Bt = ss.transform(SC_E[te])
    m = Ridge(alpha=100.0).fit(B, A)
    Rtr = A - m.predict(B)
    Rte = At - m.predict(Bt)
    Xtr = np.hstack([Rtr, B]) if use_sc else Rtr
    Xte = np.hstack([Rte, Bt]) if use_sc else Rte
    return pred_ridge(Xtr, ytr, Xte, alpha)


def pred_knn_scores(tr, te, ytr, K=5, k=25):
    sf = StandardScaler().fit(FC_E[tr]); ss = StandardScaler().fit(SC_E[tr])
    X = np.hstack([sf.transform(FC_E[tr]), ss.transform(SC_E[tr])])
    Xt = np.hstack([sf.transform(FC_E[te]), ss.transform(SC_E[te])])
    pls = PLSRegression(n_components=K, scale=False).fit(X, ytr)
    S = pls.transform(X); St = pls.transform(Xt)
    st = StandardScaler().fit(S)
    return KNeighborsRegressor(n_neighbors=k, weights="distance").fit(
        st.transform(S), ytr).predict(st.transform(St))


def pred_rbf(tr, te, ytr, mult=1.0, lam=3.0):
    sf = StandardScaler().fit(FC_E[tr]); ss = StandardScaler().fit(SC_E[tr])
    X = np.hstack([sf.transform(FC_E[tr]), ss.transform(SC_E[tr])])
    Xt = np.hstack([sf.transform(FC_E[te]), ss.transform(SC_E[te])])
    G = X @ X.T
    d2 = np.maximum(np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G, 0)
    sig = mult * np.median(np.sqrt(d2[d2 > 0]))
    K = np.exp(-d2 / (2 * sig ** 2))
    yc = ytr - ytr.mean()
    a = np.linalg.solve(K + lam * np.eye(len(ytr)), yc)
    Gt = Xt @ X.T
    d2t = np.maximum((Xt ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2 * Gt, 0)
    Kt = np.exp(-d2t / (2 * sig ** 2))
    return Kt @ a + ytr.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    seed = args.seed

    cands = {
        "gbasisFC": lambda tr, te, ytr: pred_ridge(
            *group_basis_feats(tr, te, "FC"), ytr) if False else None,
    }
    del cands

    def mk_gbasis(which, alpha=1000.0):
        def f(tr, te, ytr):
            A, B = group_basis_feats(tr, te, which)
            return pred_ridge(A, ytr, B, alpha)
        return f

    def mk_rank(alpha=1000.0):
        def f(tr, te, ytr):
            A = np.hstack([ranknorm(FC_E[tr]), ranknorm(SC_E[tr])])
            B = np.hstack([ranknorm(FC_E[te]), ranknorm(SC_E[te])])
            return pred_ridge(A, ytr, B, alpha)
        return f

    cands = {
        "gbasisFC": mk_gbasis("FC"),
        "gbasisSC": mk_gbasis("SC"),
        "ranknorm": mk_rank(),
        "crossres_wsc": lambda tr, te, ytr: pred_crossresid_ridge(tr, te, ytr, True),
        "crossres_only": lambda tr, te, ytr: pred_crossresid_ridge(tr, te, ytr, False),
        "knn15": lambda tr, te, ytr: pred_knn_scores(tr, te, ytr, K=5, k=15),
        "knn40": lambda tr, te, ytr: pred_knn_scores(tr, te, ytr, K=5, k=40),
        "rbf": lambda tr, te, ytr: pred_rbf(tr, te, ytr),
    }
    names = list(cands)

    r0 = R0Baseline(FC_E, SC_E, Y["WM"], Y["FI"], np.array([str(i) for i in range(N)]))
    r0_r = {t: [] for t in TASKS}
    st_r = {t: {n: [] for n in names} for t in TASKS}
    bl_r = {t: {n: [] for n in names} for t in TASKS}
    w_sel = {t: {n: [] for n in names} for t in TASKS}

    idx = np.random.RandomState(seed).permutation(N)
    sizes = np.full(args.folds, N // args.folds)
    sizes[: N % args.folds] += 1
    cur = 0
    for fold in range(args.folds):
        a, b = cur, cur + sizes[fold]
        te = idx[a:b]
        tr = np.concatenate([idx[:a], idx[b:]])
        cur = b
        t0 = time.time()
        inner = partition(tr, seed, f"outer{fold}", 3)
        pos = {int(g): i for i, g in enumerate(tr)}
        r0_in = {t: np.zeros(len(tr)) for t in TASKS}
        c_in = {n: {t: np.zeros(len(tr)) for t in TASKS} for n in names}
        for B, C in inner:
            for t in TASKS:
                p = r0.fit_r0_predict(t, B, C, seed, 700 + fold, cache_tag=f"s{seed}f{fold}i")
                for g, v in zip(C, p):
                    r0_in[t][pos[int(g)]] = v
            for n, fn in cands.items():
                for t in TASKS:
                    p = fn(B, C, Y[t][B])
                    for g, v in zip(C, p):
                        c_in[n][t][pos[int(g)]] = v
        w_best = {t: {} for t in TASKS}
        for t in TASKS:
            yin = Y[t][tr]
            for n in names:
                best = (-9.0, 0.0)
                for w in np.linspace(0.0, 1.0, 21):
                    p = (1 - w) * r0_in[t] + w * c_in[n][t]
                    r = np.corrcoef(p, yin)[0, 1]
                    if r > best[0]:
                        best = (r, w)
                w_best[t][n] = best[1]
                w_sel[t][n].append(round(best[1], 2))
        for t in TASKS:
            yV = Y[t][te]
            r0V = r0.fit_r0_predict(t, tr, te, seed, fold, cache_tag=f"s{seed}f{fold}o")
            r0_r[t].append(np.corrcoef(r0V, yV)[0, 1])
            for n, fn in cands.items():
                cV = fn(tr, te, Y[t][tr])
                st_r[t][n].append(np.corrcoef(cV, yV)[0, 1])
                w = w_best[t][n]
                bl_r[t][n].append(np.corrcoef((1 - w) * r0V + w * cV, yV)[0, 1])
        print(f"  fold {fold} [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n=== SCREEN2 seed {seed} ===")
    for t in TASKS:
        base = float(np.mean(r0_r[t]))
        print(f"\n{t}: R0 r = {base:.4f}")
        for n in names:
            s = float(np.mean(st_r[t][n]))
            b = float(np.mean(bl_r[t][n]))
            print(f"  {n:14s} standalone r={s:.4f} ({s-base:+.4f})  "
                  f"blended r={b:.4f} ({b-base:+.4f})  w_sel={w_sel[t][n]}")


if __name__ == "__main__":
    main()
