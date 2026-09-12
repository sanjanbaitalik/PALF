#!/usr/bin/env python3
"""OpenChallenge candidate bank with cached R0 predictions.

Caches R0 inner-OOF and outer-fold predictions per screening seed, then
evaluates a bank of distinct candidate predictors with honest nested blending.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
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
CACHE = ROOT / "outputs" / "iclr" / "_oc2_cache"
CACHE.mkdir(parents=True, exist_ok=True)


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


def outer_folds(seed, k=5):
    idx = np.random.RandomState(seed).permutation(N)
    sizes = np.full(k, N // k)
    sizes[: N % k] += 1
    out, cur = [], 0
    for i in range(k):
        a, b = cur, cur + sizes[i]
        out.append((idx[a:b], np.concatenate([idx[:a], idx[b:]])))
        cur = b
    return out


def get_r0_cache(seed):
    path = CACHE / f"r0_seed{seed}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    r0 = R0Baseline(FC_E, SC_E, Y["WM"], Y["FI"], np.array([str(i) for i in range(N)]))
    cache = {"inner": {}, "outer": {}, "splits": []}
    for fold, (te, tr) in enumerate(outer_folds(seed)):
        cache["splits"].append((te.copy(), tr.copy()))
        inner = partition(tr, seed, f"outer{fold}", 3)
        pos = {int(g): i for i, g in enumerate(tr)}
        for t in TASKS:
            oof = np.zeros(len(tr))
            for B, C in inner:
                p = r0.fit_r0_predict(t, B, C, seed, 700 + fold,
                                      cache_tag=f"bank{seed}f{fold}i")
                for g, v in zip(C, p):
                    oof[pos[int(g)]] = v
            cache["inner"][(fold, t)] = oof
            cache["outer"][(fold, t)] = r0.fit_r0_predict(
                t, tr, te, seed, fold, cache_tag=f"bank{seed}f{fold}o")
        print(f"  R0 cached fold {fold}", flush=True)
    pickle.dump(cache, open(path, "wb"))
    return cache


def pred_ridge(Xtr, ytr, Xte, alpha=1000.0):
    s = StandardScaler().fit(Xtr)
    return Ridge(alpha=alpha).fit(s.transform(Xtr), ytr).predict(s.transform(Xte))


def pred_knn(Xtr, ytr, Xte, K=5, k=40, alpha=1000.0):
    s = StandardScaler().fit(Xtr)
    A = s.transform(Xtr)
    B = s.transform(Xte)
    pls = PLSRegression(n_components=K, scale=False).fit(A, ytr)
    S = pls.transform(A)
    St = pls.transform(B)
    st = StandardScaler().fit(S)
    return KNeighborsRegressor(n_neighbors=k, weights="distance").fit(
        st.transform(S), ytr).predict(st.transform(St))


def pred_rbf(Xtr, ytr, Xte, mult=1.0, lam=3.0):
    s = StandardScaler().fit(Xtr)
    A = s.transform(Xtr); B = s.transform(Xte)
    G = A @ A.T
    d2 = np.maximum(np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G, 0)
    sig = mult * np.median(np.sqrt(d2[d2 > 0]))
    K = np.exp(-d2 / (2 * sig ** 2))
    a = np.linalg.solve(K + lam * np.eye(len(ytr)), ytr - ytr.mean())
    Gt = B @ A.T
    d2t = np.maximum((B ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :] - 2 * Gt, 0)
    return np.exp(-d2t / (2 * sig ** 2)) @ a + ytr.mean()


def pred_pcr(Xtr, ytr, Xte, rank=100):
    s = StandardScaler().fit(Xtr)
    A = s.transform(Xtr); B = s.transform(Xte)
    rank = int(min(rank, A.shape[0] - 1))
    p = PCA(n_components=rank, svd_solver="randomized", random_state=0).fit(A)
    return Ridge(alpha=1.0).fit(p.transform(A), ytr).predict(p.transform(B))


def pred_gbdt(Xtr, ytr, Xte, n_iter=120):
    s = StandardScaler().fit(Xtr)
    m = HistGradientBoostingRegressor(max_iter=n_iter, learning_rate=0.05,
                                      max_depth=3, min_samples_leaf=30,
                                      l2_regularization=1.0, random_state=0)
    m.fit(s.transform(Xtr), ytr)
    return m.predict(s.transform(Xte))


def sc_transform(kind):
    if kind == "raw":
        return SC_E
    if kind == "log":
        return np.log1p(SC_E)
    if kind == "sqrt":
        return np.sqrt(SC_E)
    if kind == "bin":
        return (SC_E > 0).astype(np.float64)
    raise ValueError(kind)


def sc_diffusion(tau):
    out = np.zeros((N, len(IU[0])))
    for s in range(N):
        W = np.clip(SC[s], 0, None).copy()
        np.fill_diagonal(W, 0)
        d = W.sum(1); d[d <= 1e-12] = 1e-12
        L = np.diag(d) - W
        w, V = np.linalg.eigh(np.diag(1 / np.sqrt(d)) @ L @ np.diag(1 / np.sqrt(d)))
        S = (V * np.exp(-tau * w)) @ V.T
        out[s] = (S @ SC[s] @ S)[IU]
    return out


def node_topo(M):
    out = np.zeros((len(M), 116 * 4))
    for s, m in enumerate(M):
        W = np.clip(m, 0, None).copy(); np.fill_diagonal(W, 0)
        st = W.sum(1)
        Wt = W ** (1 / 3)
        tri = np.einsum("ij,jk,ik->i", Wt, Wt, Wt)
        deg = (W > 0).sum(1).astype(float)
        clus = np.where(deg > 1, tri / np.maximum(deg * (deg - 1), 1), 0.0)
        eigc = np.abs(np.linalg.eigh(W)[1][:, -1])
        out[s] = np.concatenate([st, clus, eigc, np.log1p(st)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="11,12,13")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("building global feature transforms (unsupervised, no labels)...", flush=True)
    DB = {"sd1": sc_diffusion(1.0), "sd2": sc_diffusion(2.0)}
    TO = {"fc": node_topo(FC), "sc": node_topo(SC)}
    print("  done", flush=True)

    # candidate bank: name -> (view builder)
    cands = {}
    cands["ridge_fc"] = lambda tr, te, y: pred_ridge(FC_E[tr], y, FC_E[te])
    cands["ridge_sc"] = lambda tr, te, y: pred_ridge(SC_E[tr], y, SC_E[te])
    cands["ridge_joint"] = lambda tr, te, y: pred_ridge(
        np.hstack([FC_E[tr], SC_E[tr]]), y, np.hstack([FC_E[te], SC_E[te]]))
    cands["ridge_sc_log"] = lambda tr, te, y: pred_ridge(
        sc_transform("log")[tr], y, sc_transform("log")[te])
    cands["ridge_sc_bin"] = lambda tr, te, y: pred_ridge(
        sc_transform("bin")[tr], y, sc_transform("bin")[te])
    cands["ridge_sc_sqrt"] = lambda tr, te, y: pred_ridge(
        sc_transform("sqrt")[tr], y, sc_transform("sqrt")[te])
    cands["ridge_fc_sclog"] = lambda tr, te, y: pred_ridge(
        np.hstack([FC_E[tr], sc_transform("log")[tr]]), y,
        np.hstack([FC_E[te], sc_transform("log")[te]]))
    cands["knn40"] = lambda tr, te, y: pred_knn(FC_E, None, None) if False else pred_knn(
        np.hstack([FC_E[tr], SC_E[tr]]), y, np.hstack([FC_E[te], SC_E[te]]), k=40)
    cands["knn20"] = lambda tr, te, y: pred_knn(
        np.hstack([FC_E[tr], SC_E[tr]]), y, np.hstack([FC_E[te], SC_E[te]]), k=20)
    cands["knn_sc"] = lambda tr, te, y: pred_knn(SC_E[tr], y, SC_E[te], k=40)
    cands["rbf_joint"] = lambda tr, te, y: pred_rbf(
        np.hstack([FC_E[tr], SC_E[tr]]), y, np.hstack([FC_E[te], SC_E[te]]))
    cands["rbf_sc"] = lambda tr, te, y: pred_rbf(SC_E[tr], y, SC_E[te])
    cands["pcr100"] = lambda tr, te, y: pred_pcr(
        np.hstack([FC_E[tr], SC_E[tr]]), y, np.hstack([FC_E[te], SC_E[te]]), rank=100)
    cands["pcr300"] = lambda tr, te, y: pred_pcr(
        np.hstack([FC_E[tr], SC_E[tr]]), y, np.hstack([FC_E[te], SC_E[te]]), rank=300)
    cands["gbdt"] = lambda tr, te, y: pred_gbdt(
        np.hstack([FC_E[tr], SC_E[tr]]), y, np.hstack([FC_E[te], SC_E[te]]))
    cands["sc_diff1"] = lambda tr, te, y: pred_ridge(
        np.hstack([DB["sd1"][tr], FC_E[tr]]), y, np.hstack([DB["sd1"][te], FC_E[te]]))
    cands["sc_diff2"] = lambda tr, te, y: pred_ridge(
        np.hstack([DB["sd2"][tr], FC_E[tr]]), y, np.hstack([DB["sd2"][te], FC_E[te]]))
    cands["topo"] = lambda tr, te, y: pred_ridge(
        np.hstack([TO["fc"][tr], TO["sc"][tr]]), y, np.hstack([TO["fc"][te], TO["sc"][te]]))
    names = list(cands)

    summary = {}
    for seed in seeds:
        print(f"\n### seed {seed}", flush=True)
        cache = get_r0_cache(seed)
        splits = cache["splits"]
        res = {t: {"R0": [], **{n: [] for n in names}} for t in TASKS}
        bl = {t: {n: [] for n in names} for t in TASKS}
        wsel = {t: {n: [] for n in names} for t in TASKS}
        for fold, (te, tr) in enumerate(splits):
            t0 = time.time()
            inner = partition(tr, seed, f"outer{fold}", 3)
            pos = {int(g): i for i, g in enumerate(tr)}
            c_in = {n: {t: np.zeros(len(tr)) for t in TASKS} for n in names}
            for n, fn in cands.items():
                for B, C in inner:
                    for t in TASKS:
                        p = fn(B, C, Y[t][B])
                        for g, v in zip(C, p):
                            c_in[n][t][pos[int(g)]] = v
            for t in TASKS:
                yin = Y[t][tr]
                r0in = cache["inner"][(fold, t)]
                res[t]["R0"].append(np.corrcoef(cache["outer"][(fold, t)], Y[t][te])[0, 1])
                for n in names:
                    p_out = cands[n](tr, te, Y[t][tr])
                    res[t][n].append(np.corrcoef(p_out, Y[t][te])[0, 1])
                    best = (-9.0, 0.0)
                    for w in np.linspace(0.0, 1.0, 21):
                        r = np.corrcoef((1 - w) * r0in + w * c_in[n][t], yin)[0, 1]
                        if r > best[0]:
                            best = (r, w)
                    wsel[t][n].append(round(best[1], 2))
                    blended = (1 - best[1]) * cache["outer"][(fold, t)] + best[1] * p_out
                    bl[t][n].append(np.corrcoef(blended, Y[t][te])[0, 1])
            print(f"  fold {fold} [{time.time()-t0:.0f}s]", flush=True)
        summary[seed] = {}
        for t in TASKS:
            base = float(np.mean(res[t]["R0"]))
            summary[seed][t] = {"R0": base}
            print(f"\n{t}: R0 = {base:.4f}")
            for n in names:
                s = float(np.mean(res[t][n]))
                b = float(np.mean(bl[t][n]))
                summary[seed][t][n] = {"standalone": s, "blended": b,
                                       "delta_s": s - base, "delta_b": b - base}
                print(f"  {n:14s} standalone {s-base:+.4f}  blended {b-base:+.4f}  "
                      f"w={wsel[t][n]}")
        pickle.dump(summary, open(CACHE / "summary_bank.pkl", "wb"))


if __name__ == "__main__":
    main()
