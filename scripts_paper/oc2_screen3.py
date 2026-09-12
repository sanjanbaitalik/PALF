#!/usr/bin/env python3
"""OpenChallenge final targeted screen: remaining untried families.

Candidates: data-driven edge clusters (ridge on cluster means), per-subject
ridge normalization, multi-alpha ridge averaging, count GLM on low-dim
components, and SC-rank features. Honest nested blending with cached R0.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oc2_bank import (FC_E, SC_E, Y, N, TASKS, get_r0_cache,  # noqa: E402
                      partition, pred_ridge)

CACHE = ROOT / "outputs" / "iclr" / "_oc2_cache"


def per_subject_ridge(tr, te, y, alpha=1000.0):
    """Z-score each subject's edge vector (shape removed), then ridge."""
    def z(A):
        mu = A.mean(1, keepdims=True)
        sd = A.std(1, keepdims=True)
        return (A - mu) / (sd + 1e-12)
    Xtr = np.hstack([z(FC_E[tr]), z(SC_E[tr])])
    Xte = np.hstack([z(FC_E[te]), z(SC_E[te])])
    return pred_ridge(Xtr, y, Xte, alpha)


def edge_cluster_ridge(tr, te, y, K=200, alpha=10.0):
    """Cluster edges by subject-response profile; ridge on cluster means."""
    X = np.hstack([FC_E, SC_E])                       # 412 x 13340
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    km = KMeans(n_clusters=K, n_init=3, random_state=0).fit(Z.T)
    lab = km.labels_
    Ctr = np.zeros((len(tr), K)); Cte = np.zeros((len(te), K))
    for k in range(K):
        m = lab == k
        Ctr[:, k] = Z[tr][:, m].mean(1)
        Cte[:, k] = Z[te][:, m].mean(1)
    return pred_ridge(Ctr, y, Cte, alpha)


def multi_alpha_ridge(tr, te, y, alphas=(3, 10, 30, 100, 300, 1000, 3000)):
    Xtr = np.hstack([FC_E[tr], SC_E[tr]])
    Xte = np.hstack([FC_E[te], SC_E[te]])
    preds = []
    for a in alphas:
        preds.append(pred_ridge(Xtr, y, Xte, a))
    return np.mean(preds, 0)


def count_glm_components(tr, te, y, rank=50):
    """GLM-like count model: Poisson loss GBDT on PCA components."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xtr = np.hstack([FC_E[tr], SC_E[tr]])
    Xte = np.hstack([FC_E[te], SC_E[te]])
    s = StandardScaler().fit(Xtr)
    p = PCA(n_components=rank, svd_solver="randomized", random_state=0).fit(s.transform(Xtr))
    A = p.transform(s.transform(Xtr)); B = p.transform(s.transform(Xte))
    m = HistGradientBoostingRegressor(loss="poisson", max_iter=120,
                                      learning_rate=0.05, max_depth=3,
                                      min_samples_leaf=30, random_state=0)
    m.fit(A, y)
    return m.predict(B)


def sc_rank_ridge(tr, te, y, alpha=1000.0):
    """Ridge on within-subject rank-normalized SC edges."""
    from scipy.stats import norm
    def rn(A):
        r = np.argsort(np.argsort(A, axis=1), axis=1).astype(np.float64)
        return norm.ppf((r + 0.5) / A.shape[1])
    return pred_ridge(rn(SC_E[tr]), y, rn(SC_E[te]), alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="11,12,13")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    cands = {
        "persubj": lambda tr, te, y: per_subject_ridge(tr, te, y),
        "eclus50": lambda tr, te, y: edge_cluster_ridge(tr, te, y, K=50),
        "eclus200": lambda tr, te, y: edge_cluster_ridge(tr, te, y, K=200),
        "malpha": lambda tr, te, y: multi_alpha_ridge(tr, te, y),
        "countglm": lambda tr, te, y: count_glm_components(tr, te, y),
        "scrank": lambda tr, te, y: sc_rank_ridge(tr, te, y),
    }
    names = list(cands)

    for seed in seeds:
        print(f"\n### seed {seed}", flush=True)
        cache = get_r0_cache(seed)
        splits = cache["splits"]
        res = {t: {"R0": []} for t in TASKS}
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
                    best = (-9.0, 0.0)
                    for w in np.linspace(0.0, 1.0, 21):
                        r = np.corrcoef((1 - w) * r0in + w * c_in[n][t], yin)[0, 1]
                        if r > best[0]:
                            best = (r, w)
                    wsel[t][n].append(round(best[1], 2))
                    blended = (1 - best[1]) * cache["outer"][(fold, t)] + best[1] * p_out
                    bl[t][n].append(np.corrcoef(blended, Y[t][te])[0, 1])
            print(f"  fold {fold} [{time.time()-t0:.0f}s]", flush=True)
        for t in TASKS:
            base = float(np.mean(res[t]["R0"]))
            print(f"\n{t}: R0 = {base:.4f}")
            for n in names:
                b = float(np.mean(bl[t][n]))
                print(f"  {n:10s} blended {b-base:+.4f}  w={wsel[t][n]}")


if __name__ == "__main__":
    main()
