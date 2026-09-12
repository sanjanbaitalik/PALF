#!/usr/bin/env python3
"""Independent OpenChallenge screen.

Honest nested protocol: within each outer-training scope, select the blend
weight of each candidate predictor with R0 on a 3-fold inner OOF, then
evaluate the blended predictor on the untouched outer-test fold.
Screening seeds are fresh (default 11) and are NOT the locked seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
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
N = 412
IU = np.triu_indices(116, 1)
FC_E = FC[:, IU[0], IU[1]]
SC_E = SC[:, IU[0], IU[1]]
TASKS = ("WM", "FI")


def partition(scope, seed, tag, k=3):
    h = int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest()[:8], 16)
    perm = np.random.RandomState(h).permutation(len(scope))
    sizes = np.full(k, len(scope) // k)
    sizes[: len(scope) % k] += 1
    folds, cur = [], 0
    for i in range(k):
        a, b = cur, cur + sizes[i]
        c = perm[a:b]
        d = np.concatenate([perm[:a], perm[b:]])
        folds.append((scope[d], scope[c]))
        cur = b
    return folds


def norm_lap(W):
    W = np.clip(W, 0, None).copy()
    np.fill_diagonal(W, 0)
    d = W.sum(1)
    d[d <= 1e-12] = 1e-12
    Dm = np.diag(1.0 / np.sqrt(d))
    return np.eye(116) - Dm @ W @ Dm


def expm_neg(L, t):
    w, V = np.linalg.eigh(L)
    return (V * np.exp(-t * w)) @ V.T


def build_spec(taus):
    out = {}
    for t in taus:
        A = np.zeros((N, len(IU[0])))
        for s in range(N):
            S = expm_neg(norm_lap(SC[s]), t)
            Fs = S @ FC[s] @ S
            A[s] = Fs[IU]
        out[t] = A
    return out


def pred_ridge(Xtr, ytr, Xte, alpha=1000.0):
    s = StandardScaler().fit(Xtr)
    return Ridge(alpha=alpha).fit(s.transform(Xtr), ytr).predict(s.transform(Xte))


def pred_ridge_joint(fc_tr, sc_tr, ytr, fc_te, sc_te, alpha=1000.0):
    return pred_ridge(np.hstack([fc_tr, sc_tr]), ytr,
                      np.hstack([fc_te, sc_te]), alpha)


def pred_cvp(fc_tr, sc_tr, ytr, fc_te, sc_te, K=5):
    """Cross-view PLS: FC<->SC latent scores, ridge on scores."""
    sf = StandardScaler().fit(fc_tr)
    ss = StandardScaler().fit(sc_tr)
    Xf = sf.transform(fc_tr)
    Xs = ss.transform(sc_tr)
    pls = PLSRegression(n_components=K, scale=False).fit(Xf, Xs)
    Ttr = pls.x_scores_
    Utr = pls.y_scores_
    Tte = sf.transform(fc_te) @ pls.x_rotations_
    Ute = ss.transform(sc_te) @ pls.y_rotations_
    st = StandardScaler().fit(np.hstack([Ttr, Utr]))
    Ztr = st.transform(np.hstack([Ttr, Utr]))
    Zte = st.transform(np.hstack([Tte, Ute]))
    return Ridge(alpha=10.0).fit(Ztr, ytr).predict(Zte)


def pred_spls(fc_tr, sc_tr, ytr, fc_te, sc_te, K=5):
    """Supervised PLS regression on concatenated FC+SC."""
    Xtr = np.hstack([fc_tr, sc_tr])
    Xte = np.hstack([fc_te, sc_te])
    s = StandardScaler().fit(Xtr)
    pls = PLSRegression(n_components=K, scale=False).fit(s.transform(Xtr), ytr)
    return pls.predict(s.transform(Xte))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    seed = args.seed

    print("building structural-diffusion features...", flush=True)
    t0 = time.time()
    spec = build_spec([1.0, 5.0])
    print(f"  done in {time.time()-t0:.0f}s", flush=True)

    cands = {
        "ridge_joint": lambda tr, te, ytr: pred_ridge_joint(
            FC_E[tr], SC_E[tr], ytr, FC_E[te], SC_E[te]),
        "cvp2": lambda tr, te, ytr: pred_cvp(
            FC_E[tr], SC_E[tr], ytr, FC_E[te], SC_E[te], K=2),
        "cvp5": lambda tr, te, ytr: pred_cvp(
            FC_E[tr], SC_E[tr], ytr, FC_E[te], SC_E[te], K=5),
        "spls2": lambda tr, te, ytr: pred_spls(
            FC_E[tr], SC_E[tr], ytr, FC_E[te], SC_E[te], K=2),
        "spls5": lambda tr, te, ytr: pred_spls(
            FC_E[tr], SC_E[tr], ytr, FC_E[te], SC_E[te], K=5),
        "spls10": lambda tr, te, ytr: pred_spls(
            FC_E[tr], SC_E[tr], ytr, FC_E[te], SC_E[te], K=10),
        "spec1": lambda tr, te, ytr: pred_ridge(
            np.hstack([spec[1.0][tr], SC_E[tr]]), ytr,
            np.hstack([spec[1.0][te], SC_E[te]])),
        "spec5": lambda tr, te, ytr: pred_ridge(
            np.hstack([spec[5.0][tr], SC_E[tr]]), ytr,
            np.hstack([spec[5.0][te], SC_E[te]])),
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
                p = r0.fit_r0_predict(t, B, C, seed, 700 + fold,
                                      cache_tag=f"s{seed}f{fold}i")
                for g, v in zip(C, p):
                    r0_in[t][pos[int(g)]] = v
            for n, fn in cands.items():
                for t in TASKS:
                    p = fn(B, C, Y[t][B])
                    for g, v in zip(C, p):
                        c_in[n][t][pos[int(g)]] = v
        # select blend weights on inner OOF
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
        # outer evaluation
        for t in TASKS:
            yV = Y[t][te]
            r0V = r0.fit_r0_predict(t, tr, te, seed, fold,
                                    cache_tag=f"s{seed}f{fold}o")
            r0_r[t].append(np.corrcoef(r0V, yV)[0, 1])
            for n, fn in cands.items():
                cV = fn(tr, te, Y[t][tr])
                st_r[t][n].append(np.corrcoef(cV, yV)[0, 1])
                w = w_best[t][n]
                p = (1 - w) * r0V + w * cV
                bl_r[t][n].append(np.corrcoef(p, yV)[0, 1])
        print(f"  fold {fold} [{time.time()-t0:.0f}s]", flush=True)

    print("\n=== SCREEN seed", seed, "(mean over folds) ===")
    for t in TASKS:
        base = float(np.mean(r0_r[t]))
        print(f"\n{t}: R0 r = {base:.4f}")
        for n in names:
            s = float(np.mean(st_r[t][n]))
            b = float(np.mean(bl_r[t][n]))
            print(f"  {n:12s} standalone r={s:.4f} ({s-base:+.4f})  "
                  f"blended r={b:.4f} ({b-base:+.4f})  w_sel={w_sel[t][n]}")


if __name__ == "__main__":
    main()
