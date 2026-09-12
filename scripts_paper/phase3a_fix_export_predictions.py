#!/usr/bin/env python3
"""Phase 3A-FIX: export per-split test predictions from cached selection.

Refits only the selected final models per outer split (deterministic) and
writes predictions/ artifacts. Does not re-run candidate search.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_paper"))

import phase3a_fix_pilot as P  # noqa: E402

OUT = P.OUT
CKPT = P.CKPT


def main():
    FC, SC, fc_feat, sc_feat, y_wm, y_fi, subjects = P.load_dev_data()
    holdout = P.HOLDOUT_TXT.read_text().strip().split("\n")
    access = P.AccessLogger(OUT / "_export_access.jsonl")
    access.set_holdout(holdout)
    p_wm = P.load_prior(P.PRIOR_WM_PATH)
    p_fi = P.load_prior(P.PRIOR_FI_PATH)
    controls = P.make_controls(p_wm, p_fi)
    r0 = P.R0Baseline(fc_feat, sc_feat, y_wm, y_fi, np.array(subjects),
                      access_logger=access)

    rows = []
    t0 = time.time()
    for ck in sorted(CKPT.glob("split_seed*_fold*.pkl")):
        out = pickle.load(open(ck, "rb"))
        seed, fold = out["seed"], out["fold"]
        tr = np.array(out["train_idx"]); te = np.array(out["test_idx"])
        sel = out["selection"]

        cf_T = {t: r0.crossfit_r0_within(t, tr, seed, f"{seed}:{fold}:T")
                for t in ["WM", "FI"]}
        base = {t: r0.fit_r0_predict(t, tr, te, seed, fold,
                                     cache_tag=f"{seed}:{fold}:test")
                for t in ["WM", "FI"]}
        fcT, scT, fcV, scV = P.standardize_scope(FC[tr], SC[tr], FC[te], SC[te])
        rT_WM = y_wm[tr] - cf_T["WM"]; rT_FI = y_fi[tr] - cf_T["FI"]
        zT_WM = (rT_WM - rT_WM.mean()) / (rT_WM.std() + 1e-12)
        zT_FI = (rT_FI - rT_FI.mean()) / (rT_FI.std() + 1e-12)

        preds = {"A0": (base["WM"], base["FI"])}
        prior_sets = {
            "C0": (np.ones(116) / 116, np.ones(116) / 116, np.ones(116) / 116),
            "C1": (P.shared_prior(p_wm, p_fi), p_wm, p_fi),
            "C2": controls["cross"], "C3": controls["shuffled"], "C4": controls["random"],
        }
        for mid in ["C0", "C1", "C2", "C3", "C4"]:
            best = sel[mid][0]
            psh, pwm, pfi = prior_sets[mid]
            r = P.train_bcr(fcT, scT, zT_WM, zT_FI, psh, pwm, pfi,
                            best["lambda_amp"], best["lambda_prior"], kind="MT",
                            seed=seed)
            rW, rF = P.predict_bcr(r, fcV, scV)
            pW = base["WM"] + best["alpha_WM"] * (rW * rT_WM.std() + rT_WM.mean())
            pF = base["FI"] + best["alpha_FI"] * (rF * rT_FI.std() + rT_FI.mean())
            preds[mid] = (pW, pF)
        for mid in ["B0", "B1"]:
            use_prior = mid == "B1"
            for task in ["WM", "FI"]:
                best = sel[f"{mid}_{task}"][0]
                if task == "WM":
                    pwm_u = p_wm if use_prior else np.ones(116) / 116
                    psh, pfi = pwm_u, np.ones(116) / 116
                    r = P.train_bcr(fcT, scT, zT_WM, None, psh, pwm_u, pfi,
                                    best["lambda_amp"], best["lambda_prior"],
                                    kind="ST_WM", seed=seed)
                    rW, _ = P.predict_bcr(r, fcV, scV)
                    pW = base["WM"] + best["alpha"] * (rW * rT_WM.std() + rT_WM.mean())
                    pF = base["FI"]
                else:
                    pfi_u = p_fi if use_prior else np.ones(116) / 116
                    psh, pwm_u = np.ones(116) / 116, np.ones(116) / 116
                    r = P.train_bcr(fcT, scT, None, zT_FI, psh, pwm_u, pfi_u,
                                    best["lambda_amp"], best["lambda_prior"],
                                    kind="ST_FI", seed=seed)
                    _, rF = P.predict_bcr(r, fcV, scV)
                    pW = base["WM"]
                    pF = base["FI"] + best["alpha"] * (rF * rT_FI.std() + rT_FI.mean())
                preds.setdefault(mid, (base["WM"], base["FI"]))
                pw0, pf0 = preds[mid]
                preds[mid] = (pW if task == "WM" else pw0, pF if task == "FI" else pf0)

        sid = [subjects[i] for i in te]
        for mid, (pW, pF) in preds.items():
            for i in range(len(te)):
                rows.append({"seed": seed, "fold": fold, "model": mid,
                             "subject": sid[i], "y_WM": y_wm[te][i], "y_FI": y_fi[te][i],
                             "pred_WM": pW[i], "pred_FI": pF[i]})
        print(f"  seed={seed} fold={fold} exported ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "predictions" / "test_predictions.csv", index=False)
    # per-split files
    pdir = OUT / "predictions"
    for (seed, fold), sub in df.groupby(["seed", "fold"]):
        sub.to_csv(pdir / f"pred_seed{seed}_fold{fold}.csv", index=False)
    access.assert_no_holdout()
    print(f"Exported {len(df)} prediction rows.")


if __name__ == "__main__":
    main()
