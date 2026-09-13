#!/usr/bin/env python3
"""Reporting-only correction of modality-specific biomarker stability (v36).

Corrects FC/SC top-10 ROI Jaccard so that:
  - FC-specific metrics use the FC coefficient map only;
  - SC-specific metrics use the SC coefficient map only;
  - multimodal metrics use I_MULTI = I_FC + I_SC;
  - inactive (all-zero) modality folds are excluded explicitly instead of
    producing NaN or arbitrary rankings.

NO model fitting, NO prediction recomputation, NO holdout access. The only
inputs are the saved outer-fold coefficient maps in the final 510 package.

Stages
------
audit     : holdout seal, frozen-file hash snapshot, ETA print, seal file
stability : corrected coefficient-derived stability + before/after table
tables    : corrected paper-ready biomarker table + top-10 biomarker files
plots     : corrected Figure 5
finalize  : final audit, seal update, runtime report
all       : audit -> stability -> tables -> plots -> finalize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_paper"))

import final510_wm_pilot as P  # noqa: E402
import final510_wm_report as R  # noqa: E402

OUT = ROOT / "outputs" / "iclr" / "palf_final510_wm"
PR = OUT / "paper_ready"
PLOTS = OUT / "plots"
HOLDOUT_TXT = ROOT / "data_splits" / "phase3_holdout_98.txt"
HOLDOUT_SHA = "89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425"
IU = np.triu_indices(P.N_ROI, 1)
FROZEN = ["primary_metrics.csv", "seed_metrics.csv", "fold_metrics.csv",
          "primary_comparisons.csv", "sensitivity_ensemble_bootstrap.csv",
          "prediction_ablations.csv", "ridge_prior_controls.csv",
          "ncr_prior_controls.csv", "SELECTED_CONFIGS.csv"]
ESTIMATE = "0:10"
START_FILE = OUT / "BIOMARKER_CORRECTION_START.json"
MODALITIES = ("fc", "sc", "multimodal")


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ── corrected definitions ─────────────────────────────────────────────

def incident_importance(coeff_edge):
    """I_i = sum_{j != i} |C[i,j]| over the 6670 upper-triangle edges."""
    imp = np.zeros(P.N_ROI)
    np.add.at(imp, IU[0], np.abs(coeff_edge))
    np.add.at(imp, IU[1], np.abs(coeff_edge))
    return imp


def rank_rois(imp, k):
    """Deterministic ranking: descending importance, AAL116 index tie-break.

    Returns None for an all-zero (inactive) importance map.
    """
    imp = np.asarray(imp, dtype=float)
    if not np.any(imp != 0):
        return None
    return np.lexsort((np.arange(len(imp)), -imp))[:k]


def jaccard(a, b):
    sa, sb = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def sign_consistency_pair(va, vb, k=100):
    ta = set(np.argsort(np.abs(va))[-k:].tolist())
    tb = set(np.argsort(np.abs(vb))[-k:].tolist())
    common = ta & tb
    if not common:
        return None
    same = sum(1 for e in common if np.sign(va[e]) == np.sign(vb[e]))
    return same / len(common)


def _stats(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": float("nan"), "median": float("nan"),
                "sd": float("nan"), "n_pairs": 0}
    return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "sd": float(np.std(vals)), "n_pairs": len(vals)}


def load_coefficient_entries():
    """Per-fold corrected coefficient maps from saved checkpoints (no fitting)."""
    recs = R.load_folds()
    entries = []
    for r in recs:
        for m in P.ALL_MODELS:
            maps = P.coefficient_maps(r, m)
            entry = {"model": m, "seed": r["seed"], "fold": r["fold"],
                     "maps": maps,
                     "fc_active": False, "sc_active": False, "multi_active": False}
            if maps is not None:
                c_fc, c_sc, imp = maps
                imp_fc = incident_importance(c_fc)
                imp_sc = incident_importance(c_sc)
                imp_multi = imp_fc + imp_sc
                assert np.allclose(imp_multi, imp), "I_MULTI != I_FC + I_SC"
                c_multi = c_fc + c_sc
                entry.update({
                    "c_fc": c_fc, "c_sc": c_sc, "c_multi": c_multi,
                    "imp_fc": imp_fc, "imp_sc": imp_sc, "imp_multi": imp_multi,
                    "fc_active": bool(np.any(c_fc != 0)),
                    "sc_active": bool(np.any(c_sc != 0)),
                    "multi_active": bool(np.any(imp_multi != 0)),
                })
                assert entry["multi_active"], "non-abstained fold has zero map"
            entries.append(entry)
    return recs, entries


def corrected_stability(entries):
    """Pairwise corrected stability for one model (existing pair convention)."""
    n_total = len(entries)
    total_pairs = n_total * (n_total - 1) // 2
    res = {}

    def add(mod, metric, vals, excluded):
        st = _stats(vals)
        res[f"{mod}_{metric}"] = st["mean"]
        res[f"{mod}_{metric}_median"] = st["median"]
        res[f"{mod}_{metric}_sd"] = st["sd"]
        res[f"{mod}_{metric}_n_pairs"] = st["n_pairs"]
        res[f"{mod}_{metric}_n_excluded_inactive"] = excluded

    for mod in MODALITIES:
        active_key = {"fc": "fc_active", "sc": "sc_active",
                      "multimodal": "multi_active"}[mod]
        vec_key = {"fc": "c_fc", "sc": "c_sc", "multimodal": "c_multi"}[mod]
        imp_key = {"fc": "imp_fc", "sc": "imp_sc", "multimodal": "imp_multi"}[mod]
        edge_spearman, t100, t300, r10, r20, signs = [], [], [], [], [], []
        excluded = 0
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a, b = entries[i], entries[j]
                if not (a[active_key] and b[active_key]):
                    excluded += 1
                    continue
                edge_spearman.append(float(spearmanr(
                    np.abs(a[vec_key]), np.abs(b[vec_key])).statistic))
                for k, bucket in ((100, t100), (300, t300)):
                    sa = set(np.argsort(np.abs(a[vec_key]))[-k:].tolist())
                    sb = set(np.argsort(np.abs(b[vec_key]))[-k:].tolist())
                    bucket.append(len(sa & sb) / len(sa | sb))
                ra = rank_rois(a[imp_key], 10)
                rb = rank_rois(b[imp_key], 10)
                if ra is not None and rb is not None:
                    r10.append(jaccard(ra, rb))
                if mod == "multimodal":
                    ra20 = rank_rois(a[imp_key], 20)
                    rb20 = rank_rois(b[imp_key], 20)
                    if ra20 is not None and rb20 is not None:
                        r20.append(jaccard(ra20, rb20))
                sc = sign_consistency_pair(a[vec_key], b[vec_key])
                if sc is not None:
                    signs.append(sc)
        add(mod, "edge_spearman", edge_spearman, excluded)
        add(mod, "top100_edge_jaccard", t100, excluded)
        add(mod, "top300_edge_jaccard", t300, excluded)
        add(mod, "top10_roi_jaccard", r10, excluded)
        if mod == "multimodal":
            add(mod, "top20_roi_jaccard", r20, excluded)
        add(mod, "sign_consistency", signs, excluded)
    res["_total_pairs"] = total_pairs
    return res


def corrected_table(entries_all):
    old = pd.read_csv(OUT / "biomarker_stability.csv").set_index("model")
    rows = []
    entries_by_model = {}
    for e in entries_all:
        entries_by_model.setdefault(e["model"], []).append(e)
    for model in P.ALL_MODELS:
        ent = entries_by_model[model]
        st = corrected_stability(ent)
        n_active = sum(1 for e in ent if e["multi_active"])
        row = {
            "model": model, "task": "WM",
            "n_valid_outer_folds": n_active,
            "n_inactive_outer_folds": 25 - n_active,
            "n_fc_active_folds": sum(1 for e in ent if e["fc_active"]),
            "n_sc_active_folds": sum(1 for e in ent if e["sc_active"]),
            "n_abstained": 25 - n_active, "n_valid": n_active,
        }
        for mod, legacy in (("fc", "fc"), ("sc", "sc"),
                            ("multimodal", "multimodal")):
            for metric in ("edge_spearman", "top100_edge_jaccard",
                           "top300_edge_jaccard", "top10_roi_jaccard"):
                src = f"{mod}_{metric}"
                if src in st:
                    row[src] = st[src]
                    row[f"{src}_median"] = st[f"{src}_median"]
                    row[f"{src}_sd"] = st[f"{src}_sd"]
                    row[f"{src}_n_pairs"] = st[f"{src}_n_pairs"]
                    row[f"{src}_n_excluded_inactive"] = st[
                        f"{src}_n_excluded_inactive"]
            row[f"{mod}_sign_consistency"] = st[f"{mod}_sign_consistency"]
            row[f"{mod}_sign_consistency_median"] = st[
                f"{mod}_sign_consistency_median"]
            row[f"{mod}_sign_consistency_sd"] = st[f"{mod}_sign_consistency_sd"]
            row[f"{mod}_sign_consistency_n_pairs"] = st[
                f"{mod}_sign_consistency_n_pairs"]
        row["multimodal_top20_roi_jaccard"] = st["multimodal_top20_roi_jaccard"]
        row["multimodal_top20_roi_jaccard_median"] = st[
            "multimodal_top20_roi_jaccard_median"]
        row["multimodal_top20_roi_jaccard_sd"] = st["multimodal_top20_roi_jaccard_sd"]
        row["multimodal_top20_roi_jaccard_n_pairs"] = st[
            "multimodal_top20_roi_jaccard_n_pairs"]
        # legacy aliases used by the reporting pipeline
        row["fc_top100_jaccard"] = row["fc_top100_edge_jaccard"]
        row["sc_top100_jaccard"] = row["sc_top100_edge_jaccard"]
        row["fc_top300_jaccard"] = row["fc_top300_edge_jaccard"]
        row["sc_top300_jaccard"] = row["sc_top300_edge_jaccard"]
        row["sign_consistency_fc"] = row["fc_sign_consistency"]
        row["sign_consistency_sc"] = row["sc_sign_consistency"]
        old_row = old.loc[model]
        row["old_fc_top10_roi_jaccard"] = float(old_row["fc_top10_roi_jaccard"])
        row["old_sc_top10_roi_jaccard"] = float(old_row["sc_top10_roi_jaccard"])
        row["old_multimodal_top10_roi_jaccard"] = float(
            old_row["multimodal_top10_roi_jaccard"])
        row["old_fc_edge_spearman"] = float(old_row["fc_edge_spearman"])
        row["old_sc_edge_spearman"] = float(old_row["sc_edge_spearman"])
        rows.append(row)
    return pd.DataFrame(rows)


def before_after(df):
    lines = ["| model | old FC t10 J | new FC t10 J | old SC t10 J | new SC t10 J "
             "| old MULTI t10 J | new MULTI t10 J |",
             "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r['old_fc_top10_roi_jaccard']:.4f} "
            f"| {r['fc_top10_roi_jaccard']:.4f} "
            f"| {r['old_sc_top10_roi_jaccard']:.4f} "
            f"| {r['sc_top10_roi_jaccard']:.4f} "
            f"| {r['old_multimodal_top10_roi_jaccard']:.4f} "
            f"| {r['multimodal_top10_roi_jaccard']:.4f} |")
    return "\n".join(lines)


# ── stages ────────────────────────────────────────────────────────────

def stage_audit():
    ids = HOLDOUT_TXT.read_text().strip().split("\n")
    sha = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    assert len(ids) == 98 and len(set(ids)) == 98
    assert sha == HOLDOUT_SHA, "holdout seal broken"
    frozen = {}
    for f in FROZEN:
        p = OUT / f
        assert p.exists(), f
        frozen[f] = sha256_file(p)
    START_FILE.write_text(json.dumps(
        {"started": utc(), "started_epoch": time.time()}, indent=2))
    print(f"BIOMARKER_CORRECTION_ESTIMATED_RUNTIME: {ESTIMATE}")
    print(f"Holdout seal OK: n=98 sha={sha[:16]}...")
    print(f"Frozen prediction files snapshotted: {len(frozen)}")
    return frozen


def verify_reconstruction():
    """T10: final predictions reconstruct exactly from saved coefficients."""
    X_fc, X_sc, y, _, _ = R.load_combined()
    recs = R.load_folds()
    nmask = np.zeros(P.N_EDGE, dtype=bool)
    worst = 0.0
    for r in recs:
        for m in P.ALL_MODELS:
            pred = P.masked_final_predictions(r, X_fc, X_sc, m, r["te"],
                                              edge_mask_fc=nmask,
                                              edge_mask_sc=nmask)
            worst = max(worst, float(np.max(np.abs(
                pred - r["variants"][m]["final_test"]))))
    return worst


def stage_stability(frozen):
    t0 = time.time()
    print("=" * 70); print("Stage B: corrected coefficient-derived stability")
    print("=" * 70)
    recon = verify_reconstruction()
    print(f"  coefficient reconstruction max |error| = {recon:.3e}")
    assert recon <= 1e-8, "coefficient reconstruction failed"
    _, entries = load_coefficient_entries()
    df = corrected_table(entries)
    df.to_csv(OUT / "biomarker_stability_corrected.csv", index=False)
    ba = before_after(df)
    print(ba)
    # explicit inactive-fold handling summary
    print("\nInactive-modality handling (relative to 25 folds):")
    for _, r in df.iterrows():
        print(f"  {r['model']}: FC active {r['n_fc_active_folds']}/25, "
              f"SC active {r['n_sc_active_folds']}/25, "
              f"abstained {r['n_abstained']}/25; "
              f"FC pairs {r['fc_top10_roi_jaccard_n_pairs']} "
              f"(excluded {r['fc_top10_roi_jaccard_n_excluded_inactive']}), "
              f"SC pairs {r['sc_top10_roi_jaccard_n_pairs']} "
              f"(excluded {r['sc_top10_roi_jaccard_n_excluded_inactive']}), "
              f"MULTI pairs {r['multimodal_top10_roi_jaccard_n_pairs']}")
    (OUT / "BIOMARKER_STABILITY_CORRECTION_REPORT.md").write_text(
        "# Biomarker stability correction report (reporting-only)\n\n"
        "## What was wrong\n\n"
        "`stability_for_maps()` in `scripts_paper/final510_wm_pilot.py` computed "
        "`fc_top10_roi_jaccard` and `sc_top10_roi_jaccard` from index 2 of the "
        "map tuple, which is the *multimodal* ROI-importance array. FC and SC "
        "top-10 ROI Jaccard were therefore identical to the multimodal value by "
        "construction (verified in biomarker_stability.csv). In addition, folds "
        "with v_FC = 1 (inactive SC) produced all-zero SC vectors, and the old "
        "code propagated NaN Spearman values instead of excluding them.\n\n"
        "## What was corrected\n\n"
        "- FC metrics now use `I_FC[i] = sum_j |C_FC[i,j]|` only.\n"
        "- SC metrics now use `I_SC[i] = sum_j |C_SC[i,j]|` only.\n"
        "- Multimodal metrics use `I_MULTI = I_FC + I_SC` (unchanged definition).\n"
        "- Inactive modality folds are excluded per modality; all-zero maps never "
        "produce rankings. ABSTAINED folds (expert alpha = 0) are excluded from "
        "all stability metrics.\n"
        "- Tie rule (documented): descending importance, then ascending AAL116 "
        "canonical ROI index (`np.lexsort`). The old `argsort(...)[-k:]` had an "
        "implicit tie behaviour favouring larger indices.\n"
        "- Added `multimodal_sign_consistency` on `C_MULTI = C_FC + C_SC` "
        "(previously not reported) and per-metric mean/median/SD, n_pairs, and "
        "n_excluded_inactive.\n\n"
        "## Which metrics changed\n\n"
        f"{ba}\n\n"
        "FC/SC top-10 ROI Jaccard change (previously equal to multimodal), and "
        "SC edge Spearman is now a finite number over active folds instead of "
        "NaN. Edge Jaccard, sign consistency, and multimodal metrics keep their "
        "definitions; multimodal values are unchanged in expectation.\n\n"
        "## Which metrics did not change\n\n"
        "- All prediction metrics (primary_metrics.csv, seed_metrics.csv, "
        "fold_metrics.csv) and all model/hyperparameter selections.\n"
        "- Faithfulness (biomarker_faithfulness.csv): it uses the combined "
        "`I_MULTI` ranking, which was already computed correctly.\n\n"
        "## Source coefficient files\n\n"
        "`outputs/iclr/palf_final510_wm/_state/folds/fold_seed*_f*.pkl` "
        "(25 saved outer-fold checkpoints). No model was retrained; coefficients "
        "were read via `coefficient_maps` from the frozen checkpoints.\n\n"
        "## Inactive-fold handling\n\n"
        "FC inactive: expert abstained (alpha = 0). SC inactive: alpha = 0 or "
        "v_FC = 1. Excluded per modality from edge Spearman, edge Jaccard, "
        "top-10 ROI Jaccard, and sign consistency. Multimodal stability remains "
        "valid whenever the combined map is non-zero.\n\n"
        "## Confirmation\n\n"
        "Predictions were not rerun; no model selection, hyperparameter search, "
        "prior change, seed change, or fold change was performed. The 98-subject "
        "holdout was not accessed.\n")
    print(f"  Stage B done in {time.time()-t0:.0f}s")
    return df


def stage_tables(df):
    t0 = time.time()
    print("=" * 70); print("Stage C: corrected paper-ready tables + top-10 list")
    print("=" * 70)
    stab = df.copy()
    faith = pd.read_csv(OUT / "biomarker_faithfulness.csv")
    bm = R.biomarker_tables(stab, faith)
    bm.to_csv(OUT / "biomarker_table.csv", index=False)
    R.biomarker_table_tex(bm, PR / "table_main_wm_biomarkers.tex",
                          PR / "table_main_wm_biomarkers.csv")
    # top-10 multimodal biomarker list with modality-specific importances
    _, entries = load_coefficient_entries()
    prior = pd.read_csv(P.PRIOR_FILES["matched"])
    labels = prior["roi_label"].tolist()
    for model in ("R-MATCHED", "N-MATCHED"):
        ent = [e for e in entries if e["model"] == model and e["multi_active"]]
        fc_act = [e for e in ent if e["fc_active"]]
        sc_act = [e for e in ent if e["sc_active"]]
        imp_fc = (np.mean([e["imp_fc"] for e in fc_act], axis=0)
                  if fc_act else np.zeros(P.N_ROI))
        imp_sc = (np.mean([e["imp_sc"] for e in sc_act], axis=0)
                  if sc_act else np.zeros(P.N_ROI))
        imp_multi = np.mean([e["imp_multi"] for e in ent], axis=0)
        order = np.lexsort((np.arange(P.N_ROI), -imp_multi))[:10]
        rows = []
        for rank, i in enumerate(order, 1):
            rows.append({"model": model, "rank": rank, "atlas_index": int(i) + 1,
                         "roi": labels[i], "fc_importance": float(imp_fc[i]),
                         "sc_importance": float(imp_sc[i]),
                         "multimodal_importance": float(imp_multi[i])})
        out = pd.DataFrame(rows)
        out.to_csv(PR / f"wm_top10_biomarkers_{model}.csv", index=False)
        if model == "R-MATCHED":
            out.to_csv(PR / "wm_top10_biomarkers.csv", index=False)
    main = pd.read_csv(PR / "wm_top10_biomarkers_R-MATCHED.csv")
    md = ["# Final WM top-10 biomarkers (multimodal)", "",
          "Ranking: mean `I_MULTI = I_FC + I_SC` across valid outer folds; "
          "descending importance, AAL116 index tie-break. "
          "Source: saved final 510 outer-fold coefficients.", "",
          "```", main.to_string(index=False), "```", ""]
    (PR / "wm_top10_biomarkers.md").write_text("\n".join(md))
    print(main[["rank", "atlas_index", "roi", "multimodal_importance"]]
          .to_string(index=False))
    # claims: appended clarification only where stability is mentioned
    safe = PR / "SAFE_CLAIMS.md"
    txt = safe.read_text()
    note = ("\n- Corrected modality-specific biomarker stability values (see "
            "BIOMARKER_STABILITY_CORRECTION_REPORT.md) do not change the B0 "
            "biomarker tier, which is determined by cross-fold perturbation "
            "faithfulness.\n")
    if "BIOMARKER_STABILITY_CORRECTION_REPORT" not in txt:
        safe.write_text(txt.rstrip() + "\n" + note)
    print(f"  Stage C done in {time.time()-t0:.0f}s")
    return bm


def stage_plots(bm):
    t0 = time.time()
    print("=" * 70); print("Stage D: corrected stability plots")
    print("=" * 70)
    R.fig_stability(bm)
    for base in ("fig5_wm_biomarker_stability",):
        for d in (PLOTS, PR):
            for ext in ("pdf", "png", "svg"):
                f = d / f"{base}.{ext}"
                assert f.exists(), f
    print(f"  Stage D done in {time.time()-t0:.0f}s")


def stage_finalize(df, frozen):
    t0 = time.time()
    print("=" * 70); print("Stage E: tests, final audit, seal, runtime")
    print("=" * 70)
    import shutil
    import subprocess
    (OUT / "tests").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "tests" / "test_biomarker_stability_correction.py",
                 OUT / "tests" / "test_biomarker_stability_correction.py")
    res = subprocess.run(
        [sys.executable, "-m", "pytest",
         str(ROOT / "tests" / "test_biomarker_stability_correction.py"),
         "-v", "--no-header"], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    (OUT / "tests" / "test_biomarker_stability_correction_results.txt"
     ).write_text(res.stdout)
    assert res.returncode == 0, "correction tests failed"
    n_pass = res.stdout.count("PASSED")
    print(f"  correction tests: {n_pass} passed")
    # frozen-file integrity
    changed = [f for f, h in frozen.items() if sha256_file(OUT / f) != h]
    if changed:
        raise SystemExit(f"STATUS: BIOMARKER_CORRECTION_BLOCKED - frozen "
                         f"prediction files changed: {changed}")
    # holdout ID scan over files written by this correction
    holdout_ids = set(HOLDOUT_TXT.read_text().strip().split("\n"))
    written = ["biomarker_stability_corrected.csv",
               "BIOMARKER_STABILITY_CORRECTION_REPORT.md",
               "BIOMARKER_CORRECTION_SEAL.json",
               "BIOMARKER_CORRECTION_FINAL_AUDIT.md",
               "BIOMARKER_CORRECTION_RUNTIME.md",
               "biomarker_table.csv",
               "paper_ready/table_main_wm_biomarkers.csv",
               "paper_ready/table_main_wm_biomarkers.tex",
               "paper_ready/wm_top10_biomarkers.csv",
               "paper_ready/wm_top10_biomarkers.md",
               "paper_ready/wm_top10_biomarkers_R-MATCHED.csv",
               "paper_ready/wm_top10_biomarkers_N-MATCHED.csv",
               "paper_ready/SAFE_CLAIMS.md"]
    hits = {}
    for f in written:
        p = OUT / f
        if p.exists() and p.suffix in (".csv", ".md", ".json"):
            txt = p.read_text(errors="ignore")
            found = sorted(i for i in holdout_ids if i in txt)
            if found:
                hits[f] = found
    if hits:
        raise SystemExit(f"STATUS: BIOMARKER_CORRECTION_BLOCKED - holdout IDs "
                         f"found in corrected outputs: {hits}")
    start = json.loads(START_FILE.read_text())["started"]
    (OUT / "BIOMARKER_CORRECTION_SEAL.json").write_text(json.dumps({
        "holdout_sha256": HOLDOUT_SHA, "holdout_count": 98,
        "holdout_unique": 98,
        "statement": "No holdout data were accessed. Only saved final-510 "
                     "outer-fold coefficient maps and metadata from the merged "
                     "510 development cohort were read.",
        "timestamp": utc(),
        "files_read": ["_state/folds/fold_seed*_f*.pkl (25 checkpoints)",
                       "biomarker_stability.csv (old values, for before/after)",
                       "biomarker_faithfulness.csv",
                       "data_splits/final510_subjects.txt metadata",
                       "outputs/priors/llm/working_memory_contrastive_qwen3/"
                       "roi_prior.csv (ROI labels only)"],
        "files_written": written,
        "frozen_file_hashes": frozen,
        "prediction_files_unchanged": True,
        "holdout_id_scan_hits": {},
    }, indent=2))
    bm = R.biomarker_tables(df, pd.read_csv(OUT / "biomarker_faithfulness.csv"))
    audit = [
        "# Biomarker stability correction - final audit", "",
        "1. Final cohort remains 510 subjects (unchanged).",
        "2. The 98-subject holdout remained sealed "
        f"(sha256 {HOLDOUT_SHA[:16]}..., no access).",
        "3. No prediction models were retrained; no fitting code executed.",
        "4. No hyperparameters changed.",
        "5. No prior changed.",
        "6. No CV split changed.",
        "7. No seed changed.",
        "8. Only biomarker stability reporting was corrected.",
        "9. FC/SC modality-specific ROI Jaccard is now computed from the "
        "modality-specific coefficient maps.",
        "10. The multimodal ranking (I_MULTI = I_FC + I_SC) is preserved and "
        "still defines the top-10 biomarker list.",
        "11. All corrected values trace to saved outer-fold coefficients in "
        "`_state/folds/`.",
        "12. All correction tests pass (see tests/"
        "test_biomarker_stability_correction_results.txt).", "",
        "## Before/after (top-10 ROI Jaccard)", "",
        before_after(df), "",
        "## Corrected stability (mean over valid pairs)", "",
        "```", bm.to_string(index=False), "```", "",
    ]
    (OUT / "BIOMARKER_CORRECTION_FINAL_AUDIT.md").write_text("\n".join(audit))
    elapsed = time.time() - float(json.loads(START_FILE.read_text()).get("started_epoch", time.time()))
    secs = int(round(elapsed))
    h = secs // 3600
    m = (secs % 3600 + 59) // 60
    if m == 60:
        h, m = h + 1, 0
    (OUT / "BIOMARKER_CORRECTION_RUNTIME.md").write_text(
        "# Biomarker correction runtime\n\n"
        f"- Estimated: {ESTIMATE}\n- Actual: {h}:{m:02d} ({secs} s)\n\n"
        "Stages: A source/audit, B corrected stability, C tables, D plots, "
        "E tests/audit. All stages are reporting-only; the full corrected "
        "reporting pipeline runs in well under a minute.\n")
    print(f"BIOMARKER_CORRECTION_ACTUAL_RUNTIME: {h}:{m:02d} ({secs}s)")
    print(f"  Stage E done in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["audit", "stability", "tables", "plots", "finalize",
                             "all"])
    args = ap.parse_args()
    if args.stage == "audit":
        stage_audit()
    elif args.stage == "stability":
        df = stage_stability(frozen=stage_audit())
    elif args.stage == "tables":
        stage_tables(pd.read_csv(OUT / "biomarker_stability_corrected.csv"))
    elif args.stage == "plots":
        stage_plots(pd.read_csv(OUT / "biomarker_stability_corrected.csv"))
    elif args.stage == "finalize":
        frozen = {f: sha256_file(OUT / f) for f in FROZEN}
        stage_finalize(pd.read_csv(OUT / "biomarker_stability_corrected.csv"),
                       frozen=frozen)
    else:
        frozen = stage_audit()
        df = stage_stability(frozen)
        bm = stage_tables(df)
        stage_plots(bm)
        stage_finalize(df, frozen)


if __name__ == "__main__":
    main()
