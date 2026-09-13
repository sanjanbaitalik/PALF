#!/usr/bin/env python3
"""FINAL 510 WM evidence package: aggregation, tiers, tables, figures, paper.

Primary estimand (Section 18): mean of seed-wise outer-CV Pearson values.
Sensitivity (clearly separated): Pearson after averaging each subject's
repeated-CV predictions, with paired subject bootstrap (seed 9701, 10k).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts_paper"))

from final510_wm_pilot import (  # noqa: E402
    ALL_MODELS, ARCH_OF, FINAL510_SEEDS, FOLDS, N510, OUTER_FOLDS, PRIOR_OF,
    PRIOR_TYPES, STATE, fold_path, load_combined, load_priors, metrics)

OUT = ROOT / "outputs" / "iclr" / "palf_final510_wm"
PR = OUT / "paper_ready"
SUP = OUT / "supplementary"
PLOTS = OUT / "plots"
BOOT_SEED = 9701
N_BOOT = 10000
MODEL_ORDER = ["R0", "R-MATCHED", "R-CROSS", "R-SHUFFLED", "R-RANDOM",
               "N-MATCHED", "N-CROSS", "N-SHUFFLED", "N-RANDOM"]
CONTROLS = {"R": ["R-CROSS", "R-SHUFFLED", "R-RANDOM"],
            "N": ["N-CROSS", "N-SHUFFLED", "N-RANDOM"]}
MATCHED = {"R": "R-MATCHED", "N": "N-MATCHED"}


def load_folds():
    recs = []
    for seed in FINAL510_SEEDS:
        for fold in range(OUTER_FOLDS):
            recs.append(pickle.load(open(fold_path(seed, fold), "rb")))
    return recs


def model_pred(rec, model):
    if model == "R0":
        return rec["r0"]["pred"]
    return rec["variants"][model]["final_test"]


def seedwise_table(recs, y):
    rows = []
    for model in MODEL_ORDER:
        for seed in FINAL510_SEEDS:
            sub = sorted([r for r in recs if r["seed"] == seed],
                         key=lambda r: r["fold"])
            idx = np.concatenate([r["te"] for r in sub])
            pred = np.concatenate([model_pred(r, model) for r in sub])
            rows.append({"model": model, "seed": seed, "n": len(idx),
                         **metrics(y[idx], pred)})
    return pd.DataFrame(rows)


def foldwise_table(recs, y):
    rows = []
    for r in recs:
        for model in MODEL_ORDER:
            rows.append({"model": model, "seed": r["seed"], "fold": r["fold"],
                         "n": len(r["te"]),
                         **metrics(y[r["te"]], model_pred(r, model))})
    return pd.DataFrame(rows)


def primary_table(seed_df):
    rows = []
    a0 = seed_df[seed_df.model == "R0"].set_index("seed")["pearson"]
    for model in MODEL_ORDER:
        sub = seed_df[seed_df.model == model].set_index("seed")
        d = sub["pearson"] - a0
        rows.append({
            "model": model, "arch": ("R0" if model == "R0" else ARCH_OF[model]),
            "prior": ("none" if model == "R0" else PRIOR_OF[model]),
            "pearson_mean": sub["pearson"].mean(),
            "pearson_std": sub["pearson"].std(),
            "pearson_median": sub["pearson"].median(),
            "rmse_mean": sub["rmse"].mean(), "mae_mean": sub["mae"].mean(),
            "delta_r_mean": d.mean(), "delta_r_std": d.std(),
            "positive_seeds": int((d > 0).sum()), "n_seeds": len(d),
            "delta_seed_values": list(np.round(d.values, 6)),
        })
    return pd.DataFrame(rows)


def subject_ensemble(recs, model):
    P = np.zeros(N510); n = np.zeros(N510)
    for r in recs:
        pred = model_pred(r, model)
        P[r["te"]] += pred
        n[r["te"]] += 1
    assert np.all(n == len(FINAL510_SEEDS))
    return P / n


def sensitivity(recs, y):
    ens = {m: subject_ensemble(recs, m) for m in MODEL_ORDER}
    a0r = pearsonr(y, ens["R0"]).statistic
    rng = np.random.RandomState(BOOT_SEED)
    idx = rng.randint(0, N510, size=(N_BOOT, N510))
    rows = []
    for model in MODEL_ORDER:
        pa, pb = ens["R0"], ens[model]
        obs = pearsonr(y, pb).statistic - a0r
        d = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            i = idx[b]
            d[b] = (pearsonr(y[i], pb[i]).statistic
                    - pearsonr(y[i], pa[i]).statistic)
        rows.append({"model": model,
                     "ensemble_r": pearsonr(y, pb).statistic,
                     "ensemble_delta_r_vs_R0": obs,
                     "boot_ci_lo": np.percentile(d, 2.5),
                     "boot_ci_hi": np.percentile(d, 97.5),
                     "fraction_le_0": float(np.mean(d <= 0)),
                     "estimand": "sensitivity_ensemble"})
    return pd.DataFrame(rows), ens


def comparisons(seed_df, prim):
    prim_idx = prim.set_index("model")
    rows = []
    defs = [("C1_R-MATCHED - R0", "R-MATCHED", "R0"),
            ("C2_R-MATCHED - R-CROSS", "R-MATCHED", "R-CROSS"),
            ("C2_R-MATCHED - R-SHUFFLED", "R-MATCHED", "R-SHUFFLED"),
            ("C2_R-MATCHED - R-RANDOM", "R-MATCHED", "R-RANDOM"),
            ("C3_N-MATCHED - R-MATCHED", "N-MATCHED", "R-MATCHED"),
            ("C4_N-MATCHED - N-CROSS", "N-MATCHED", "N-CROSS"),
            ("C4_N-MATCHED - N-SHUFFLED", "N-MATCHED", "N-SHUFFLED"),
            ("C4_N-MATCHED - N-RANDOM", "N-MATCHED", "N-RANDOM")]
    for name, a, b in defs:
        da = seed_df[seed_df.model == a].set_index("seed")["pearson"]
        db = seed_df[seed_df.model == b].set_index("seed")["pearson"]
        d = da - db
        rows.append({"comparison": name,
                     "mean_delta_r": prim_idx.loc[a, "pearson_mean"]
                     - prim_idx.loc[b, "pearson_mean"],
                     "seed_mean_delta": d.mean(), "positive_seeds": int((d > 0).sum()),
                     "n_seeds": len(d), "seed_deltas": list(np.round(d.values, 6))})
    return pd.DataFrame(rows)


def prediction_tier(prim, comp):
    pm = prim.set_index("model")
    r0 = pm.loc["R0", "pearson_mean"]
    cr = pm.loc["R-MATCHED", "pearson_mean"]
    dm = cr - r0
    pos = int(pm.loc["R-MATCHED", "positive_seeds"])
    c2 = comp.set_index("comparison")
    beats_controls_mean = all(
        pm.loc["R-MATCHED", "pearson_mean"] > pm.loc[c, "pearson_mean"]
        for c in CONTROLS["R"])
    c2_pos = {c: int(c2.loc[f"C2_R-MATCHED - {c}", "positive_seeds"])
              for c in CONTROLS["R"]}
    beats_sr = (c2_pos["R-SHUFFLED"] >= 4 and c2_pos["R-RANDOM"] >= 4)
    if dm <= 0:
        tier = "P0"
    elif dm >= 0.005 and pos >= 4:
        tier = "P2"
        if beats_controls_mean and beats_sr:
            tier = "P3"
    else:
        tier = "P1"
    return {"RIDGE_PREDICTION_LEVEL": tier, "r_matched": cr, "r0": r0,
            "delta_r_mean": dm, "positive_seeds": pos,
            "beats_controls_mean": bool(beats_controls_mean),
            "control_positive_seeds": c2_pos,
            "beats_shuffled_random_4of5": bool(beats_sr)}


def _mean_of(df, model, cols):
    sub = df[df.model == model]
    return {c: float(sub[c].mean()) for c in cols if c in sub}


def biomarker_tables(stab, faith):
    rows = []
    for model in MODEL_ORDER[1:]:
        s = stab[stab.model == model]
        f = faith[(faith.model == model) & (faith.abstained == 0)]
        rows.append({
            "model": model, "arch": ARCH_OF[model], "prior": PRIOR_OF[model],
            "n_valid": int(s["n_valid"].iloc[0]) if len(s) else 0,
            "n_abstained": int(s["n_abstained"].iloc[0]) if len(s) else 0,
            "fc_edge_spearman": float(s["fc_edge_spearman"].mean()) if len(s) else np.nan,
            "sc_edge_spearman": float(s["sc_edge_spearman"].mean()) if len(s) else np.nan,
            "edge_rank_stability": float(np.nanmean(
                [s["fc_edge_spearman"].mean(), s["sc_edge_spearman"].mean()])) if len(s) else np.nan,
            "multimodal_top10_roi_jaccard": float(s["multimodal_top10_roi_jaccard"].mean()) if len(s) else np.nan,
            "fc_top10_roi_jaccard": float(s["fc_top10_roi_jaccard"].mean()) if len(s) else np.nan,
            "sc_top10_roi_jaccard": float(s["sc_top10_roi_jaccard"].mean()) if len(s) else np.nan,
            "top10_roi_jaccard": float(np.nanmean(
                [s["fc_top10_roi_jaccard"].mean(), s["sc_top10_roi_jaccard"].mean(),
                 s["multimodal_top10_roi_jaccard"].mean()])) if len(s) else np.nan,
            "sign_consistency": float(np.nanmean(
                [s["sign_consistency_fc"].mean(), s["sign_consistency_sc"].mean()])) if len(s) else np.nan,
            "top10_delta_rmse": float(f["top10_delta_rmse"].mean()) if len(f) else np.nan,
            "random10_mean": float(f["random10_mean"].mean()) if len(f) else np.nan,
            "top10_minus_random10": float(f["top10_minus_random10"].mean()) if len(f) else np.nan,
            "percentile": float(f["top10_random10_percentile"].mean()) if len(f) else np.nan,
            "bottom10_delta_rmse": float(f["bottom10_delta_rmse"].mean()) if len(f) else np.nan,
            "top5_minus_random5": float(f["top5_minus_random5"].mean()) if len(f) else np.nan,
        })
    return pd.DataFrame(rows)


def seed_contrast(faith, model):
    g = faith[(faith.model == model) & (faith.abstained == 0)]
    if not len(g):
        return pd.Series(dtype=float)
    return g.groupby("seed")["top10_minus_random10"].mean()


def biomarker_tier(bm):
    b = bm.set_index("model")
    matched = MATCHED
    res = {}
    for arch in ("R", "N"):
        m = b.loc[matched[arch]]
        contrast = m["top10_minus_random10"]
        seeds_pos = None
        res[arch] = {"matched": matched[arch], "contrast": contrast}
    # use Ridge architecture for the headline level, report both
    return res


def biomarker_levels(bm, faith):
    out = {}
    for arch in ("R", "N"):
        m = bm[bm.model == MATCHED[arch]].iloc[0]
        contrast = m["top10_minus_random10"]
        sc = seed_contrast(faith, MATCHED[arch])
        pos_seeds = int((sc > 0).sum()) if len(sc) else 0
        if contrast <= 0:
            level = "B0"
        elif pos_seeds >= 4:
            level = "B2"
            level = "B2"  # B3 evaluated below
        else:
            level = "B1"
        crit = {}
        if level == "B2":
            matched_row = bm[bm.model == MATCHED[arch]].iloc[0]
            n_better = 0
            detail = {}
            for crit_name, col, controls in (
                    ("edge_rank_stability", "edge_rank_stability",
                     [f"{arch}-SHUFFLED", f"{arch}-RANDOM"]),
                    ("top10_roi_jaccard", "top10_roi_jaccard",
                     [f"{arch}-SHUFFLED", f"{arch}-RANDOM"]),
                    ("top10_random_faithfulness", "top10_minus_random10",
                     [f"{arch}-SHUFFLED", f"{arch}-RANDOM"]),
                    ("sign_consistency", "sign_consistency",
                     [f"{arch}-SHUFFLED", f"{arch}-RANDOM"])):
                ctrl_vals = [bm[bm.model == c].iloc[0][col] for c in controls]
                better = matched_row[col] > max(ctrl_vals)
                detail[crit_name] = {"matched": float(matched_row[col]),
                                     "controls": [float(x) for x in ctrl_vals],
                                     "better_than_both": bool(better)}
                n_better += int(better)
            crit = {"criteria_better_than_both_shuffled_and_random": detail,
                    "n_criteria": n_better}
            if n_better >= 2:
                level = "B3"
        out[arch] = {"level": level, "top10_minus_random10": float(contrast),
                     "positive_seed_contrasts": pos_seeds, "criteria": crit}
    return out


def selected_configs(recs):
    rows = []
    for r in recs:
        for model in MODEL_ORDER[1:]:
            v = r["variants"][model]
            rows.append({
                "seed": r["seed"], "fold": r["fold"], "model": model,
                "arch": v["arch"], "prior": PRIOR_OF[model],
                "fc_family": v["fc_family"], "fc_size": v["fc_size"],
                "sc_family": v["sc_family"], "sc_size": v["sc_size"],
                "n_fc_edges": int(v["fc_mask"].sum()),
                "n_sc_edges": int(v["sc_mask"].sum()),
                "lambda_r_fc": v["lambda_r_fc"], "lambda_r_sc": v["lambda_r_sc"],
                "ratio_fc": v["ratio_fc"], "ratio_sc": v["ratio_sc"],
                "v_fc": v["v_fc"], "alpha": v["alpha"],
                "abstained": int(v["alpha"] == 0.0),
                "runtime_s": r["runtime"].get(model, np.nan),
            })
    return pd.DataFrame(rows)


def control_tables(recs, y):
    """Per-fold control metrics and matched-minus-control deltas."""
    rows = []
    for r in recs:
        for arch, tag in (("ridge", "R"), ("ncr", "N")):
            m = MATCHED[tag]
            for c in CONTROLS[tag]:
                pm = metrics(y[r["te"]], model_pred(r, m))["pearson"]
                pc = metrics(y[r["te"]], model_pred(r, c))["pearson"]
                rows.append({"arch": arch, "seed": r["seed"], "fold": r["fold"],
                             "matched": m, "control": c,
                             "pearson_matched": pm, "pearson_control": pc,
                             "delta_r": pm - pc})
    return pd.DataFrame(rows)


def result_reuse_audit():
    v34 = ROOT / "outputs" / "iclr" / "palf_412_vs_510_scaling" / "full510_secondary_metrics.csv"
    audit = {"reused_results": [], "cross_checks": [],
             "decision": "NO_RESULTS_REUSED",
             "reason": ("The v34 510-secondary run covers only seed 7171 and models "
                        "A0/A1/B1 without coefficient artifacts, while the final510 "
                        "study requires 5 seeds, 9 architecture-matched models, and "
                        "per-fold coefficient maps for biomarkers."),
             "v34_file": str(v34.relative_to(ROOT)) if v34.exists() else None}
    if v34.exists():
        d = pd.read_csv(v34)
        d = d[d.target == "WM"]
        audit["v34_models"] = sorted(d["model"].unique().tolist())
        audit["v34_seeds"] = sorted(d["seed"].unique().tolist())
    return audit


def write_core_tables(recs, y, seed_df, fold_df, prim, sens, comp, sel,
                      ctrl):
    seed_df.to_csv(OUT / "seed_metrics.csv", index=False)
    fold_df.to_csv(OUT / "fold_metrics.csv", index=False)
    prim.drop(columns=["delta_seed_values"]).to_csv(OUT / "primary_metrics.csv",
                                                    index=False)
    sens.to_csv(OUT / "sensitivity_ensemble_bootstrap.csv", index=False)
    comp.to_csv(OUT / "primary_comparisons.csv", index=False)
    sel.to_csv(OUT / "SELECTED_CONFIGS.csv", index=False)
    ctrl[ctrl.arch == "ridge"].to_csv(OUT / "ridge_prior_controls.csv", index=False)
    ctrl[ctrl.arch == "ncr"].to_csv(OUT / "ncr_prior_controls.csv", index=False)


def _delta_row(prim, comp, m):
    """Delta r vs R0 (mean, positive seeds) from primary/seed tables."""
    r = prim[prim.model == m].iloc[0]
    if m == "R0":
        return "0.000", 0
    return f"{r['delta_r_mean']:+.3f}", int(r["positive_seeds"])


def main_table_tex(prim, comp, path_tex, path_csv, highlight=True):
    rows = []
    for m in MODEL_ORDER:
        r = prim[prim.model == m].iloc[0]
        dr, pos = _delta_row(prim, comp, m)
        rows.append({"method": m, "prior": r["prior"], "pearson": r["pearson_mean"],
                     "pearson_std": r["pearson_std"], "rmse": r["rmse_mean"],
                     "mae": r["mae_mean"], "delta_r_vs_R0": dr,
                     "positive_seeds": f"{pos}/5" if m != "R0" else "--"})
    df = pd.DataFrame(rows)
    df.to_csv(path_csv, index=False)
    lines = ["% Auto-generated final 510 WM prediction table (nested CV).",
             "\\begin{table}[t]", "\\centering",
             "\\caption{Working-memory prediction on the final 510-subject cohort "
             "(repeated nested cross-validation; mean over 5 seeds).}",
             "\\label{tab:final510_wm_prediction}",
             "\\begin{tabular}{llrrrrr}", "\\toprule",
             "Method & Prior & Pearson $r$ & RMSE & MAE & $\\Delta r$ vs R0 & Positive seeds \\\\",
             "\\midrule"]
    for _, r in df.iterrows():
        name = r["method"]
        if highlight and name == "R-MATCHED":
            name = "\\textbf{" + name + "}"
        lines.append(f"{name} & {r['prior']} & {r['pearson']:.3f} & {r['rmse']:.2f} "
                     f"& {r['mae']:.2f} & {r['delta_r_vs_R0']} & {r['positive_seeds']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path(path_tex).write_text("\n".join(lines))
    return df


def biomarker_table_tex(bm, path_tex, path_csv):
    bm.to_csv(path_csv, index=False)
    lines = ["% Auto-generated final 510 WM biomarker table (nested CV).",
             "\\begin{table}[t]", "\\centering",
             "\\caption{Semantic-expert biomarker stability and cross-fold "
             "perturbation faithfulness on the final 510-subject cohort.}",
             "\\label{tab:final510_wm_biomarkers}",
             "\\begin{tabular}{llrrrrrrr}", "\\toprule",
             "Arch & Prior & FC Spear. & SC Spear. & MM top10 Jacc. & Sign cons. "
             "& top10 $\\Delta$RMSE & rand10 & top10$-$rand10 \\\\",
             "\\midrule"]
    for _, r in bm.iterrows():
        lines.append(f"{r['arch']} & {r['prior']} & {r['fc_edge_spearman']:.3f} & "
                     f"{r['sc_edge_spearman']:.3f} & "
                     f"{r['multimodal_top10_roi_jaccard']:.3f} & "
                     f"{r['sign_consistency']:.3f} & {r['top10_delta_rmse']:.3f} & "
                     f"{r['random10_mean']:.3f} & {r['top10_minus_random10']:+.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    Path(path_tex).write_text("\n".join(lines))
    return bm


# ══════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════

def _save(fig, path_base):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{path_base}.{ext}", bbox_inches="tight")


def _box(ax, x, y, w, h, text, fc="white", ec="black", fs=8, lw=1.2):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1, y1, x2, y2, lw=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color="black"))


def fig_method_overview(priors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 11)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 13)
    _box(ax, 3.3, 12.0, 3.4, 0.7, "Task text:\n\"Working Memory\"", fs=9)
    _arrow(ax, 5.0, 12.0, 5.0, 11.5)
    _box(ax, 3.2, 10.7, 3.6, 0.8,
         "LLM prior generator\n(qwen3.8:27b, temperature 0.2, seed 42)",
         fc="#ffe8b0", ec="#b8860b", fs=9, lw=2.0)
    _arrow(ax, 5.0, 10.7, 5.0, 10.2)
    _box(ax, 2.8, 9.4, 4.4, 0.8,
         "AAL116 semantic ROI prior\n$p_i$, frozen + SHA256", fs=9)
    _arrow(ax, 4.2, 9.4, 3.2, 8.9); _arrow(ax, 5.8, 9.4, 6.8, 8.9)
    _box(ax, 0.6, 7.9, 4.2, 1.0,
         "FC prior-selected subspace\n$\\{$top-$K$ edges, ROI-incident$\\}$", fs=8)
    _box(ax, 5.2, 7.9, 4.2, 1.0,
         "SC prior-selected subspace\n$\\{$top-$K$ edges, ROI-incident$\\}$", fs=8)
    _arrow(ax, 2.7, 7.9, 2.7, 7.4); _arrow(ax, 7.3, 7.9, 7.3, 7.4)
    _box(ax, 0.9, 6.4, 3.6, 1.0, "Ridge / NCR expert\non FC subspace", fs=8)
    _box(ax, 5.5, 6.4, 3.6, 1.0, "Ridge / NCR expert\non SC subspace", fs=8)
    _arrow(ax, 2.7, 6.4, 4.2, 5.9); _arrow(ax, 7.3, 6.4, 5.8, 5.9)
    _box(ax, 3.4, 5.2, 3.2, 0.7, "modality fusion $v$", fs=8)
    _arrow(ax, 5.0, 5.2, 5.0, 4.7)
    _box(ax, 3.2, 4.0, 3.6, 0.7, "semantic expert $\\hat y_E$", fs=9)
    _box(ax, 0.6, 2.6, 3.0, 0.9, "Corrected R0 backbone\n(no prior)", fs=8)
    _arrow(ax, 2.1, 3.5, 4.4, 4.35); _arrow(ax, 3.9, 4.0, 4.4, 4.35)
    _box(ax, 4.6, 2.6, 3.2, 0.9, "hierarchical fusion $\\alpha$", fs=9)
    _arrow(ax, 6.2, 2.6, 6.2, 2.1)
    _arrow(ax, 6.2, 2.1, 3.4, 1.5); _arrow(ax, 6.2, 2.1, 8.0, 1.5)
    _box(ax, 1.8, 0.6, 3.2, 0.9, "WM prediction", fs=9)
    _box(ax, 6.4, 0.6, 3.2, 0.9, "biomarker map\n$I_i=\\sum_j |c^{FC}_{ij}|+|c^{SC}_{ij}|$", fs=8)
    _arrow(ax, 8.0, 0.6, 8.0, 0.05, lw=0)
    ax.text(8.0, -0.25,
            "cross-fold stability + held-out-fold\nperturbation faithfulness",
            ha="center", va="top", fontsize=7.5)
    ax.text(5.0, -0.75, "No external holdout is used or depicted. "
                        "All estimates are nested CV within the 510-subject cohort.",
            ha="center", va="top", fontsize=7, style="italic")
    _save(fig, str(PR / "fig_method_overview"))
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def fig_priors_s1(priors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = priors["matched"]
    df = pd.DataFrame({"label": p["labels"], "prior": p["array"]})
    hi = df.nlargest(6, "prior"); lo = df.nsmallest(6, "prior").iloc[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4),
                             gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
    axes[0].axis("off")
    steps = ["Task description", "prompt template\n(contrastive clause)",
             "LLM\n(qwen3.8:27b)", "ROI raw scores",
             "normalized AAL116 prior\n$p_i\\in[0,1]$", "frozen checksum"]
    for i, t in enumerate(steps):
        _box(axes[0], 0.05, 0.86 - i * 0.155, 0.9, 0.12, t, fs=7)
        if i:
            _arrow(axes[0], 0.5, 0.99 - i * 0.155, 0.5, 0.86 - i * 0.155 + 0.12)
    axes[1].barh(hi["label"][::-1], hi["prior"][::-1], color="#c44e52")
    axes[1].set_title("Highest-prior WM regions", fontsize=9)
    axes[1].tick_params(labelsize=7)
    axes[2].barh(lo["label"][::-1], lo["prior"][::-1], color="#4c72b0")
    axes[2].set_title("Lowest-prior WM regions", fontsize=9)
    axes[2].tick_params(labelsize=7)
    fig.suptitle(f"Frozen LLM prior: {p['path']}  sha256={p['sha256'][:16]}...",
                 fontsize=8)
    fig.tight_layout()
    _save(fig, str(SUP / "fig_S1_llm_prior"))
    plt.close(fig)


def fig_masks_s2(priors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = priors["matched"]["array"]
    iu = np.triu_indices(116, 1)
    ep = p[iu[0]] * p[iu[1]]
    top = np.argsort(ep)[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].axis("off")
    _box(axes[0], 0.02, 0.65, 0.28, 0.2, "ROI prior\n$p_i$", fs=8)
    _arrow(axes[0], 0.30, 0.75, 0.40, 0.75)
    _box(axes[0], 0.42, 0.65, 0.30, 0.2, "pairwise score\n$q_{ij}=p_i p_j$", fs=8)
    _arrow(axes[0], 0.62, 0.65, 0.62, 0.52)
    _box(axes[0], 0.40, 0.28, 0.34, 0.2,
         f"top-$K$ edge mask\n$K\\in\\{{100,300,600,1200\\}}$", fs=8)
    axes[0].text(0.5, 0.05, f"example top edge: ROI {iu[0][top]+1} - ROI "
                            f"{iu[1][top]+1}\n$q_{{ij}}={ep[top]:.4f}$",
                 ha="center", fontsize=7)
    axes[1].axis("off")
    _box(axes[1], 0.02, 0.65, 0.28, 0.2, "ROI prior\n$p_i$", fs=8)
    _arrow(axes[1], 0.30, 0.75, 0.40, 0.75)
    _box(axes[1], 0.42, 0.65, 0.30, 0.2, "top-$M$ ROIs\n$M\\in\\{5,10,15\\}$", fs=8)
    _arrow(axes[1], 0.62, 0.65, 0.62, 0.52)
    _box(axes[1], 0.36, 0.28, 0.38, 0.2, "incident-edge mask\n(all edges touching the ROIs)", fs=8)
    axes[1].text(0.5, 0.05, "Masks are selected independently for FC and SC\n"
                            "by inner-CV Ridge on training folds only.", ha="center", fontsize=7)
    _save(fig, str(SUP / "fig_S2_mask_construction"))
    plt.close(fig)


def fig_ridge_ncr_s3():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    _box(ax, 0.3, 2.4, 4.2, 1.2,
         "Ridge expert\n$\\lambda_R\\|\\beta\\|_2^2$\n"
         "$\\lambda_R\\in\\{10^{-3},...,10^{3}\\}$", fs=9)
    _box(ax, 5.5, 2.4, 4.2, 1.2,
         "NCR expert\n$\\lambda_R\\|\\beta\\|_2^2+\\lambda_L\\,\\beta^\\top L_E\\beta$\n"
         "$\\lambda_L=ratio\\cdot\\lambda_R$", fs=9)
    _box(ax, 2.4, 0.3, 5.2, 1.2,
         "selected-edge subgraph $E$ (prior mask)\n"
         "$\\beta^\\top L_E\\beta=\\sum_{(e,e')\\in E}(\\beta_e-\\beta_{e'})^2$"
         "  over line-graph neighbours", fs=8)
    _arrow(ax, 5.0, 2.4, 5.0, 1.5)
    ax.text(0.05, 3.85, "Both experts share the same prior-selected subspace; "
                        "only the structured penalty differs (ratio=0 $\\equiv$ Ridge).",
            fontsize=8)
    _save(fig, str(SUP / "fig_S3_ridge_vs_ncr"))
    plt.close(fig)


def fig_fusion_s4():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.0)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    _box(ax, 0.4, 1.7, 2.2, 0.8, "FC expert\n$\\hat y_{FC}$", fs=9)
    _box(ax, 0.4, 0.4, 2.2, 0.8, "SC expert\n$\\hat y_{SC}$", fs=9)
    _box(ax, 3.6, 1.05, 2.2, 0.9, "$\\hat y_E=v\\hat y_{FC}$\n$+(1-v)\\hat y_{SC}$", fs=9)
    _arrow(ax, 2.6, 2.1, 3.6, 1.6); _arrow(ax, 2.6, 0.8, 3.6, 1.35)
    _box(ax, 6.6, 1.85, 2.4, 0.8, "Corrected R0\n$\\hat y_{R0}$", fs=9)
    _box(ax, 6.6, 0.4, 2.4, 0.8, "$\\hat y=(1-\\alpha)\\hat y_{R0}$\n$+\\alpha\\hat y_E$", fs=9)
    _arrow(ax, 5.8, 1.5, 6.6, 1.25); _arrow(ax, 7.8, 1.85, 7.8, 1.2)
    ax.text(0.4, 2.75, "v and alpha are selected on training OOF only; "
                       "grid = {0, 0.05, ..., 1}", fontsize=8)
    _save(fig, str(SUP / "fig_S4_hierarchical_fusion"))
    plt.close(fig)


def fig_controls_s5():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 2.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    _box(ax, 0.2, 1.0, 2.4, 1.0, "SPSEF architecture\n(identical)", fs=8)
    for i, (name, col) in enumerate((("matched", "#55a868"), ("cross-task", "#c44e52"),
                                     ("shuffled", "#8172b2"), ("random", "#937860"))):
        _box(ax, 3.4 + i * 1.6, 1.2, 1.4, 0.8, name, fc=col, fs=8)
        _arrow(ax, 2.6, 1.5, 3.4 + i * 1.6, 1.6)
    ax.text(5.0, 0.45, "Only the prior array changes: masks families/sizes, lambda grid, "
                       "NCR grid, preprocessing, CV folds, tie-breaks are identical.",
            ha="center", fontsize=7.5)
    _save(fig, str(SUP / "fig_S5_semantic_prior_controls"))
    plt.close(fig)


def fig_biomarker_s6():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 2.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    steps = ["expert primal\n$\\beta_m$", "modality-weighted\n$c_m=\\alpha w_m\\beta_m$",
             "incident absolute\nweights", "ROI importance\n$I_i$", "top WM\nbiomarkers"]
    for i, t in enumerate(steps):
        _box(ax, 0.2 + i * 2.0, 1.0, 1.7, 1.0, t, fs=7.5)
        if i:
            _arrow(ax, 0.2 + i * 2.0 - 0.3, 1.5, 0.2 + i * 2.0, 1.5)
    ax.text(5.0, 0.45, "FC weight = v, SC weight = 1-v. R0 coefficients are excluded "
                       "from the proposed biomarker.", ha="center", fontsize=7.5)
    _save(fig, str(SUP / "fig_S6_biomarker_extraction"))
    plt.close(fig)


def fig_faithfulness_s7():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 2.6)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    steps = ["train-fold\nbiomarker ranking", "freeze ranking",
             "outer-test subject", "mask top/random/bottom\nROIs (edges $\\to$ train means)",
             "no retraining", "$\\Delta$RMSE"]
    for i, t in enumerate(steps):
        _box(ax, 0.1 + i * 1.63, 1.0, 1.45, 1.0, t, fs=6.8)
        if i:
            _arrow(ax, 0.1 + i * 1.63 - 0.18, 1.5, 0.1 + i * 1.63, 1.5)
    ax.text(5.0, 0.35, "$\\Delta$RMSE = RMSE$_{masked}$ - RMSE$_{unmasked}$;  "
                       "positive = masking hurts prediction = faithful",
            ha="center", fontsize=7.5)
    _save(fig, str(SUP / "fig_S7_faithfulness"))
    plt.close(fig)


def fig_prediction(seed_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = [("Ridge", ["R0", "R-MATCHED", "R-CROSS", "R-SHUFFLED", "R-RANDOM"]),
              ("NCR", ["R0", "N-MATCHED", "N-CROSS", "N-SHUFFLED", "N-RANDOM"])]
    for ax, (name, models) in zip(axes, panels):
        for m in models:
            sub = seed_df[seed_df.model == m].sort_values("seed")
            ax.plot(sub["seed"], sub["pearson"], marker="o", label=m)
        ax.set_title(f"{name} architecture")
        ax.set_xlabel("outer-CV seed"); ax.legend(fontsize=7)
        ax.set_xticks(FINAL510_SEEDS)
    axes[0].set_ylabel("outer-CV Pearson r")
    fig.suptitle("Repeated nested-CV WM prediction on 510 subjects "
                 "(seed-level results)", fontsize=9)
    fig.tight_layout()
    _save(fig, str(PLOTS / "fig2_wm_prediction_seedlevel"))
    _save(fig, str(PR / "fig2_wm_prediction_seedlevel"))
    plt.close(fig)


def fig_specificity(seed_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, arch, matched, ctrls in (
            (axes[0], "Ridge", "R-MATCHED", ["R-CROSS", "R-SHUFFLED", "R-RANDOM"]),
            (axes[1], "NCR", "N-MATCHED", ["N-CROSS", "N-SHUFFLED", "N-RANDOM"])):
        width = 0.25
        for i, c in enumerate(ctrls):
            d = (seed_df[seed_df.model == matched].set_index("seed")["pearson"]
                 - seed_df[seed_df.model == c].set_index("seed")["pearson"])
            ax.bar(np.arange(len(FINAL510_SEEDS)) + (i - 1) * width, d.values,
                   width, label=f"{matched} - {c}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(FINAL510_SEEDS)))
        ax.set_xticklabels(FINAL510_SEEDS)
        ax.set_title(f"{arch} semantic specificity")
        ax.set_xlabel("outer-CV seed"); ax.legend(fontsize=7)
    axes[0].set_ylabel("paired delta Pearson r")
    fig.tight_layout()
    _save(fig, str(PLOTS / "fig3_wm_prior_specificity"))
    _save(fig, str(PR / "fig3_wm_prior_specificity"))
    plt.close(fig)


def fig_waterfall(abl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    comp = abl[abl.group == "B_matched_ridge_components"]
    r0 = abl[abl.group == "A_backbone"]
    stages = [("R0", float(r0["pearson"].mean())),
              ("+ FC expert", float(comp[comp.variant == "FC_only"]["pearson"].mean())),
              ("+ SC expert (expert)", float(comp[comp.variant == "FC_SC_expert"]["pearson"].mean())),
              ("+ R0 fusion (final)", float(comp[comp.variant == "R0_plus_expert"]["pearson"].mean()))]
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [s[0] for s in stages]; vals = [s[1] for s in stages]
    bars = ax.bar(names, vals, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("outer-CV Pearson r (mean over seeds)")
    ax.set_title("SPSEF-Ridge ablation waterfall (matched prior)")
    fig.tight_layout()
    _save(fig, str(PLOTS / "fig4_ablation_waterfall"))
    _save(fig, str(PR / "fig4_ablation_waterfall"))
    plt.close(fig)


def fig_stability(bm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for col, arch in enumerate(("ridge", "ncr")):
        sub = bm[bm.arch == arch]
        x = np.arange(len(sub))
        ax = axes[0, col]
        ax.bar(x - 0.2, sub["fc_edge_spearman"], 0.4, label="FC edge Spearman")
        ax.bar(x + 0.2, sub["sc_edge_spearman"], 0.4, label="SC edge Spearman")
        ax.set_xticks(x); ax.set_xticklabels(sub["prior"], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{arch.capitalize()}: edge-rank stability", fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[1, col]
        ax.bar(x - 0.25, sub["fc_top10_roi_jaccard"], 0.25, label="FC top10 ROI")
        ax.bar(x, sub["sc_top10_roi_jaccard"], 0.25, label="SC top10 ROI")
        ax.bar(x + 0.25, sub["multimodal_top10_roi_jaccard"], 0.25,
               label="multimodal top10 ROI")
        ax.set_xticks(x); ax.set_xticklabels(sub["prior"], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{arch.capitalize()}: top-10 ROI Jaccard", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("WM biomarker stability (modality-specific, corrected)", fontsize=10)
    fig.tight_layout()
    _save(fig, str(PLOTS / "fig5_wm_biomarker_stability"))
    _save(fig, str(PR / "fig5_wm_biomarker_stability"))
    plt.close(fig)


def fig_faithfulness(bm, arch="ridge", name="fig6_wm_faithfulness_ridge",
                     title="Ridge architecture"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = bm[bm.arch == arch]
    x = np.arange(len(sub)); w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, sub["top10_delta_rmse"], w, label="top10 $\\Delta$RMSE")
    ax.bar(x, sub["random10_mean"], w, label="random10 mean")
    ax.bar(x + w, sub["top10_minus_random10"], w, label="top10 - random10")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(sub["prior"])
    ax.set_ylabel("$\\Delta$RMSE (positive = faithful)")
    ax.set_title(f"WM cross-fold faithfulness: {title}")
    ax.legend(fontsize=8); fig.tight_layout()
    _save(fig, str(PLOTS / name))
    _save(fig, str(PR / name))
    plt.close(fig)


def _aal_centroids():
    """Actual AAL116 centroids from the repository atlas (never invented)."""
    try:
        import nibabel as nib
        img = nib.load(str(ROOT / "inputs" / "atlases" / "AAL116.nii.gz"))
        data = np.asarray(img.dataobj)
        zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
        cents = {}
        for i in range(1, 117):
            idx = np.argwhere(data == i)
            if len(idx) == 0:
                return None
            cents[i] = idx.mean(0) * zooms
        return cents
    except Exception:
        return None


def fig_roi_biomarker(ranking):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = ranking[ranking.model == "R-MATCHED"].sort_values("rank").head(12)
    cents = _aal_centroids()
    ncols = 3 if cents is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.6))
    axes[0].barh(sub["roi_label"][::-1], sub["mean_importance"][::-1], color="#c44e52")
    axes[0].set_title("Top WM ROI importance (R-MATCHED)", fontsize=9)
    axes[0].tick_params(labelsize=7)
    top = ranking[ranking.model == "R-MATCHED"].sort_values("rank").head(10)
    roi_idx = [int(x) for x in top["roi_index"]]
    imp = top.set_index("roi_index")["mean_importance"].to_dict()
    if cents is not None:
        allc = np.array([cents[i] for i in range(1, 117)])
        for ax, (ix, iy, name) in zip(axes[1:3],
                                      [(0, 1, "axial (x-y)"), (1, 2, "sagittal (y-z)")]):
            ax.scatter(allc[:, ix], allc[:, iy], s=6, color="#cccccc")
            for i in roi_idx:
                c = cents[i]
                ax.scatter([c[ix]], [c[iy]], s=30 + 300 * imp[i] / max(imp.values()),
                           color="#c44e52")
                ax.annotate(str(i), (c[ix], c[iy]), fontsize=5.5,
                            xytext=(2, 2), textcoords="offset points")
            for a in range(len(roi_idx)):
                for b in range(a + 1, len(roi_idx)):
                    ca, cb = cents[roi_idx[a]], cents[roi_idx[b]]
                    ax.plot([ca[ix], cb[ix]], [ca[iy], cb[iy]], color="gray",
                            lw=0.6, alpha=0.6)
            ax.set_title(f"AAL116 centroids, {name}", fontsize=9)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    else:
        ang = np.linspace(0, 2 * np.pi, len(roi_idx), endpoint=False)
        pos = {r: (np.cos(a), np.sin(a)) for r, a in zip(roi_idx, ang)}
        ax = axes[1]; ax.axis("off")
        for r in roi_idx:
            x, y = pos[r]
            ax.plot(x, y, "o", color="#4c72b0", ms=6)
            ax.text(x * 1.15, y * 1.15, str(r), ha="center", va="center", fontsize=6.5)
        for i, r1 in enumerate(roi_idx):
            for r2 in roi_idx[i + 1:]:
                x1, y1 = pos[r1]; x2, y2 = pos[r2]
                ax.plot([x1, x2], [y1, y2], color="gray", lw=0.6, alpha=0.6)
        ax.set_title("Top-10 WM ROI network", fontsize=9)
    fig.suptitle("Final WM semantic-expert biomarker", fontsize=10)
    fig.tight_layout()
    _save(fig, str(PLOTS / "fig7_wm_biomarker_roi"))
    _save(fig, str(PR / "fig7_wm_biomarker_roi"))
    plt.close(fig)
    # supplementary circular network with labels
    fig, ax = plt.subplots(figsize=(6.2, 6.0)); ax.axis("off")
    labels = top.set_index("roi_index").loc[roi_idx, "roi_label"].tolist()
    ang = np.linspace(0, 2 * np.pi, len(roi_idx), endpoint=False)
    pos = {r: (np.cos(a), np.sin(a)) for r, a in zip(roi_idx, ang)}
    for r, lb in zip(roi_idx, labels):
        x, y = pos[r]; ax.plot(x, y, "o", color="#4c72b0", ms=6)
        ax.text(x * 1.18, y * 1.18, lb, ha="center", va="center", fontsize=7)
    for i, r1 in enumerate(roi_idx):
        for r2 in roi_idx[i + 1:]:
            x1, y1 = pos[r1]; x2, y2 = pos[r2]
            ax.plot([x1, x2], [y1, y2], color="gray", lw=0.6, alpha=0.6)
    ax.set_title("Top-10 WM ROI network (final 510 cohort)", fontsize=9)
    fig.tight_layout()
    _save(fig, str(SUP / "fig_S9_wm_biomarker_network"))
    plt.close(fig)


def make_all_figures(priors, seed_df, abl, bm, ranking):
    fig_method_overview(priors)
    fig_priors_s1(priors)
    fig_masks_s2(priors)
    fig_ridge_ncr_s3()
    fig_fusion_s4()
    fig_controls_s5()
    fig_biomarker_s6()
    fig_faithfulness_s7()
    fig_prediction(seed_df)
    fig_specificity(seed_df)
    fig_waterfall(abl)
    fig_stability(bm)
    fig_faithfulness(bm, "ridge", "fig6_wm_faithfulness_ridge", "Ridge architecture")
    fig_faithfulness(bm, "ncr", "fig_S8_wm_faithfulness_ncr", "NCR architecture")
    fig_roi_biomarker(ranking)


# ══════════════════════════════════════════════════════════════════════
# Paper narrative / claims / novelty / examples
# ══════════════════════════════════════════════════════════════════════

TRANSPARENCY = ("> All reported performance estimates are obtained by repeated "
                "nested cross-validation within the final 510-subject cohort; "
                "no separate external validation cohort is claimed.")


def paper_status(plev, blev):
    if plev in ("P2", "P3") and blev != "B0":
        return "STRONG_WM_METHOD_PAPER"
    if plev in ("P2", "P3") and blev == "B0":
        return "WM_PREDICTION_METHOD_PAPER"
    if plev == "P1":
        return "EXPLORATORY_WM_METHOD_PAPER"
    return "NULL_OR_NEGATIVE_WM_METHOD_PAPER"


def write_narrative(prim, comp, plev, blev, sens, bm):
    pm = prim.set_index("model")
    c = comp.set_index("comparison")
    matched = pm.loc["R-MATCHED"]; base = pm.loc["R0"]
    c1 = c.loc["C1_R-MATCHED - R0"]
    sens_m = sens[sens.model == "R-MATCHED"].iloc[0]
    lines = ["# Final 510-subject Working-Memory results", "",
             TRANSPARENCY, "",
             "The primary estimand is the mean of seed-wise outer-CV Pearson "
             "values across 5 seeds x 5 folds (repeated nested CV). Ensemble "
             "sensitivity is reported separately and is not the primary metric.", "",
             "## 31.1 Working-memory prediction", "",
             f"- R0 backbone: r = {base['pearson_mean']:.4f} "
             f"(seed SD {base['pearson_std']:.4f}), RMSE = {base['rmse_mean']:.3f}, "
             f"MAE = {base['mae_mean']:.3f}.",
             f"- SPSEF-Ridge (R-MATCHED): r = {matched['pearson_mean']:.4f} "
             f"(seed SD {matched['pearson_std']:.4f}), RMSE = {matched['rmse_mean']:.3f}, "
             f"MAE = {matched['mae_mean']:.3f}.",
             f"- Paired delta r (SPSEF-Ridge - R0): {c1['mean_delta_r']:+.4f} "
             f"(positive seeds {int(c1['positive_seeds'])}/5).",
             f"- Ensemble sensitivity delta r vs R0: "
             f"{sens_m['ensemble_delta_r_vs_R0']:+.4f} "
             f"(95% CI [{sens_m['boot_ci_lo']:+.4f}, {sens_m['boot_ci_hi']:+.4f}]; "
             f"fraction <= 0 = {sens_m['fraction_le_0']:.3f}).",
             f"- RIDGE_PREDICTION_LEVEL: **{plev['RIDGE_PREDICTION_LEVEL']}**.",
             "",
             "## 31.2 Does the LLM semantic prior matter?", "",
             "Architecture-matched Ridge controls isolate prior identity:", ""]
    for ctrl in CONTROLS["R"]:
        row = c.loc[f"C2_R-MATCHED - {ctrl}"]
        lines.append(f"- R-MATCHED - {ctrl}: {row['mean_delta_r']:+.4f} "
                     f"(positive seeds {int(row['positive_seeds'])}/5).")
    lines += ["",
              f"Tier P3 requires positive mean deltas against all controls and "
              f">=4/5 positive seeds against shuffled and random; "
              f"achieved: {plev['beats_shuffled_random_4of5']}.",
              "",
              "## 31.3 Structured NCR regularization", ""]
    c3 = c.loc["C3_N-MATCHED - R-MATCHED"]
    lines.append(f"- N-MATCHED - R-MATCHED: {c3['mean_delta_r']:+.4f} "
                 f"(positive seeds {int(c3['positive_seeds'])}/5).")
    if c3["mean_delta_r"] > 0:
        lines.append("- NCR numerically exceeds Ridge in this run; treat as an "
                     "ablation result, not as the key predictive novelty unless "
                     "supported by the control comparisons.")
    else:
        lines.append("- NCR does not beat Ridge; NCR is reported as a structured-"
                     "regularization ablation and not as the strongest predictor.")
    lines += ["", "## 31.4 Biomarker reproducibility", "",
              "Coefficient stability across valid outer folds under architecture-"
              "matched priors is summarised below (see Table 5):", ""]
    for arch in ("ridge", "ncr"):
        sub = bm[bm.arch == arch]
        for _, r in sub.iterrows():
            lines.append(f"- {r['model']}: FC Spearman {r['fc_edge_spearman']:.3f}, "
                         f"SC Spearman {r['sc_edge_spearman']:.3f}, multimodal top10 "
                         f"Jaccard {r['multimodal_top10_roi_jaccard']:.3f}, sign "
                         f"consistency {r['sign_consistency']:.3f}.")
    lines += ["", "## 31.5 Biomarker faithfulness", "",
              "Outer-fold perturbation faithfulness (positive = faithful):", ""]
    for arch in ("ridge", "ncr"):
        for _, r in bm[bm.arch == arch].iterrows():
            lines.append(f"- {r['model']}: top10 delta_RMSE {r['top10_delta_rmse']:+.3f}, "
                         f"random10 mean {r['random10_mean']:+.3f}, "
                         f"top10-random10 {r['top10_minus_random10']:+.3f}, "
                         f"percentile {r['percentile']:.3f}, bottom10 "
                         f"{r['bottom10_delta_rmse']:+.3f}.")
    lines += ["", "## Decision", "",
              f"- RIDGE_PREDICTION_LEVEL: **{plev['RIDGE_PREDICTION_LEVEL']}**",
              f"- BIOMARKER_LEVEL (Ridge): **{blev['R']['level']}**",
              f"- BIOMARKER_LEVEL (NCR): **{blev['N']['level']}**",
              f"- FINAL510_PAPER_STATUS: **{paper_status(plev['RIDGE_PREDICTION_LEVEL'], blev['R']['level'])}**",
              "", TRANSPARENCY, ""]
    (PR / "MAIN_PAPER_RESULTS.md").write_text("\n".join(lines) + "\n")
    tex = ["% Auto-generated final 510 WM results (exploratory boundaries).",
           "\\section{Working memory on the final 510-subject cohort}",
           TRANSPARENCY.replace("> ", ""), "",
           f"The R0 backbone reaches $r={base['pearson_mean']:.3f}$ and SPSEF-Ridge "
           f"reaches $r={matched['pearson_mean']:.3f}$ "
           f"($\\Delta r={c1['mean_delta_r']:+.3f}$, "
           f"{int(c1['positive_seeds'])}/5 seeds positive). "
           f"RIDGE_PREDICTION_LEVEL is {plev['RIDGE_PREDICTION_LEVEL']}; "
           f"BIOMARKER_LEVEL is {blev['R']['level']}. "
           "Architecture-matched cross-task, shuffled, and random prior controls are "
           "reported in Table~\\ref{tab:final510_wm_prediction}.", ""]
    (PR / "MAIN_PAPER_RESULTS.tex").write_text("\n".join(tex))


def write_novelty():
    (PR / "NOVELTY_STATEMENT.md").write_text(
        "# Methodological contributions\n\n"
        "## Contribution 1 - LLM-generated task semantics as a frozen atlas prior\n"
        "A task description is converted into an AAL116 ROI prior by a frozen LLM "
        "call (temperature 0.2, seed 42). The prior is generated independently of "
        "HCP labels, prediction residuals, model coefficients, and test folds, and "
        "is versioned and SHA256-hashed.\n\n"
        "## Contribution 2 - Prior-guided subspace selection, not global prior "
        "regularization\n"
        "The prior defines compact FC/SC expert subspaces via direct top-K edge "
        "masks (q_ij = p_i p_j) and ROI-incident masks (top-M ROIs), rather than "
        "penalizing all 6670 edges.\n\n"
        "## Contribution 3 - Hierarchical fusion with a strong no-prior backbone\n"
        "The model preserves the validated corrected R0 backbone and adds a "
        "restricted prior-guided expert: yhat = (1-alpha) yhat_R0 + alpha "
        "(v yhat_FC + (1-v) yhat_SC), with v and alpha selected on training OOF only.\n\n"
        "## Contribution 4 - Architecture-matched semantic controls\n"
        "Matched, cross-task, shuffled, and random priors are evaluated under "
        "identical architecture, grids, preprocessing, folds, and tie-breaks; only "
        "the prior array changes. This separates semantic prior value from generic "
        "regularization or feature restriction. Claims are conditional on these "
        "controls (Tier P3).\n\n"
        "## Contribution 5 - Prediction-linked biomarker discovery\n"
        "The same expert that contributes to prediction yields primal coefficient "
        "maps c_m = alpha * w_m * beta_m, ROI importance "
        "I_i = sum_j |c_FC_ij| + sum_j |c_SC_ij|, evaluated by cross-fold stability "
        "and outer-fold perturbation faithfulness (masking edges to training-fold "
        "means without retraining).\n\n"
        "Novelty is not claimed merely because an LLM is used: the prior enters as a "
        "frozen external score vector that controls which edges are estimable, and "
        "its value is tested against architecture-matched controls.\n")


def write_claims(plev, blev, bm):
    pl = plev["RIDGE_PREDICTION_LEVEL"]
    bl = blev["R"]["level"]
    safe = ["# Safe claims (conditional on achieved evidence)", ""]
    if pl in ("P2", "P3"):
        safe += ["- \"On 510 HCP participants, SPSEF-Ridge consistently improved "
                 "repeated nested-CV WM prediction over the corrected no-prior "
                 "backbone.\" (supported: P2/P3)", ""]
    elif pl == "P1":
        safe += ["- \"On 510 HCP participants, SPSEF-Ridge numerically improved "
                 "repeated nested-CV WM prediction over the corrected no-prior "
                 "backbone, without consistent seed-level support.\"", ""]
    else:
        safe += ["- \"On 510 HCP participants, SPSEF-Ridge did not improve "
                 "repeated nested-CV WM prediction over the corrected no-prior "
                 "backbone.\"", ""]
    if pl == "P3":
        safe += ["- \"The matched LLM-derived WM prior outperformed "
                 "architecture-matched cross-task, shuffled, and random priors.\"",
                 ""]
    else:
        safe += ["- The matched LLM prior did not outperform all "
                 "architecture-matched controls; report the control comparisons "
                 "as negative or mixed.", ""]
    if bl in ("B2", "B3"):
        safe += ["- \"Matched-prior biomarkers were reproducible and predictively "
                 "faithful across outer folds.\"", ""]
    elif bl == "B1":
        safe += ["- \"Matched-prior biomarker maps showed nominal cross-fold "
                 "faithfulness; seed-level consistency was not established.\"", ""]
    else:
        safe += ["- \"Matched-prior biomarker maps did not show cross-fold "
                 "faithfulness on this cohort.\"", ""]
    safe += ["- \"All performance estimates are nested cross-validation results "
             "within a 510-subject cohort; no independent external validation is "
             "claimed.\"", ""]
    (PR / "SAFE_CLAIMS.md").write_text("\n".join(safe))
    (PR / "UNSAFE_CLAIMS.md").write_text(
        "# Unsafe claims (do not use)\n\n"
        "- \"independently validated\"\n"
        "- \"externally validated\"\n"
        "- \"the 98-subject holdout confirmed the method\"\n"
        "- \"the LLM prior improves prediction\" (unless Tier P3 supports it)\n"
        "- \"NCR is the key predictive novelty\" (unless C3 is positive)\n"
        "- \"the sample-size increase caused the improvement\"\n"
        "- any claim that the biomarker is causal rather than predictive-faithful\n")


def write_examples(recs, priors, faith, fold_df, sel):
    p = priors["matched"]
    order = np.argsort(p["array"])[::-1]
    top_roi = int(order[0])
    iu = np.triu_indices(116, 1)
    ep = p["array"][iu[0]] * p["array"][iu[1]]
    e = int(np.argmax(ep))
    rec = [r for r in recs if r["seed"] == 7171 and r["fold"] == 0][0]
    v = rec["variants"]["R-MATCHED"]
    selected_incident = int((v["fc_mask"] & ((iu[0] == top_roi) | (iu[1] == top_roi))).sum())
    edge_selected = bool(v["fc_mask"][e])
    te = rec["te"]
    frm = metrics_from = None
    row = faith[(faith.model == "R-MATCHED") & (faith.seed == 7171)
                & (faith.fold == 0)].iloc[0]
    lines = [
        "# Illustrative examples (all values traced to saved artifacts)", "",
        "## 1. High-prior WM ROI and selected incident edges", "",
        f"- Highest-prior ROI: index {top_roi + 1} ({p['labels'][top_roi]}), "
        f"prior score {p['array'][top_roi]:.4f}.",
        f"- In fold seed=7171/fold=0, the R-MATCHED FC mask contains "
        f"{selected_incident} edges incident to this ROI "
        f"(mask family {v['fc_family']}, size {v['fc_size']}, "
        f"{int(v['fc_mask'].sum())} edges total).",
        "",
        "## 2. Example direct top-K edge", "",
        f"- Candidate edge: ROI {iu[0][e] + 1} ({p['labels'][iu[0][e]]}) - "
        f"ROI {iu[1][e] + 1} ({p['labels'][iu[1][e]]}).",
        f"- p_i = {p['array'][iu[0][e]]:.4f}, p_j = {p['array'][iu[1][e]]:.4f}, "
        f"q_ij = p_i*p_j = {ep[e]:.4f}.",
        f"- Selected in the fold's FC mask: {edge_selected}.",
        "",
        "## 3. Fold example (seed=7171, fold=0, R-MATCHED)", "",
        f"- FC mask: {v['fc_family']} size {v['fc_size']} "
        f"({int(v['fc_mask'].sum())} edges); SC mask: {v['sc_family']} size "
        f"{v['sc_size']} ({int(v['sc_mask'].sum())} edges).",
        f"- v = {v['v_fc']:.2f}, alpha = {v['alpha']:.2f}, "
        f"lambda_fc = {v['lambda_r_fc']}, lambda_sc = {v['lambda_r_sc']}.",
        f"- Fold Pearson r: "
        f"{fold_df[(fold_df.model == 'R-MATCHED') & (fold_df.seed == 7171) & (fold_df.fold == 0)]['pearson'].iloc[0]:.4f}.",
        f"- Coefficient maps valid (alpha>0): {v['alpha'] > 0}.",
        "",
        "## 4. Biomarker perturbation example (seed=7171, fold=0, R-MATCHED)", "",
        f"- base RMSE = {row['base_rmse']:.4f}.",
        f"- top10 masked RMSE = {row['base_rmse'] + row['top10_delta_rmse']:.4f} "
        f"(delta = {row['top10_delta_rmse']:+.4f}).",
        f"- random10 mean delta = {row['random10_mean']:+.4f}; "
        f"top10 - random10 = {row['top10_minus_random10']:+.4f}.",
        f"- bottom10 delta = {row['bottom10_delta_rmse']:+.4f}.",
        "",
        "Masking replaces affected raw FC/SC edges with training-fold feature means; "
        "no retraining is performed.",
    ]
    (PR / "ILLUSTRATIVE_EXAMPLES.md").write_text("\n".join(lines) + "\n")


def write_supplementary(abl, sel, bm, recs, seed_df, fold_df, y, trans):
    lines = ["# Supplementary ablations - final 510 WM", "",
             "## Per-seed metrics", "",
             seed_df.pivot_table(index="model", columns="seed",
                                 values="pearson").round(4).to_string(), "",
             "## Per-fold metrics (all 25 folds)", "",
             "See `fold_metrics.csv`.", "",
             "## Selected masks / lambdas / ratios / v / alpha", "",
             "See `SELECTED_CONFIGS.csv`. Summary:", ""]
    g = sel.groupby("model").agg(
        fc_family=("fc_family", lambda x: x.value_counts().to_dict()),
        fc_size=("fc_size", lambda x: x.value_counts().to_dict()),
        sc_family=("sc_family", lambda x: x.value_counts().to_dict()),
        sc_size=("sc_size", lambda x: x.value_counts().to_dict()),
        lambda_fc_mean=("lambda_r_fc", "mean"),
        lambda_sc_mean=("lambda_r_sc", "mean"),
        ratio_fc_mean=("ratio_fc", "mean"),
        ratio_sc_mean=("ratio_sc", "mean"),
        v_mean=("v_fc", "mean"), alpha_mean=("alpha", "mean"),
        abstained=("abstained", "sum"))
    lines += ["```", g.to_string(), "```", "",
              "## FC-only / SC-only / expert / final (R-MATCHED)", ""]
    sub = abl[abl.group == "B_matched_ridge_components"]
    lines += ["```", sub.to_string(index=False), "```", "",
              "## Direct-edge vs ROI-incident (candidate inner-CV)", ""]
    sub = abl[abl.group == "F_G_mask_family_size"]
    lines += ["```", sub.to_string(index=False), "```", "",
              "## Ridge vs NCR and prior identity", ""]
    sub = abl[abl.group.isin(["C_ridge_prior_identity", "E_ncr_prior_identity"])]
    lines += ["```", sub.to_string(index=False), "```", "",
              "## NCR ratio ablation (inner-CV diagnostics)", ""]
    sub = abl[abl.group == "D_ncr_ratio"]
    lines += ["```", sub.to_string(index=False), "```", "",
              "## Top5/top10 faithfulness, random distributions, abstention", ""]
    lines += ["```", bm.to_string(index=False), "```", "",
              trans, ""]
    (SUP / "SUPPLEMENTARY_ABLATIONS.md").write_text("\n".join(lines))
    tex = ["% Auto-generated supplementary ablations (final 510 WM).",
           "\\section{Supplementary ablations}",
           "Per-seed and per-fold metrics, selected masks, lambdas, NCR ratios, "
           "fusion weights, prior-identity controls, mask-family/size sweeps, "
           "faithfulness distributions, abstention counts, and runtimes are "
           "provided in the companion CSV files. The headline control comparison "
           "uses identical architectures and grids; only the prior array changes.",
           ""]
    (SUP / "SUPPLEMENTARY_ABLATIONS.tex").write_text("\n".join(tex))
    (SUP / "DEVELOPMENT_HISTORY_NOTE.md").write_text(
        "# Development history note (supplementary)\n\n"
        "Substantial method development preceded the final 510-subject study. "
        "Earlier exploratory configurations, forensic audits, and negative "
        "development results are retained separately in the repository for "
        "reproducibility (for example, prior PALF/PS-NCR development outputs and "
        "the 412-vs-510 scaling study). Those artifacts are development history, "
        "not results of the final paper. Because model selection for this paper "
        "was finalized within the same cohort used for estimation, the final "
        "results should be interpreted as within-cohort cross-validation rather "
        "than external confirmation.\n")


# ══════════════════════════════════════════════════════════════════════
# Validation + evidence audit
# ══════════════════════════════════════════════════════════════════════

BANNED = ["externally validated", "independently validated",
          "independent confirmation", "98-subject holdout confirmed"]
NEGATED_OK = ["external validation", "independent validation"]


def scan_banned(paths):
    hits = {}
    for p in paths:
        if p.is_file() and p.suffix in (".md", ".tex", ".csv") \
                and "unsafe" not in p.name.lower():
            txt = p.read_text(errors="ignore")
            low = txt.lower()
            found = [b for b in BANNED if b in low]
            for b in NEGATED_OK:
                start = 0
                while True:
                    i = low.find(b, start)
                    if i < 0:
                        break
                    if "no " not in low[max(0, i - 60):i]:
                        found.append(b)
                        break
                    start = i + len(b)
            if found:
                hits[str(p.relative_to(OUT))] = sorted(set(found))
    return hits


def write_validation(recs, priors, plevel, blevel, status):
    recon = max(r["variants"][m]["recon_err"] for r in recs for m in ALL_MODELS)
    r0recon = max(r["r0"]["recon_err"] for r in recs)
    leakage = []
    for r in recs:
        leakage.append(bool(np.intersect1d(r["tr"], r["te"]).size))
    audit_status = json.loads((OUT / "BASELINE_AUDIT.json").read_text())["status"]
    banned_hits = scan_banned(list(PR.rglob("*")) + list(SUP.rglob("*")))
    val = {
        "r0_audit_status": audit_status,
        "max_final_reconstruction_error": float(recon),
        "max_r0_map_reconstruction_error": float(r0recon),
        "reconstruction_tolerance": 1e-8,
        "outer_split_leakage_any": bool(any(leakage)),
        "n_folds": len(recs), "n_models": len(ALL_MODELS),
        "prior_sha256": {k: v["sha256"] for k, v in priors.items()},
        "banned_wording_hits": banned_hits,
        "rdige_prediction_level": plevel, "biomarker_levels": blevel,
        "paper_status": status,
        "primary_estimand": "mean seed-wise outer-CV Pearson",
        "sensitivity_estimand": "ensemble Pearson after averaging repeated-CV "
                                "predictions (not primary)",
    }
    (OUT / "VALIDATION_REPORT.json").write_text(json.dumps(val, indent=2))
    return val


def evidence_audit(prim, comp, plev, blev, bm, val, sens, t_total):
    pm = prim.set_index("model")
    lines = [
        "# FINAL 510 WM evidence audit", "",
        "## A. Runtime", f"- total report+analysis time: {t_total:.0f}s", "",
        "## B. Cohort", "- n = 510 unique subjects; see COHORT_AUDIT.json", "",
        "## C. R0 validation", f"- historical implementation audit: "
        f"{val['r0_audit_status']}", "",
        "## D. Final WM prediction (mean seed-wise outer-CV)", ""]
    for m in MODEL_ORDER:
        r = pm.loc[m]
        lines.append(f"- {m}: r={r['pearson_mean']:.4f} (SD {r['pearson_std']:.4f}), "
                     f"RMSE={r['rmse_mean']:.3f}, MAE={r['mae_mean']:.3f}, "
                     f"delta vs R0={r['delta_r_mean']:+.4f}, positive seeds "
                     f"{int(r['positive_seeds'])}/5")
    lines += ["", "## E. Semantic prior evidence (architecture-matched)", ""]
    for name, row in comp.set_index("comparison").iterrows():
        if name.startswith(("C2_", "C4_")):
            lines.append(f"- {name}: {row['mean_delta_r']:+.4f} "
                         f"(positive seeds {int(row['positive_seeds'])}/5)")
    lines += ["", "## F. Ablations", "- See prediction_ablations.csv and "
              "ABLATION_PREDICTION.md.", "",
              "## G. Biomarker stability", ""]
    for _, r in bm.iterrows():
        lines.append(f"- {r['model']}: FC Spearman {r['fc_edge_spearman']:.3f}, "
                     f"SC Spearman {r['sc_edge_spearman']:.3f}, top10 Jaccard "
                     f"{r['multimodal_top10_roi_jaccard']:.3f}, sign "
                     f"{r['sign_consistency']:.3f}")
    lines += ["", "## H. Faithfulness", ""]
    for _, r in bm.iterrows():
        lines.append(f"- {r['model']}: top10 {r['top10_delta_rmse']:+.3f}, random10 "
                     f"{r['random10_mean']:+.3f}, contrast "
                     f"{r['top10_minus_random10']:+.3f}, bottom10 "
                     f"{r['bottom10_delta_rmse']:+.3f}")
    lines += ["", "## I. Top WM biomarkers",
              "- See paper_ready/table_final_wm_roi_ranking.csv and Figure 7.", "",
              "## J. Novelty", "- See paper_ready/NOVELTY_STATEMENT.md.", "",
              "## K. Paper figures",
              "- paper_ready/fig_method_overview.*; plots/fig2..fig7; "
              "supplementary/fig_S1..fig_S8.", "",
              "## L. Claim levels",
              f"- RIDGE_PREDICTION_LEVEL: {plev['RIDGE_PREDICTION_LEVEL']}",
              f"- BIOMARKER_LEVEL Ridge: {blev['R']['level']}",
              f"- BIOMARKER_LEVEL NCR: {blev['N']['level']}", "",
              "## M. Paper status",
              f"- FINAL510_PAPER_STATUS: {val['paper_status']}", "",
              "STATUS: FINAL510_WM_EVIDENCE_FREEZE_COMPLETE"]
    txt = "\n".join(lines) + "\n"
    (OUT / "FINAL_EVIDENCE_AUDIT.md").write_text(txt)
    return txt


# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70); print("FINAL 510 WM: reports, tables, figures"); print("=" * 70)
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    y = y_wm
    recs = load_folds()
    priors = load_priors()
    seed_df = seedwise_table(recs, y)
    fold_df = foldwise_table(recs, y)
    prim = primary_table(seed_df)
    sens, ens = sensitivity(recs, y)
    comp = comparisons(seed_df, prim)
    sel = selected_configs(recs)
    ctrl = control_tables(recs, y)
    write_core_tables(recs, y, seed_df, fold_df, prim, sens, comp, sel, ctrl)

    stab_path = OUT / "biomarker_stability_corrected.csv"
    stab = pd.read_csv(stab_path if stab_path.exists()
                       else OUT / "biomarker_stability.csv")
    faith = pd.read_csv(OUT / "biomarker_faithfulness.csv")
    bm = biomarker_tables(stab, faith)
    bm.to_csv(OUT / "biomarker_table.csv", index=False)
    plev = prediction_tier(prim, comp)
    blev = biomarker_levels(bm, faith)
    status = paper_status(plev["RIDGE_PREDICTION_LEVEL"], blev["R"]["level"])
    ranking = pd.read_csv(OUT / "final_wm_biomarker_ranking.csv")
    main_table_tex(prim, comp, PR / "table_main_wm_prediction.tex",
                   PR / "table_main_wm_prediction.csv")
    biomarker_table_tex(bm, PR / "table_main_wm_biomarkers.tex",
                        PR / "table_main_wm_biomarkers.csv")
    ranking[ranking.model == "R-MATCHED"].to_csv(
        PR / "table_final_wm_roi_ranking.csv", index=False)
    defs = comparisons(seed_df, prim)
    comp.to_csv(OUT / "prediction_comparisons.csv", index=False)
    abl = pd.read_csv(OUT / "prediction_ablations.csv")
    make_all_figures(priors, seed_df, abl, bm, ranking)
    write_narrative(prim, comp, plev, blev, sens, bm)
    write_novelty()
    write_claims(plev, blev, bm)
    write_examples(recs, priors, faith, fold_df, sel)
    write_supplementary(abl, sel, bm, recs, seed_df, fold_df, y, TRANSPARENCY)
    reuse = result_reuse_audit()
    (OUT / "RESULT_REUSE_AUDIT.json").write_text(json.dumps(reuse, indent=2))
    val = write_validation(recs, priors, plev, blev, status)
    audit = evidence_audit(prim, comp, plev, blev, bm, val, sens, time.time() - t0)
    (OUT / "FINAL_STATUS.json").write_text(json.dumps({
        "prediction_level": plev, "biomarker_levels": blev,
        "paper_status": status,
        "primary_estimand": "mean seed-wise outer-CV Pearson",
        "transparency": TRANSPARENCY,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    (OUT / "models" / "FINAL_METHOD_SPEC.json").write_text(json.dumps({
        "method": "SPSEF (Semantic Prior-Selected Expert Fusion)",
        "model_aliases": {"A0": "R0 (corrected no-prior backbone)",
                          "A1": "R-MATCHED (SPSEF-Ridge, ratio=0)",
                          "A2": "N-MATCHED (SPSEF-NCR, frozen ratio grid)"},
        "variants": ["SPSEF-Ridge (ratio=0)", "SPSEF-NCR (frozen ratio grid)"],
        "architecture_matched_controls": {
            "ridge": ["R-MATCHED", "R-CROSS", "R-SHUFFLED", "R-RANDOM"],
            "ncr": ["N-MATCHED", "N-CROSS", "N-SHUFFLED", "N-RANDOM"]},
        "backbone": "corrected same-solver R0 (FP + SC Ridge + cross-fitted "
                    "convex fusion)",
        "masks": {"direct_topk_K": [100, 300, 600, 1200],
                  "roi_incident_M": [5, 10, 15]},
        "ridge_grid": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "ncr_ratio_grid": [0.0, 0.1, 0.3, 1.0],
        "fusion_grid": "v, alpha in {0,0.05,...,1} selected on training OOF only",
        "cv": {"outer_seeds": FINAL510_SEEDS, "outer_folds": 5, "inner_folds": 3,
               "cohort": 510},
        "priors": {k: {"path": v["path"], "sha256": v["sha256"]}
                   for k, v in priors.items()},
        "claim_boundary": "repeated nested CV within the 510-subject cohort; "
                          "NOT independent external validation",
    }, indent=2))
    (OUT / "COMPLETE").write_text("FINAL510_WM_EVIDENCE_FREEZE_COMPLETE\n")
    print(audit)
    print(f"  report done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()


