#!/usr/bin/env python3
"""Scaling study report: aggregation, bootstrap, ablations, plots, paper-ready."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "iclr" / "palf_412_vs_510_scaling"
STATE = OUT / "_state"
CKPT = STATE / "splits"
sys.path.insert(0, str(ROOT / "src"))
from metascfc.experiments.palf_crossfit_ablation import make_outer_splits  # noqa
from scaling_412_vs_510_pilot import (SCALING_OUTER_SEEDS, OUTER_FOLDS,  # noqa
                                      ALL_MODELS, TASKS, load_combined)

BOOT_SEED = 9401
N_BOOT = 10000


def metrics(y, p):
    r = float(pearsonr(y, p).statistic) if np.std(p) > 0 else 0.0
    return {"pearson": r, "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae": float(np.mean(np.abs(p - y)))}


def load_all():
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    recs = {}
    for seed in SCALING_OUTER_SEEDS:
        for _, fold, tr412, te in make_outer_splits(412, [seed], OUTER_FOLDS):
            recs[(seed, fold)] = pickle.load(open(CKPT / f"split_seed{seed}_fold{fold}.pkl", "rb"))
    return X_fc, X_sc, y_wm, y_fi, ids, recs


def subject_matrix(recs, target, model, regime, y):
    """(412, n_seeds) predictions, one per seed, every subject once."""
    n = 412
    seed_idx = {s: i for i, s in enumerate(SCALING_OUTER_SEEDS)}
    P = np.full((n, len(SCALING_OUTER_SEEDS)), np.nan)
    for (seed, fold), rec in recs.items():
        te = rec[regime]["te"]
        P[te, seed_idx[seed]] = rec[regime]["targets"][target]["preds"][model]
    assert np.all(np.isfinite(P))
    return P


def primary_tables(recs, y_wm, y_fi):
    rows = []
    for target, y in (("WM", y_wm), ("FI", y_fi)):
        for model in ALL_MODELS:
            for regime in ("A", "B"):
                P = subject_matrix(recs, target, model, regime, y)
                for si, seed in enumerate(SCALING_OUTER_SEEDS):
                    m = metrics(y[:412], P[:, si])
                    rows.append({"target": target, "model": model, "regime": regime,
                                 "seed": seed, **m})
    return pd.DataFrame(rows)


def model_metrics(seed_df):
    rows = []
    for (target, model, regime), sub in seed_df.groupby(["target", "model", "regime"]):
        rows.append({"target": target, "model": model, "regime": regime,
                     "r_mean": sub["pearson"].mean(), "r_median": sub["pearson"].median(),
                     "r_std": sub["pearson"].std(),
                     "rmse_mean": sub["rmse"].mean(), "mae_mean": sub["mae"].mean(),
                     "n_seeds": len(sub)})
    return pd.DataFrame(rows)


def paired_bootstrap(recs, target, model, y):
    PA = subject_matrix(recs, target, model, "A", y)[:, :].mean(1)
    PB = subject_matrix(recs, target, model, "B", y)[:, :].mean(1)
    rA = pearsonr(y[:412], PA).statistic
    rB = pearsonr(y[:412], PB).statistic
    rng = np.random.RandomState(BOOT_SEED)
    idx = rng.randint(0, 412, size=(N_BOOT, 412))
    dr = np.zeros(N_BOOT); drm = np.zeros(N_BOOT); dma = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        ib = idx[b]
        dr[b] = (pearsonr(y[:412][ib], PB[ib]).statistic
                 - pearsonr(y[:412][ib], PA[ib]).statistic)
        rmA = np.sqrt(np.mean((PA[ib] - y[:412][ib]) ** 2))
        rmB = np.sqrt(np.mean((PB[ib] - y[:412][ib]) ** 2))
        drm[b] = rmB - rmA
        dma[b] = (np.mean(np.abs(PB[ib] - y[:412][ib]))
                  - np.mean(np.abs(PA[ib] - y[:412][ib])))
    fz = np.arctanh(np.clip(rB, -0.999, 0.999)) - np.arctanh(np.clip(rA, -0.999, 0.999))
    return {"target": target, "model": model,
            "r_412": float(rA), "r_plus98": float(rB),
            "delta_r_observed": float(rB - rA),
            "delta_r_mean": float(dr.mean()),
            "delta_r_ci_lo": float(np.percentile(dr, 2.5)),
            "delta_r_ci_hi": float(np.percentile(dr, 97.5)),
            "delta_r_fraction_le_0": float(np.mean(dr <= 0)),
            "delta_rmse_mean": float(drm.mean()),
            "delta_rmse_ci_lo": float(np.percentile(drm, 2.5)),
            "delta_rmse_ci_hi": float(np.percentile(drm, 97.5)),
            "delta_mae_mean": float(dma.mean()),
            "delta_mae_ci_lo": float(np.percentile(dma, 2.5)),
            "delta_mae_ci_hi": float(np.percentile(dma, 97.5)),
            "fisher_z_diff": float(fz)}


def seed_deltas_primary(seed_df, target, model):
    a = seed_df[(seed_df.target == target) & (seed_df.model == model) & (seed_df.regime == "A")].set_index("seed")["pearson"]
    b = seed_df[(seed_df.target == target) & (seed_df.model == model) & (seed_df.regime == "B")].set_index("seed")["pearson"]
    d = (b - a).reindex(SCALING_OUTER_SEEDS)
    return d


def tier(delta_mean_seed, pos, ci_lo):
    if delta_mean_seed >= 0.005 and ci_lo > 0:
        return "ROBUST_SCALE_GAIN"
    if delta_mean_seed >= 0.005 and pos >= 4:
        return "CONSISTENT_SCALE_GAIN"
    if delta_mean_seed > 0:
        return "NOMINAL_SCALE_GAIN"
    return "NO_GAIN"


def method_effects(seed_df):
    """Seed-level deltas within each regime (method contrasts)."""
    contrasts = [("PALF_full-R0", "A1", "A0"), ("PALF_aniso-R0", "A2", "A0"),
                 ("PALF_network-R0", "A3", "A0"), ("NCR-Ridge", "B1", "B0"),
                 ("NCR-cross", "B1", "B2"), ("NCR-shuffled", "B1", "B3"),
                 ("NCR-random", "B1", "B4"), ("NCR-R0", "B1", "A0")]
    rows = []
    for target in TASKS:
        for name, m1, m2 in contrasts:
            for regime in ("A", "B"):
                d = []
                for seed in SCALING_OUTER_SEEDS:
                    a = seed_df[(seed_df.target == target) & (seed_df.model == m1)
                                & (seed_df.regime == regime) & (seed_df.seed == seed)]["pearson"].item()
                    b = seed_df[(seed_df.target == target) & (seed_df.model == m2)
                                & (seed_df.regime == regime) & (seed_df.seed == seed)]["pearson"].item()
                    d.append(a - b)
                d = np.array(d)
                rows.append({"target": target, "contrast": name, "regime": regime,
                             "mean": d.mean(), "median": np.median(d),
                             "positive": int((d > 0).sum()), "n": 5,
                             "seeds": list(np.round(d, 6))})
    return pd.DataFrame(rows)


def interactions(seed_df):
    """Method advantage at +98 minus advantage at 412 (paired by seed)."""
    rows = []
    for target in TASKS:
        for name, m1, m2 in [("B1_R0", "B1", "A0"), ("PALF_R0", "A1", "A0"),
                             ("NCR_Ridge", "B1", "B0")]:
            d = []
            for seed in SCALING_OUTER_SEEDS:
                advB = (seed_df[(seed_df.target == target) & (seed_df.model == m1)
                                & (seed_df.regime == "B") & (seed_df.seed == seed)]["pearson"].item()
                        - seed_df[(seed_df.target == target) & (seed_df.model == m2)
                                  & (seed_df.regime == "B") & (seed_df.seed == seed)]["pearson"].item())
                advA = (seed_df[(seed_df.target == target) & (seed_df.model == m1)
                                & (seed_df.regime == "A") & (seed_df.seed == seed)]["pearson"].item()
                        - seed_df[(seed_df.target == target) & (seed_df.model == m2)
                                  & (seed_df.regime == "A") & (seed_df.seed == seed)]["pearson"].item())
                d.append(advB - advA)
            d = np.array(d)
            rows.append({"target": target, "interaction": name,
                         "mean": d.mean(), "median": np.median(d),
                         "positive": int((d > 0).sum()), "n": 5, "seeds": list(np.round(d, 6))})
    return pd.DataFrame(rows)


def _size_label(n_requested, n_eff):
    if n_requested in (329, 330) and n_eff == n_requested:
        return "full_T412"
    if n_requested in (427, 428) and n_eff == n_requested:
        return "full_TB"
    return f"n{n_requested}"


def learning_table():
    p = STATE / "learning_preds.pkl"
    if not p.exists():
        return None
    rows = pickle.load(open(p, "rb"))
    X_fc, X_sc, y_wm, y_fi, ids = load_combined()
    ymap = {"WM": y_wm, "FI": y_fi}
    out = []
    for target in TASKS:
        for model in ["A0", "B1"]:
            labels = sorted({_size_label(r["n_requested"], r["n_eff"]) for r in rows})
            for label in labels:
                sub = [r for r in rows if r["target"] == target and r["model"] == model
                       and _size_label(r["n_requested"], r["n_eff"]) == label]
                vals = {}
                for r in sub:
                    key = (r["seed"], r["subset_seed"])
                    vals.setdefault(key, []).append((r["fold"], r["pred"]))
                rs = []
                for key, folds in vals.items():
                    seed = key[0]
                    concat = []
                    yy = []
                    for fold, pred in sorted(folds):
                        rec = pickle.load(open(CKPT / f"split_seed{seed}_fold{fold}.pkl", "rb"))
                        te = rec["A"]["te"]
                        concat.append(pred); yy.append(ymap[target][te])
                    rs.append(pearsonr(np.concatenate(yy), np.concatenate(concat)).statistic)
                out.append({"target": target, "model": model, "size_label": label,
                            "n_requested": int(np.mean([r["n_requested"] for r in sub])),
                            "r_mean": float(np.mean(rs)), "r_std": float(np.std(rs)),
                            "n_eff_mean": float(np.mean([r["n_eff"] for r in sub])),
                            "n_runs": len(rs)})
    order = {"n250": 0, "n300": 1, "n350": 2, "full_T412": 3, "full_TB": 4}
    df = pd.DataFrame(out)
    df["ord"] = df["size_label"].map(order)
    return df.sort_values(["target", "model", "ord"]).drop(columns="ord").reset_index(drop=True)


def ablation_tables(recs, seed_df, X_fc, X_sc, y_wm, y_fi):
    rows = []
    # A sample-size: delta per model
    for target in TASKS:
        for model in ALL_MODELS:
            d = seed_deltas_primary(seed_df, target, model)
            rows.append({"group": "A_sample_size", "target": target, "variant": model,
                         "metric": "delta_r(+98-412)", "value": d.mean(),
                         "positive": int((d > 0).sum()), "n": 5})
    # B PALF components (within regime)
    me = method_effects(seed_df)
    for _, r in me.iterrows():
        rows.append({"group": "B_PALF_components", "target": r["target"],
                     "variant": r["contrast"], "metric": r["regime"],
                     "value": r["mean"], "positive": r["positive"], "n": 5})
    # C NCR vs Ridge, D priors included above.
    # E modality for B1; F fusion; using stored expert parts
    for target, y in (("WM", y_wm), ("FI", y_fi)):
        for regime in ("A", "B"):
            fc_only, sc_only, expert_only, fused = [], [], [], []
            for (seed, fold), rec in recs.items():
                t = rec[regime]["targets"][target]
                e = t["experts"]["B1"]
                base = t["preds"]["A0"]; a = e["alpha"]
                fc_only.append((1 - a) * base + a * e["fc_test"])
                sc_only.append((1 - a) * base + a * e["sc_test"])
                expert_only.append(e["expert_test"])
                fused.append(e["final_test"])
            # concatenate all 25 folds/seeds (descriptive)
            for name, arr in [("B1_fc_only", fc_only), ("B1_sc_only", sc_only),
                              ("B1_expert_only", expert_only), ("B1_fused", fused)]:
                cat, yy = [], []
                for ((seed, fold), rec), p in zip(sorted(recs.items()), arr):
                    te = rec[regime]["te"]
                    cat.append(p); yy.append(y[te])
                m = metrics(np.concatenate(yy), np.concatenate(cat))
                rows.append({"group": "E_modality_F_fusion", "target": target,
                             "variant": f"{name}_{regime}", "metric": "pearson",
                             "value": m["pearson"], "positive": np.nan, "n": 25})
    # G mask family
    fam = []
    for target in TASKS:
        for regime in ("A", "B"):
            for (seed, fold), rec in recs.items():
                e = rec[regime]["targets"][target]["experts"]["B1"]
                fam.append({"target": target, "regime": regime,
                            "fc_family": e["families"][0], "sc_family": e["families"][2],
                            "alpha": e["alpha"], "v_fc": e["v_fc"]})
    fdf = pd.DataFrame(fam)
    for target in TASKS:
        for regime in ("A", "B"):
            sub = fdf[(fdf.target == target) & (fdf.regime == regime)]
            for famname in ("direct_topk", "roi_incident"):
                n = int((sub["fc_family"] == famname).sum())
                rows.append({"group": "G_mask_family", "target": target,
                             "variant": f"{famname}_{regime}", "metric": "n_selected_fc",
                             "value": n, "positive": np.nan, "n": len(sub)})
    return pd.DataFrame(rows), fdf


def make_plots(seed_df, model_df, boot_df, stab, faith, learn,
               X_fc, X_sc, recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = OUT / "plots"; P.mkdir(exist_ok=True)
    (OUT / "paper_ready").mkdir(exist_ok=True)
    (OUT / "supplementary").mkdir(exist_ok=True)
    # Fig 1 framework diagram
    fig, ax = plt.subplots(figsize=(9, 5)); ax.axis("off")
    boxes = [
        (0.05, 0.75, "D412 (412 subjects)"),
        (0.05, 0.15, "D98 (98 additional)"),
        (0.38, 0.75, "Outer folds V412\n(identical for A and B)"),
        (0.38, 0.45, "Regime A train: T412"),
        (0.38, 0.15, "Regime B train: T412 + D98"),
        (0.70, 0.45, "R0 / PALF / PS-NCR-EF"),
        (0.70, 0.10, "prediction + biomarker\nstability + faithfulness"),
    ]
    for x, y, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.24, 0.16, fill=False, lw=1.2))
        ax.text(x + 0.12, y + 0.08, t, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.38, 0.83), xytext=(0.29, 0.83),
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.38, 0.23), xytext=(0.29, 0.23),
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.70, 0.53), xytext=(0.62, 0.53),
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.70, 0.18), xytext=(0.62, 0.18),
                arrowprops=dict(arrowstyle="->"))
    ax.text(0.5, 0.98, "Paired sample-size scaling design (exploratory)",
            ha="center", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "paper_ready" / f"fig_main_scaling_framework.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    # Fig 2 prediction
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, target in zip(axes, TASKS):
        for model, col in [("A0", "C0"), ("A1", "C1"), ("B1", "C2")]:
            a = seed_df[(seed_df.target == target) & (seed_df.model == model) & (seed_df.regime == "A")].set_index("seed")["pearson"]
            b = seed_df[(seed_df.target == target) & (seed_df.model == model) & (seed_df.regime == "B")].set_index("seed")["pearson"]
            for s in SCALING_OUTER_SEEDS:
                ax.plot([0, 1], [a[s], b[s]], marker="o", color=col, alpha=0.7, lw=1)
            ax.plot([0, 1], [a.mean(), b.mean()], color=col, lw=3, label=model)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["412-only", "+98"])
        ax.set_title(target); ax.set_ylabel("Pearson r"); ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(P / f"fig_phase_scaling_prediction.{ext}")
    plt.close(fig)
    # Fig 3 method effect across sample size
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, target in zip(axes, TASKS):
        for model in ["A0", "A1", "B0", "B1"]:
            a = seed_df[(seed_df.target == target) & (seed_df.model == model) & (seed_df.regime == "A")]["pearson"].mean()
            b = seed_df[(seed_df.target == target) & (seed_df.model == model) & (seed_df.regime == "B")]["pearson"].mean()
            ax.plot([0, 1], [a, b], marker="o", label=model)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["412-only", "+98"])
        ax.set_title(target); ax.set_ylabel("Pearson r"); ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(P / f"fig_phase_scaling_method_effect.{ext}")
    plt.close(fig)
    # Fig 4 learning curve
    if learn is not None:
        order = ["n250", "n300", "n350", "full_T412", "full_TB"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, target in zip(axes, TASKS):
            for model in ["A0", "B1"]:
                sub = learn[(learn.target == target) & (learn.model == model)]
                sub = sub.set_index("size_label").reindex(order).reset_index()
                ax.errorbar(range(len(order)), sub["r_mean"], yerr=sub["r_std"],
                            marker="o", capsize=3, label=model)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(["250", "300", "350", "T412", "T412+D98"])
            ax.set_xlabel("training subjects"); ax.set_ylabel("Pearson r")
            ax.set_title(target); ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png", "svg"):
            fig.savefig(P / f"fig_phase_scaling_learning_curve.{ext}")
        plt.close(fig)
    # Fig 5 stability
    fig, ax = plt.subplots(figsize=(8, 4))
    g5 = stab.groupby(["target", "regime"])["fc_edge_spearman"].mean().reset_index()
    labels5 = [t for t in TASKS]
    x = np.arange(len(labels5))
    ax.bar(x - 0.2, g5[g5.regime == "A"]["fc_edge_spearman"].values, 0.4, label="412")
    ax.bar(x + 0.2, g5[g5.regime == "B"]["fc_edge_spearman"].values, 0.4, label="+98")
    ax.set_xticks(x); ax.set_xticklabels(labels5)
    ax.set_ylabel("FC abs-edge Spearman"); ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(P / f"fig_phase_scaling_stability.{ext}")
    plt.close(fig)
    # Fig 6 faithfulness
    f = faith[faith.abstained == 0]
    g = f.groupby(["target", "model", "regime"])["top10_minus_random10"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r.target}-{r.model}" for r in g[g.regime == "A"].itertuples()]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, g[g.regime == "A"]["top10_minus_random10"].values, 0.4, label="412")
    ax.bar(x + 0.2, g[g.regime == "B"]["top10_minus_random10"].values, 0.4, label="+98")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("top10 - random10 delta_RMSE"); ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(P / f"fig_phase_scaling_faithfulness.{ext}")
    plt.close(fig)
    # supplementary diagrams S1-S6 (simple box diagrams)
    diagrams = {
        "S1_full_nested_cv": ["Outer D412 fold", "Regime A: T412", "Regime B: T412+D98",
                              "Inner 3-fold CV (train-only selection)",
                              "Same V412 evaluation", "No leakage"],
        "S2_PALF_ablation": ["R0", "+ anisotropy", "+ network", "= full PALF", "matched prior"],
        "S3_PS_NCR_EF_architecture": ["Prior", "FC/SC subspace", "Ridge/NCR",
                                      "FC-SC expert fusion", "Fusion with R0"],
        "S4_biomarker_faithfulness": ["Training fold", "coefficient map", "ROI ranking",
                                      "top/random/bottom masks", "same V412", "delta_RMSE"],
        "S5_learning_curve": ["n=250", "300", "350", "T412", "T412+D98", "fixed V412"],
        "S6_prior_controls": ["matched", "cross-task", "shuffled", "random", "same architecture"],
    }
    for name, items in diagrams.items():
        fig, ax = plt.subplots(figsize=(9, 2.2)); ax.axis("off")
        for i, t in enumerate(items):
            xx = 0.02 + i * (0.96 / len(items))
            ax.add_patch(plt.Rectangle((xx, 0.25), 0.9 / len(items), 0.5, fill=False, lw=1))
            ax.text(xx + 0.45 / len(items), 0.5, t, ha="center", va="center", fontsize=7)
            if i:
                ax.annotate("", xy=(xx, 0.5), xytext=(xx - 0.02, 0.5),
                            arrowprops=dict(arrowstyle="->"))
        fig.tight_layout()
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / "supplementary" / f"fig_{name}.{ext}", bbox_inches="tight")
        plt.close(fig)


def paper_ready(seed_df, model_df, boot_df, me, inter, stab, faith, learn):
    PR = OUT / "paper_ready"; PR.mkdir(exist_ok=True)
    nominal = boot_df[boot_df["delta_r_observed"] > 0]
    trigger = len(nominal) > 0
    (OUT / "STUDY_STATUS.md").write_text(
        "# Study status\n\n"
        f"- PAPER_READY_TRIGGER: {'YES' if trigger else 'NO'}\n"
        "- Analysis type: post-hoc combined-cohort sample-size scaling (exploratory)\n"
        "- NOT independent / external validation\n"
        "- Phase 4 result preserved and not reinterpreted\n")
    if not trigger:
        return False
    # results md
    lines = ["# Paper-ready scaling results (exploratory)", "",
             "This is a post-hoc combined-cohort sample-size scaling analysis.",
             "It is NOT independent or external validation.", "",
             "## Paired 412 vs +98 prediction", ""]
    for _, r in boot_df.sort_values(["target", "model"]).iterrows():
        d = seed_deltas_primary(seed_df, r["target"], r["model"])
        lines += [
            f"### {r['target']} / {r['model']}",
            f"- 412-only Pearson r = {r['r_412']:.4f}",
            f"- +98 Pearson r = {r['r_plus98']:.4f}",
            f"- delta_r = {r['delta_r_observed']:+.4f} "
            f"(paired bootstrap 95% CI [{r['delta_r_ci_lo']:+.4f}, {r['delta_r_ci_hi']:+.4f}])",
            f"- positive seed deltas = {int((d > 0).sum())}/5; "
            f"mean seed delta_r = {d.mean():+.4f}; median = {np.median(d):+.4f}",
            f"- tier = {tier(d.mean(), int((d > 0).sum()), r['delta_r_ci_lo'])}",
            "",
            f"> Under the paired training-expansion protocol, increasing the training pool "
            f"from T412 to T412+98 changed {r['target']} {r['model']} Pearson r from "
            f"{r['r_412']:.4f} to {r['r_plus98']:.4f} "
            f"(delta_r={r['delta_r_observed']:+.4f}, paired bootstrap 95% CI "
            f"[{r['delta_r_ci_lo']:+.4f}, {r['delta_r_ci_hi']:+.4f}]).",
            ""]
    (PR / "PAPER_READY_RESULTS.md").write_text("\n".join(lines) + "\n")
    # tables
    t1 = model_df.pivot_table(index=["target", "model"], columns="regime",
                              values=["r_mean", "rmse_mean", "mae_mean"])
    t1.to_csv(PR / "table1_primary_paired.csv")
    d_rows = []
    for target in TASKS:
        for model in ALL_MODELS:
            d = seed_deltas_primary(seed_df, target, model)
            brow = boot_df[(boot_df.target == target) & (boot_df.model == model)].iloc[0]
            d_rows.append({"target": target, "model": model,
                           "r_412": brow["r_412"], "r_plus98": brow["r_plus98"],
                           "delta_r": d.mean(), "ci_lo": brow["delta_r_ci_lo"],
                           "ci_hi": brow["delta_r_ci_hi"], "positive_seeds": int((d > 0).sum()),
                           "tier": tier(d.mean(), int((d > 0).sum()), brow["delta_r_ci_lo"])})
    pd.DataFrame(d_rows).to_csv(PR / "table1_primary_paired_deltas.csv", index=False)
    me.to_csv(PR / "table2_method_effects.csv", index=False)
    inter.to_csv(PR / "table3_interactions.csv", index=False)
    stab.to_csv(PR / "table5_biomarker_stability.csv", index=False)
    faith.to_csv(PR / "table6_cross_fold_faithfulness.csv", index=False)
    if learn is not None:
        learn.to_csv(PR / "tableS3_learning_curve.csv", index=False)
    # captions/claims/limitations
    (PR / "PAPER_READY_CAPTIONS.md").write_text(
        "# Figure/table captions (exploratory scaling)\n\n"
        "Figure 1: Paired sample-size scaling design. Identical V412 outer folds; "
        "Regime A trains on T412, Regime B trains on T412+D98.\n\n"
        "Figure 2: Paired 412 vs +98 prediction for R0, PALF, and PS-NCR-EF; "
        "lines connect the same seed.\n\n"
        "Figure 3: Method effects under both training regimes.\n\n"
        "Figure 4: Learning curve on fixed V412.\n\n"
        "Figure 5: Biomarker stability 412 vs +98.\n\n"
        "Figure 6: Cross-fold faithfulness 412 vs +98 (positive = faithful).\n")
    (PR / "PAPER_READY_CLAIMS.md").write_text(
        "# Claim-safe wording\n\n"
        "- The 510 analysis is exploratory/post-hoc; NOT independent validation.\n"
        "- Use 'numerically higher' or 'nominal exploratory improvement' for "
        "positive deltas whose CI includes 0.\n"
        "- Use 'consistent improvement across development splits' only when "
        "delta_r >= .005 and >=4/5 seeds positive.\n"
        "- Use 'robust paired improvement in the exploratory scaling analysis' "
        "only when the bootstrap CI lower bound > 0.\n"
        "- Never claim causality or that the previous negative holdout was "
        "caused by sample size.\n")
    (PR / "PAPER_READY_LIMITATIONS.md").write_text(
        "# Limitations\n\n"
        "- Post-hoc combined-cohort analysis; the 98 subjects were previously "
        "evaluated in Phase 4 and are no longer a holdout.\n"
        "- Evaluation subjects are the same D412 subjects under both regimes; "
        "this improves pairing but not external validity.\n"
        "- Nominal gains without CI support must not be described as significant.\n")
    # LaTeX
    tex = ["% Auto-generated: exploratory sample-size scaling results.",
           "\\begin{table}[t]", "\\centering",
           "\\caption{Paired 412 vs +98 prediction (exploratory).}",
           "\\label{tab:scaling_primary}",
           "\\begin{tabular}{llrrrr}",
           "\\toprule",
           "Target & Model & $r_{412}$ & $r_{+98}$ & $\\Delta r$ & Tier \\\\",
           "\\midrule"]
    for r in d_rows:
        tex.append(f"{r['target']} & {r['model']} & {r['r_412']:.3f} & "
                   f"{r['r_plus98']:.3f} & {r['delta_r']:+.3f} & {r['tier']} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (PR / "PAPER_READY_TABLES.tex").write_text("\n".join(tex))
    (PR / "PAPER_READY_RESULTS.tex").write_text(
        "\\section{Exploratory sample-size scaling on the combined cohort}\n"
        "This analysis is post-hoc and exploratory, not independent validation.\n"
        "See Table~\\ref{tab:scaling_primary} and Figure~1.\n")
    # main-paper section
    (PR / "section_sample_size_scaling.tex").write_text(
        "\\subsection{Does additional training data improve prediction?}\n"
        "Under the paired training-expansion protocol, expanding the training pool "
        "from the 412-subject development cohort to the combined 510-subject pool "
        "changed predictive performance on the identical evaluation subjects "
        "(Table~\\ref{tab:scaling_primary}). We label this analysis exploratory and "
        "post-hoc; it does not constitute independent validation.\n\n"
        "\\subsection{Does the prior-aware advantage scale with sample size?}\n"
        "Method effects (PALF-R0, NCR-Ridge, NCR-R0) are reported separately under "
        "each training regime (Table~\\ref{tab:scaling_primary}).\n\n"
        "\\subsection{Biomarker stability under training-set expansion}\n"
        "Stability metrics for the matched expert are reported in "
        "Figure~5 and the supplementary material.\n\n"
        "\\subsection{Cross-fold biomarker faithfulness}\n"
        "Cross-fold faithfulness contrasts (top10 minus random10 $\\Delta$RMSE, "
        "positive = faithful) are shown in Figure~6.\n")
    # supplementary tex
    SUP = OUT / "supplementary"
    (SUP / "SUPPLEMENTARY_METHODS.tex").write_text(
        "\\section{Scaling study methods}\n"
        "Outer folds over D412 with seeds [7171,7272,7373,7474,7575], five folds; "
        "three inner folds; identical V412 for both regimes.\n")
    (SUP / "SUPPLEMENTARY_ABLATIONS.tex").write_text(
        "\\section{Scaling ablations}\nSee ablation_results.csv.\n")
    (SUP / "SUPPLEMENTARY_RESULTS.tex").write_text(
        "\\section{Scaling results}\n"
        "All 412 vs +98 comparisons are reported, including negative ones.\n")
    (SUP / "SUPPLEMENTARY_CAPTIONS.md").write_text(
        "# Supplementary figure captions\n\n"
        "S1: Nested-CV data flow. S2: PALF ablation. S3: PS-NCR-EF architecture. "
        "S4: Biomarker faithfulness. S5: Learning-curve schematic. S6: Prior controls.\n")
    return True


def main():
    t0 = time.time()
    print("=" * 70); print("SCALING STUDY REPORT"); print("=" * 70)
    X_fc, X_sc, y_wm, y_fi, ids, recs = load_all()
    seed_df = primary_tables(recs, y_wm, y_fi)
    seed_df.to_csv(OUT / "primary_seed_metrics.csv", index=False)
    model_df = model_metrics(seed_df)
    model_df.to_csv(OUT / "primary_model_metrics.csv", index=False)
    boot = pd.DataFrame([paired_bootstrap(recs, t, m, y)
                         for t, y in (("WM", y_wm), ("FI", y_fi))
                         for m in ALL_MODELS])
    boot.to_csv(OUT / "paired_bootstrap_results.csv", index=False)
    me = method_effects(seed_df)
    me.to_csv(OUT / "method_effects.csv", index=False)
    inter = interactions(seed_df)
    inter.to_csv(OUT / "sample_size_interactions.csv", index=False)
    learn = learning_table()
    if learn is not None:
        learn.to_csv(OUT / "learning_curve.csv", index=False)
    stab = pd.read_csv(OUT / "biomarker_stability.csv")
    faith = pd.read_csv(OUT / "biomarker_faithfulness.csv")
    abl, famdf = ablation_tables(recs, seed_df, X_fc, X_sc, y_wm, y_fi)
    abl.to_csv(OUT / "ablation_results.csv", index=False)
    # subject predictions
    srows = []
    for target, y in (("WM", y_wm), ("FI", y_fi)):
        for model in ALL_MODELS:
            for regime in ("A", "B"):
                for (seed, fold), rec in recs.items():
                    te = rec[regime]["te"]
                    p = rec[regime]["targets"][target]["preds"][model]
                    for k, i in enumerate(te):
                        srows.append((target, model, regime, seed, fold, ids[i],
                                      float(y[i]), float(p[k])))
    pd.DataFrame(srows, columns=["target", "model", "regime", "seed", "fold",
                                 "subject", "y", "pred"]).to_csv(
        OUT / "primary_subject_predictions.csv", index=False)
    trigger = paper_ready(seed_df, model_df, boot, me, inter, stab, faith, learn)
    make_plots(seed_df, model_df, boot, stab, faith, learn, X_fc, X_sc, recs)
    # ablation markdown
    with open(OUT / "ABLATION_RESULTS.md", "w") as f:
        f.write("# Scaling ablations\n\n")
        for g, sub in abl.groupby("group"):
            f.write(f"## {g}\n\n```\n{sub.to_string(index=False)}\n```\n\n")
        if learn is not None:
            f.write("## H_learning_curve\n\n```\n" + learn.to_string(index=False) + "\n```\n")
    # runtime final
    prog = json.loads((OUT / "RUNTIME_PROGRESS.json").read_text()) if \
        (OUT / "RUNTIME_PROGRESS.json").exists() else {}
    (OUT / "RUNTIME_FINAL.json").write_text(json.dumps(
        {"stages": prog, "report_runtime_seconds": time.time() - t0,
         "paper_ready_trigger": trigger}, indent=2))
    (OUT / "COMPLETE").write_text("SCALING_412_VS_510_COMPLETE\n")
    print(f"  report done in {time.time()-t0:.0f}s; trigger={trigger}")


if __name__ == "__main__":
    main()
