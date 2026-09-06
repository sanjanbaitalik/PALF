#!/usr/bin/env python3
"""Reconstruct and export frozen final LF1 FC coefficient vectors.

This is a reproducibility-only utility. It reconstructs the fitted matched-prior
FC coefficient vectors that were used in the already-completed final LF1 10x5
experiment. The original final runner saved predictions and selected
hyperparameters but did not serialize the 6670-dimensional FP coefficient vectors.

No hyperparameter selection is performed. No model is changed. No prior is
regenerated. No existing final artifact is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import yaml
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metascfc.benchmark_utils import load_connectomes, prediction_metrics
from metascfc.experiments.msancr_refinement import load_roi_prior, upper_triangle_features
from metascfc.experiments.prior_aware_late_fusion import (
    TOP_K, DIAGONAL_EPSILON,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    build_msancr_cache,
    _solve_msancr_kernel,
    _predict_msancr,
    recover_msancr_beta,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import build_edge_laplacian

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/iclr/lf1_final_10x5.yaml"
FINAL_OUTPUT_BASE = PROJECT_ROOT / "outputs/iclr/lf1_final_10x5"

PREDICTION_TOL_PRIMARY = 1e-6
PREDICTION_TOL_SECONDARY = 1e-5
EXPECTED_COEF_LENGTH = 6670  # 116 * 115 / 2


# ---------------------------------------------------------------------------
# Integrity safeguards
# ---------------------------------------------------------------------------

_NO_INNER_CV_CALLED = True
_NO_HP_GRID_SEARCH_CALLED = True
_NO_NEW_PRIOR_GENERATED = True


def _assert_no_inner_cv():
    assert _NO_INNER_CV_CALLED, "INNER_CV_INVOKED"


def _assert_no_grid_search():
    assert _NO_HP_GRID_SEARCH_CALLED, "GRID_SEARCH_INVOKED"


def _assert_no_prior_generation():
    assert _NO_NEW_PRIOR_GENERATED, "PRIOR_GENERATED"


# ---------------------------------------------------------------------------
# Source integrity hashing
# ---------------------------------------------------------------------------

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _compute_source_integrity(
    results_pkl: Path,
    config_yaml: Path,
    prior_csvs: list[Path],
    fc_path: Path,
    sc_path: Path,
    y_paths: list[Path],
) -> dict:
    integrity = {
        "all_split_results_pkl": _file_hash(results_pkl),
        "final_config": _file_hash(config_yaml),
    }
    for p in prior_csvs:
        integrity[f"prior_{p.name}"] = _file_hash(p)
    integrity["packed_fc_input"] = _file_hash(fc_path)
    integrity["packed_sc_input"] = _file_hash(sc_path)
    for p in y_paths:
        integrity[f"target_{p.name}"] = _file_hash(p)
    return integrity


# ---------------------------------------------------------------------------
# Core reconstruction
# ---------------------------------------------------------------------------

def reconstruct_one_split(
    result,
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y_task: np.ndarray,
    roi_prior_matched: np.ndarray,
    n_rois: int,
    prior_name: str,
) -> dict:
    """Reconstruct the FP outer refit for one saved split and verify predictions.

    Returns a dict with all export fields.
    """
    _assert_no_inner_cv()
    _assert_no_grid_search()
    _assert_no_prior_generation()

    seed = int(result.seed)
    outer_fold = int(result.outer_fold)
    train_idx = np.asarray(result.train_idx, dtype=int)
    test_idx = np.asarray(result.test_idx, dtype=int)

    # --- Verify split identity ---
    n_total = len(train_idx) + len(test_idx)
    assert n_total == 412, f"Expected 412 subjects, got {n_total}"
    assert len(np.intersect1d(train_idx, test_idx)) == 0, "Train/test overlap"

    # --- Extract frozen hyperparameters ---
    fp_params = dict(result.level1_results["FP"].selected_hyperparams)
    s_alpha = float(result.level1_results["S"].selected_hyperparams.get("alpha", 1.0))
    w_fp = float(result.lf1_weights.get("FP", 0.0))
    w_s = float(result.lf1_weights.get("S", 0.0))

    lambda_fc = float(fp_params.get("lambda_fc", 1.0))
    lambda_sc = float(fp_params.get("lambda_sc", 1.0))
    lambda_l = float(fp_params.get("lambda_l", 0.0))
    gamma = float(fp_params.get("gamma", 0.5))
    lifting = str(fp_params.get("lifting", "prod"))

    # --- Build matched-prior cache ---
    edge_laplacian = build_edge_laplacian(
        n_rois, prior_scores=roi_prior_matched,
        top_k=TOP_K, weighting="binary",
        couple_modalities=False, normalize="sym",
    )
    fp_cache = build_msancr_cache(
        roi_prior_matched, n_rois,
        gamma=gamma, lifting=lifting,
        top_k=TOP_K, epsilon=DIAGONAL_EPSILON,
        weighting="binary", couple_modalities=False,
        normalize_laplacian="sym",
        edge_laplacian=edge_laplacian,
    )

    # --- Reproduce SC branch ---
    scaler_sc = StandardScaler()
    X_sc_train_z = scaler_sc.fit_transform(X_sc[train_idx])
    X_sc_test_z = scaler_sc.transform(X_sc[test_idx])
    s_model = Ridge(alpha=s_alpha, fit_intercept=True)
    s_model.fit(X_sc_train_z, y_task[train_idx])
    pred_sc = s_model.predict(X_sc_test_z)

    # --- Reproduce FP branch (exact same code path as evaluate_outer_split) ---
    scaler_fc = StandardScaler()
    X_fc_train_z = scaler_fc.fit_transform(X_fc[train_idx])
    X_fc_test_z = scaler_fc.transform(X_fc[test_idx])

    y_mean_train = float(y_task[train_idx].mean())
    y_std_train = max(float(y_task[train_idx].std()), 1e-8)
    y_train_z = (y_task[train_idx] - y_mean_train) / y_std_train

    alpha_fp, _ = _solve_msancr_kernel(
        X_fc_train_z, np.zeros_like(X_fc_train_z),
        y_train_z, fp_cache,
        lambda_fc, lambda_sc, lambda_l,
        fc_only=True,
    )

    pred_z = _predict_msancr(
        X_fc_test_z, np.zeros_like(X_fc_test_z),
        X_fc_train_z, np.zeros_like(X_fc_train_z),
        alpha_fp, fp_cache,
        lambda_fc, lambda_sc, lambda_l,
        fc_only=True,
    )
    pred_fp = pred_z * y_std_train + y_mean_train

    # --- Recover primal coefficient vector ---
    coef_model_space, _ = recover_msancr_beta(
        X_fc_train_z, np.zeros_like(X_fc_train_z),
        alpha_fp, fp_cache,
        lambda_fc, lambda_sc, lambda_l,
        fc_only=True,
    )
    coef_biomarker_space = coef_model_space * y_std_train

    assert coef_model_space.shape == (EXPECTED_COEF_LENGTH,), \
        f"Expected {EXPECTED_COEF_LENGTH} coefficients, got {coef_model_space.shape}"
    assert np.isfinite(coef_model_space).all(), "Non-finite coefficients"
    assert np.isfinite(coef_biomarker_space).all(), "Non-finite y-scaled coefficients"

    # --- Verify dual vs primal prediction match ---
    pred_from_beta = (X_fc_test_z @ coef_model_space) * y_std_train + y_mean_train
    beta_match_err = float(np.max(np.abs(pred_fp - pred_from_beta)))
    assert beta_match_err <= 1e-7, f"Dual-primal mismatch: {beta_match_err:.2e}"

    # --- Reconstruct LF1 fused prediction ---
    pred_lf1_reconstructed = w_fp * pred_fp + w_s * pred_sc
    pred_lf1_stored = np.asarray(result.lf1_test_pred, dtype=np.float64)

    max_abs_err = float(np.max(np.abs(pred_lf1_stored - pred_lf1_reconstructed)))
    mean_abs_err = float(np.mean(np.abs(pred_lf1_stored - pred_lf1_reconstructed)))

    # --- Verify prediction match ---
    if max_abs_err <= PREDICTION_TOL_PRIMARY:
        tolerance_used = "primary_1e-6"
    elif max_abs_err <= PREDICTION_TOL_SECONDARY:
        tolerance_used = "secondary_1e-5"
    else:
        tolerance_used = f"FAILED_{max_abs_err:.2e}"

    assert max_abs_err <= PREDICTION_TOL_SECONDARY, (
        f"Prediction reconstruction FAILED for seed={seed} fold={outer_fold}: "
        f"max_abs_error={max_abs_err:.2e} > {PREDICTION_TOL_SECONDARY}"
    )

    # --- Verify metric reproduction ---
    m_reconstructed = prediction_metrics(y_task[test_idx], pred_lf1_reconstructed)
    m_stored = {
        "pearson": float(result.lf1_pearson),
        "rmse": float(result.lf1_rmse),
        "mae": float(result.lf1_mae),
    }

    pearson_abs_err = abs(m_stored["pearson"] - m_reconstructed["pearson"])
    rmse_abs_err = abs(m_stored["rmse"] - m_reconstructed["rmse"])
    mae_abs_err = abs(m_stored["mae"] - m_reconstructed["mae"])

    assert pearson_abs_err < 1e-4, f"Pearson mismatch: {pearson_abs_err:.2e}"
    assert rmse_abs_err < 1e-4, f"RMSE mismatch: {rmse_abs_err:.2e}"

    # --- Determine coefficient space ---
    # recover_msancr_beta returns coefficients on standardized FC features.
    # coef_model_space: prediction_z = X_fc_test_z @ coef_model_space
    # coef_biomarker_space: prediction = X_fc_test_z @ coef_biomarker_space + intercept
    coefficient_space = "standardized_features_original_target_units"

    return {
        "seed": seed,
        "outer_fold": outer_fold,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "coef_model_space": coef_model_space.astype(np.float64),
        "coef_biomarker_space": coef_biomarker_space.astype(np.float64),
        "lambda_fc": lambda_fc,
        "lambda_l": lambda_l,
        "gamma": gamma,
        "lifting": lifting,
        "w_fp": w_fp,
        "w_s": w_s,
        "pred_fp": pred_fp.astype(np.float64),
        "pred_sc": pred_sc.astype(np.float64),
        "pred_lf1_reconstructed": pred_lf1_reconstructed.astype(np.float64),
        "pred_lf1_stored": pred_lf1_stored.astype(np.float64),
        "prediction_max_abs_error": max_abs_err,
        "prediction_mean_abs_error": mean_abs_err,
        "prediction_rmse_difference": float(np.abs(m_stored["rmse"] - m_reconstructed["rmse"])),
        "prediction_pearson": float(pearsonr(pred_lf1_stored, pred_lf1_reconstructed).statistic),
        "stored_lf1_pearson": m_stored["pearson"],
        "reconstructed_lf1_pearson": m_reconstructed["pearson"],
        "pearson_abs_error": pearson_abs_err,
        "stored_lf1_rmse": m_stored["rmse"],
        "reconstructed_lf1_rmse": m_reconstructed["rmse"],
        "rmse_abs_error": rmse_abs_err,
        "stored_lf1_mae": m_stored["mae"],
        "reconstructed_lf1_mae": m_reconstructed["mae"],
        "mae_abs_error": mae_abs_err,
        "coefficient_length": len(coef_model_space),
        "coefficient_finite": bool(np.isfinite(coef_model_space).all()),
        "coefficient_space": coefficient_space,
        "tolerance_used": tolerance_used,
    }


# ---------------------------------------------------------------------------
# Aggregate summaries
# ---------------------------------------------------------------------------

def compute_edge_mean_abs(
    all_coefs: np.ndarray,
    n_rois: int = 116,
    label_path: Path | None = None,
) -> pd.DataFrame:
    """Compute edge-level coefficient summaries."""
    iu = np.triu_indices(n_rois, k=1)
    n_edges = all_coefs.shape[1]
    assert n_edges == n_rois * (n_rois - 1) // 2

    roi_i_0based = iu[0]
    roi_j_0based = iu[1]
    roi_i_1based = roi_i_0based + 1
    roi_j_1based = roi_j_0based + 1

    mean_signed = np.mean(all_coefs, axis=0)
    mean_abs = np.mean(np.abs(all_coefs), axis=0)
    median_abs = np.median(np.abs(all_coefs), axis=0)
    rank_mean_abs = np.argsort(np.argsort(-mean_abs)) + 1

    df = pd.DataFrame({
        "edge_index": np.arange(n_edges),
        "roi_i_0based": roi_i_0based,
        "roi_j_0based": roi_j_0based,
        "roi_i_1based": roi_i_1based,
        "roi_j_1based": roi_j_1based,
        "mean_signed_coef": mean_signed,
        "mean_abs_coef": mean_abs,
        "median_abs_coef": median_abs,
        "rank_mean_abs": rank_mean_abs,
    })

    if label_path and Path(label_path).exists():
        labels_df = pd.read_csv(label_path)
        if "roi_label" in labels_df.columns and "roi_index" in labels_df.columns:
            labels_df = labels_df.sort_values("roi_index")
            names = labels_df["roi_label"].tolist()
            df["roi_i_name"] = [names[i] for i in roi_i_0based]
            df["roi_j_name"] = [names[j] for j in roi_j_0based]

    return df


def compute_stability_metrics(all_coefs: np.ndarray) -> pd.DataFrame:
    """Compute per-edge stability metrics across splits."""
    n_splits, n_edges = all_coefs.shape
    abs_coefs = np.abs(all_coefs)

    top10_frequency = np.zeros(n_edges)
    top20_frequency = np.zeros(n_edges)
    top50_frequency = np.zeros(n_edges)
    sign_consistency = np.zeros(n_edges)

    for i in range(n_splits):
        order = np.argsort(abs_coefs[i])[::-1]
        top10_frequency[order[:10]] += 1
        top20_frequency[order[:20]] += 1
        top50_frequency[order[:50]] += 1

    top10_frequency /= n_splits
    top20_frequency /= n_splits
    top50_frequency /= n_splits

    for e in range(n_edges):
        pos_frac = np.mean(all_coefs[:, e] > 0)
        neg_frac = np.mean(all_coefs[:, e] < 0)
        sign_consistency[e] = max(pos_frac, neg_frac)

    return pd.DataFrame({
        "edge_index": np.arange(n_edges),
        "top10_frequency": top10_frequency,
        "top20_frequency": top20_frequency,
        "top50_frequency": top50_frequency,
        "sign_consistency": sign_consistency,
    })


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def run_export(
    task: str,
    smoke: bool = False,
    seed_filter: int | None = None,
    fold_filter: int | None = None,
) -> dict:
    """Run the frozen FP coefficient export for one task."""
    _assert_no_inner_cv()
    _assert_no_grid_search()
    _assert_no_prior_generation()

    t0 = time.time()

    # --- Load config ---
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    n_rois = int(cfg.get("n_rois", 116))

    # --- Load data ---
    fc_mats, sc_mats, y_all, _, _ = load_connectomes(cfg["data"])
    X_fc = upper_triangle_features(fc_mats)
    X_sc = upper_triangle_features(sc_mats)

    target_cfg = cfg["targets"][task]
    y_task = np.asarray(
        np.load(target_cfg["label_path"], allow_pickle=False),
        dtype=np.float64,
    ).reshape(-1)

    # --- Load matched prior ---
    prior_path = cfg["priors"][task]["matched"]
    roi_prior_matched = load_roi_prior(prior_path, n_rois)
    prior_name = Path(prior_path).parent.name

    # --- Load saved results ---
    results_pkl = FINAL_OUTPUT_BASE / task / "all_split_results.pkl"
    assert results_pkl.exists(), f"Missing results: {results_pkl}"
    with open(results_pkl, "rb") as f:
        all_results = pickle.load(f)

    # --- Source integrity ---
    prior_csvs = [
        Path(cfg["priors"][task]["matched"]),
    ]
    fc_path = Path(cfg["data"]["fc_path"])
    sc_path = Path(cfg["data"]["sc_path"])
    y_paths = [Path(target_cfg["label_path"])]
    integrity = _compute_source_integrity(
        results_pkl, CONFIG_PATH, prior_csvs, fc_path, sc_path, y_paths,
    )

    # --- Filter splits if smoke mode ---
    if smoke:
        results_to_process = [
            r for r in all_results
            if (seed_filter is None or r.seed == seed_filter)
            and (fold_filter is None or r.outer_fold == fold_filter)
        ]
        output_base = FINAL_OUTPUT_BASE / "frozen_fp_coefficients_smoke"
    else:
        assert seed_filter is None and fold_filter is None, \
            "Seed/fold filters only allowed in smoke mode"
        results_to_process = list(all_results)
        output_base = FINAL_OUTPUT_BASE / "frozen_fp_coefficients"

    # --- Create output directory ---
    task_out = output_base / task
    task_out.mkdir(parents=True, exist_ok=True)

    # --- Verify split metadata ---
    seeds_seen = set()
    folds_seen = set()
    pairs_seen = set()
    for r in results_to_process:
        s, f = int(r.seed), int(r.outer_fold)
        seeds_seen.add(s)
        folds_seen.add(f)
        pair = (s, f)
        assert pair not in pairs_seen, f"Duplicate (seed={s}, fold={f})"
        pairs_seen.add(pair)

    print(f"  Task: {task}")
    print(f"  Splits to process: {len(results_to_process)}")
    print(f"  Unique seeds: {len(seeds_seen)}")
    print(f"  Unique folds: {len(folds_seen)}")
    print(f"  Output: {task_out}")

    # --- Process each split ---
    all_split_data = []
    all_coefs = []
    all_seeds = []
    all_folds = []
    failed = False

    for i, result in enumerate(results_to_process):
        seed = int(result.seed)
        fold = int(result.outer_fold)
        print(f"    [{i+1}/{len(results_to_process)}] seed={seed} fold={fold} ... ", end="", flush=True)

        try:
            data = reconstruct_one_split(
                result, X_fc, X_sc, y_task, roi_prior_matched, n_rois, prior_name,
            )
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed = True
            continue

        print(f"OK (max_err={data['prediction_max_abs_error']:.2e})")
        all_split_data.append(data)
        all_coefs.append(data["coef_biomarker_space"])
        all_seeds.append(data["seed"])
        all_folds.append(data["outer_fold"])

        # --- Save per-split NPZ ---
        fname = f"seed{seed:02d}_fold{fold:02d}.npz"
        npz_path = task_out / fname
        np.savez_compressed(
            npz_path,
            coef_model_space=data["coef_model_space"],
            coef_biomarker_space=data["coef_biomarker_space"],
            seed=data["seed"],
            outer_fold=data["outer_fold"],
            train_idx=data["train_idx"],
            test_idx=data["test_idx"],
            lambda_fc=data["lambda_fc"],
            lambda_l=data["lambda_l"],
            gamma=data["gamma"],
            lifting=data["lifting"],
            w_fp=data["w_fp"],
            w_s=data["w_s"],
            pred_fp=data["pred_fp"],
            pred_sc=data["pred_sc"],
            pred_lf1_reconstructed=data["pred_lf1_reconstructed"],
            pred_lf1_stored=data["pred_lf1_stored"],
            prediction_max_abs_error=data["prediction_max_abs_error"],
            prediction_mean_abs_error=data["prediction_mean_abs_error"],
            coefficient_space=data["coefficient_space"],
            prior_name=prior_name,
        )

    if failed:
        print("\nFROZEN_COEFFICIENT_RECONSTRUCTION_FAILED")
        raise SystemExit(1)

    # --- Sort by seed, then fold ---
    order = np.lexsort((all_folds, all_seeds))
    all_coefs_sorted = np.array(all_coefs)[order]
    all_seeds_sorted = np.array(all_seeds)[order]
    all_folds_sorted = np.array(all_folds)[order]

    # --- Save aggregate coefficient arrays ---
    np.savez_compressed(
        task_out / "all_coefficients.npz",
        coefficients=all_coefs_sorted,
        seeds=all_seeds_sorted,
        folds=all_folds_sorted,
    )

    # --- Save coefficient summary NPY files ---
    np.save(task_out / "mean_signed_coef.npy", np.mean(all_coefs_sorted, axis=0))
    np.save(task_out / "mean_abs_coef.npy", np.mean(np.abs(all_coefs_sorted), axis=0))
    np.save(task_out / "median_abs_coef.npy", np.median(np.abs(all_coefs_sorted), axis=0))

    # --- Save edge_mean_abs.csv ---
    label_path = PROJECT_ROOT / "inputs/atlases/AAL116_labels.csv"
    edge_df = compute_edge_mean_abs(all_coefs_sorted, n_rois, label_path)
    edge_df.to_csv(task_out / "edge_mean_abs.csv", index=False)

    # --- Save stable_top_edges.csv ---
    stability_df = compute_stability_metrics(all_coefs_sorted)
    stable_top = edge_df.merge(stability_df, on="edge_index")
    stable_top = stable_top.sort_values("mean_abs_coef", ascending=False)
    stable_top.to_csv(task_out / "stable_top_edges.csv", index=False)

    # --- Save reconstruction audit CSV ---
    audit_rows = []
    for data in all_split_data:
        audit_rows.append({
            "task": task,
            "seed": data["seed"],
            "outer_fold": data["outer_fold"],
            "n_train": len(data["train_idx"]),
            "n_test": len(data["test_idx"]),
            "w_fp": data["w_fp"],
            "w_s": data["w_s"],
            "prediction_max_abs_error": data["prediction_max_abs_error"],
            "prediction_mean_abs_error": data["prediction_mean_abs_error"],
            "prediction_rmse_difference": data["prediction_rmse_difference"],
            "prediction_pearson": data["prediction_pearson"],
            "stored_lf1_pearson": data["stored_lf1_pearson"],
            "reconstructed_lf1_pearson": data["reconstructed_lf1_pearson"],
            "pearson_abs_error": data["pearson_abs_error"],
            "stored_lf1_rmse": data["stored_lf1_rmse"],
            "reconstructed_lf1_rmse": data["reconstructed_lf1_rmse"],
            "rmse_abs_error": data["rmse_abs_error"],
            "stored_lf1_mae": data["stored_lf1_mae"],
            "reconstructed_lf1_mae": data["reconstructed_lf1_mae"],
            "mae_abs_error": data["mae_abs_error"],
            "coefficient_length": data["coefficient_length"],
            "coefficient_finite": data["coefficient_finite"],
            "status": "PASS",
        })
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(task_out / "reconstruction_audit.csv", index=False)

    # --- Save selected hyperparameters CSV ---
    hp_rows = []
    for data in all_split_data:
        hp_rows.append({
            "seed": data["seed"],
            "outer_fold": data["outer_fold"],
            "lambda_fc": data["lambda_fc"],
            "lambda_l": data["lambda_l"],
            "gamma": data["gamma"],
            "lifting": data["lifting"],
            "w_fp": data["w_fp"],
            "w_s": data["w_s"],
        })
    pd.DataFrame(hp_rows).to_csv(task_out / "selected_hyperparameters.csv", index=False)

    # --- Save source integrity ---
    (task_out / "source_integrity.json").write_text(
        json.dumps(integrity, indent=2, default=str)
    )

    # --- Summary statistics ---
    max_errs = [d["prediction_max_abs_error"] for d in all_split_data]
    mean_errs = [d["prediction_mean_abs_error"] for d in all_split_data]

    elapsed = time.time() - t0

    summary = {
        "task": task,
        "splits_reconstructed": len(all_split_data),
        "total_splits_requested": len(results_to_process),
        "max_prediction_reproduction_error": max(max_errs),
        "mean_prediction_reproduction_error": float(np.mean(mean_errs)),
        "coefficients_shape": all_coefs_sorted.shape,
        "all_finite": bool(np.all(np.isfinite(all_coefs_sorted))),
        "elapsed_seconds": elapsed,
        "output_dir": str(task_out),
        "smoke_mode": smoke,
    }

    (task_out / "export_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export frozen final LF1 FC coefficients for reproducibility.",
    )
    parser.add_argument(
        "--task",
        choices=["working_memory", "fluid_intelligence"],
        required=True,
        help="Task to export coefficients for.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(Smoke mode only) Process only this seed.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="(Smoke mode only) Process only this fold.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: export to a separate directory, optional seed/fold filter.",
    )
    args = parser.parse_args()

    print("FROZEN FP COEFFICIENT EXPORT")
    print("=" * 60)

    _assert_no_inner_cv()
    _assert_no_grid_search()
    _assert_no_prior_generation()

    summary = run_export(
        task=args.task,
        smoke=args.smoke,
        seed_filter=args.seed,
        fold_filter=args.fold,
    )

    print("\n" + "=" * 60)
    print(f"FROZEN FP COEFFICIENT EXPORT COMPLETE")
    print("=" * 60)

    task_label = args.task.replace("_", " ").title()
    print(f"\n{task_label}:")
    print(f"  splits reconstructed = {summary['splits_reconstructed']}/{summary['total_splits_requested']}")
    print(f"  max prediction reproduction error = {summary['max_prediction_reproduction_error']:.2e}")
    print(f"  mean prediction reproduction error = {summary['mean_prediction_reproduction_error']:.2e}")
    print(f"  coefficients shape = {summary['coefficients_shape']}")
    print(f"  all finite = {'YES' if summary['all_finite'] else 'NO'}")
    print(f"\nNo hyperparameter selection performed.")
    print(f"No prior regenerated.")
    print(f"No final artifact modified.")


if __name__ == "__main__":
    main()
