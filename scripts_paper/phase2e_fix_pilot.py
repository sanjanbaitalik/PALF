"""Phase 2E-FIX: Genuine LLM Interaction Prior + Correct Nested SFC Evaluation.

Fixes:
- F1: Genuine LLM priors via REST API (not ROI-derived)
- F2: Provenance matches actual generation
- F3: Ollama REST API with proper seed/temperature
- F4: Correct Yeo7 labels (DAN/VAN/LIMBIC/FPN/DMN)
- F5: A2 uses LLM-selected pairs (top_k ∈ {5,10,15,25})
- F6: A3-A6 search TOP_PAIR_GRID
- F7: SFC features computed per nested fold (no leakage)
- F8: Task-specific control priors
"""
from __future__ import annotations

import json
import hashlib
import logging
import pickle
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

def _build_pair_system_names(n_pairs: int = 45) -> np.ndarray:
    """Build system name array for each pair index."""
    pairs = []
    for i, s1 in enumerate(SYSTEM_NAMES):
        for j, s2 in enumerate(SYSTEM_NAMES):
            if j >= i:
                pairs.append(f"{s1}-{s2}" if s1 != s2 else s1)
    return np.array(pairs[:n_pairs])

from metascfc.benchmark_utils import prediction_metrics
from metascfc.experiments.palf_crossfit_ablation import (
    CONDITIONS,
    generate_crossfit_oof,
    make_fusion_folds,
    make_inner_selection_folds,
    make_outer_splits,
    reselect_and_fit_final,
    search_fusion_weights,
    N_ROI,
    N_EDGE,
    RIDGE_GRID,
)
from metascfc.phase2e_fix.aal_yeo_mapping_fix import (
    build_aal116_yeo7_mapping,
    build_system_pair_list,
    roi_to_system_array,
    edge_to_system_pair,
    save_mapping,
    SYSTEM_NAMES,
    N_SYSTEMS,
)
from metascfc.phase2e_fix.sfc_features_fix import build_sfc_features_from_raw
from metascfc.phase2e_fix.li_sfc_ncr_fix import (
    build_pair_laplacian,
    ridge_expert_predict,
    ncr_expert_predict,
    hierarchical_fusion_weights,
)

# ── Constants ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs/iclr/palf_phase2e_fix_li_sfc_ncr")
PLOTS_DIR = OUTPUT_DIR / "plots"
COEFF_DIR = OUTPUT_DIR / "coefficients"
PRED_DIR = OUTPUT_DIR / "predictions"
PRIORS_DIR = OUTPUT_DIR / "priors"

PHASE2E_FIX_DEV_SEEDS = [3131, 3232, 3333, 3434]
N_OUTER_FOLDS = 5
N_FUSION_FOLDS = 3
N_INNER = 3
N_FINAL_CV = 5
TOP_PAIR_GRID = [5, 10, 15, 25]
SFC_ALPHA_GRID = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
NCR_RATIO_GRID = np.array([0.0, 0.1, 0.3, 1.0])
ETA_GRID = np.arange(0, 1.05, 0.05)

log = logging.getLogger("phase2e_fix")


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    """Load HCP data arrays."""
    data_dir = REPO_ROOT / "inputs"
    fc_mats = np.load(data_dir / "dataset_FC/FC_all.npy")
    sc_mats = np.load(data_dir / "dataset_SC/SC_all.npy")
    iu = np.triu_indices(N_ROI, k=1)
    X_fc = fc_mats[:, iu[0], iu[1]].astype(np.float64)
    X_sc = sc_mats[:, iu[0], iu[1]].astype(np.float64)
    y_wm = np.load(
        data_dir / "dataset_SC/task_labels/ListSort_Unadj/label_all.npy"
    ).astype(np.float64)
    y_fi = np.load(
        data_dir / "dataset_SC/label_all.npy"
    ).astype(np.float64)
    return X_fc, X_sc, y_wm, y_fi


def load_roi_prior():
    """Load ROI-level prior for baseline."""
    roi_prior_path = REPO_ROOT / "outputs" / "priors" / "llm" / "working_memory_contrastive_qwen3" / "roi_prior.csv"
    df = pd.read_csv(roi_prior_path)
    return df["prior_score"].values.astype(np.float64)


# ── Edge-to-pair mapping ──────────────────────────────────────────────────────
_EDGE_PAIR_CACHE = None

def get_edge_pair_indices():
    """Get or compute edge-to-pair mapping."""
    global _EDGE_PAIR_CACHE
    if _EDGE_PAIR_CACHE is not None:
        return _EDGE_PAIR_CACHE

    mapping_df, _ = build_aal116_yeo7_mapping(
        str(REPO_ROOT / "inputs" / "atlases" / "AAL116.nii.gz"),
        str(REPO_ROOT / "inputs" / "atlases" / "AAL116_labels.csv"),
    )
    roi_system = roi_to_system_array(mapping_df)
    system_pairs_df = build_system_pair_list()

    iu = np.triu_indices(N_ROI, k=1)
    edge_pair_idx = edge_to_system_pair(iu[0], iu[1], roi_system, system_pairs_df)

    _EDGE_PAIR_CACHE = edge_pair_idx
    return edge_pair_idx


# ── Inner selection for SFC expert ────────────────────────────────────────────
def _select_sfc_inner(
    X_fc_raw: np.ndarray,
    X_sc_raw: np.ndarray,
    y: np.ndarray,
    train_global: np.ndarray,
    alpha_grid: np.ndarray,
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
    edge_pair_idx: np.ndarray,
    n_pairs: int,
    prior: np.ndarray = None,
    top_k: int = None,
    use_ncr: bool = False,
    ratio_grid: np.ndarray = None,
) -> Tuple[dict, float]:
    """Select best (alpha, top_k, ratio) for SFC expert via inner CV."""
    global_to_local = {int(idx): i for i, idx in enumerate(train_global)}

    # Select pairs based on prior
    if top_k is not None and prior is not None:
        # Compute SFC features on training data to find nonzero pairs
        dummy_val = np.array([], dtype=int)
        X_sfc_train, _ = build_sfc_features_from_raw(
            X_fc_raw, X_sc_raw, train_global, dummy_val, edge_pair_idx, n_pairs
        )
        nonzero_mask = np.abs(X_sfc_train).sum(axis=0) > 0
        valid_indices = np.where(nonzero_mask)[0]
        if len(valid_indices) == 0:
            return {"alpha": alpha_grid[0], "top_k": top_k, "ratio": 0.0, "selected": np.array([], dtype=int)}, -np.inf
        valid_scores = prior[valid_indices]
        order = np.argsort(valid_scores)[::-1]
        n_select = min(top_k, len(valid_indices))
        selected = valid_indices[order[:n_select]]
    else:
        # All pairs (A1)
        dummy_val = np.array([], dtype=int)
        X_sfc_train, _ = build_sfc_features_from_raw(
            X_fc_raw, X_sc_raw, train_global, dummy_val, edge_pair_idx, n_pairs
        )
        nonzero_mask = np.abs(X_sfc_train).sum(axis=0) > 0
        selected = np.where(nonzero_mask)[0]

    if len(selected) == 0:
        return {"alpha": alpha_grid[0], "top_k": top_k, "ratio": 0.0, "selected": selected}, -np.inf

    best_pearson = -np.inf
    best_params = {"alpha": alpha_grid[0], "top_k": top_k, "ratio": 0.0, "selected": selected}

    ratios = ratio_grid if (use_ncr and ratio_grid is not None) else np.array([0.0])

    for alpha in alpha_grid:
        for ratio in ratios:
            fold_ps = []
            for b_global, c_global in inner_folds:
                # Build SFC features for this inner fold
                X_b, X_c = build_sfc_features_from_raw(
                    X_fc_raw, X_sc_raw, b_global, c_global, edge_pair_idx, n_pairs
                )

                # Subset to selected pairs
                X_b_sub = X_b[:, selected]
                X_c_sub = X_c[:, selected]

                # Standardize
                scaler = StandardScaler()
                X_b_z = scaler.fit_transform(X_b_sub)
                X_c_z = scaler.transform(X_c_sub)
                X_b_z = np.nan_to_num(X_b_z, nan=0.0, posinf=0.0, neginf=0.0)
                X_c_z = np.nan_to_num(X_c_z, nan=0.0, posinf=0.0, neginf=0.0)

                y_b = y[b_global]
                y_c = y[c_global]
                y_mean_b = y_b.mean()
                y_std_b = max(y_b.std(), 1e-8)
                y_b_z = (y_b - y_mean_b) / y_std_b

                if use_ncr and ratio > 0:
                    # NCR path
                    system_names_arr = _build_pair_system_names(n_pairs)
                    L = build_pair_laplacian(prior, selected, system_names_arr)
                    if L is not None:
                        eigvals_L, eigvecs_L = np.linalg.eigh(L)
                        eigvals_L = np.maximum(eigvals_L, 0.0)
                        pred_z = ncr_expert_predict(
                            X_b_z, y_b_z, X_c_z, alpha, ratio, eigvals_L, eigvecs_L
                        )
                        pred = pred_z * y_std_b + y_mean_b
                    else:
                        pred = ridge_expert_predict(X_b_z, y_b_z, X_c_z, alpha) * y_std_b + y_mean_b
                else:
                    pred = ridge_expert_predict(X_b_z, y_b_z, X_c_z, alpha) * y_std_b + y_mean_b

                r, _ = pearsonr(y_c, pred)
                fold_ps.append(r if np.isfinite(r) else -np.inf)

            mp = float(np.mean(fold_ps))
            if mp > best_pearson + 1e-14:
                best_pearson = mp
                best_params = {"alpha": alpha, "top_k": top_k, "ratio": ratio, "selected": selected}

    return best_params, best_pearson


# ── SFC expert OOF generation ────────────────────────────────────────────────
def generate_sfc_expert_oof(
    X_fc_raw: np.ndarray,
    X_sc_raw: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    prior: np.ndarray,
    seed: int,
    outer_fold: int,
    model_type: str = "A1",
    n_fusion_folds: int = 3,
    n_inner: int = 3,
    edge_pair_idx: np.ndarray = None,
    n_pairs: int = 45,
) -> Tuple[np.ndarray, dict]:
    """Generate cross-fitted OOF predictions for SFC expert."""
    if edge_pair_idx is None:
        edge_pair_idx = get_edge_pair_indices()

    fusion_folds = make_fusion_folds(train_idx, seed, outer_fold, n_fusion_folds)
    n_train = len(train_idx)
    oof = np.full(n_train, np.nan, dtype=np.float64)

    train_local_map = {int(idx): i for i, idx in enumerate(train_idx)}

    final_params = None

    for fold_k, (a_k, v_k) in enumerate(fusion_folds):
        inner_folds = make_inner_selection_folds(a_k, seed, outer_fold, fold_k, n_inner)
        v_local = np.array([train_local_map[int(idx)] for idx in v_k])

        # Determine selection parameters
        if model_type == "A1":
            # All pairs, no prior
            top_k_val = None
            use_prior = False
        elif model_type == "A2":
            # LLM-selected, search TOP_PAIR_GRID
            top_k_val = None  # will be set by grid search
            use_prior = True
        else:
            # A3-A6: prior-selected, search TOP_PAIR_GRID
            top_k_val = None  # will be set by grid search
            use_prior = True

        # For A2-A6, search TOP_PAIR_GRID
        if model_type in ("A2", "A3", "A4", "A5", "A6"):
            best_score = -np.inf
            best_tk = TOP_PAIR_GRID[0]
            for tk in TOP_PAIR_GRID:
                params, score = _select_sfc_inner(
                    X_fc_raw, X_sc_raw, y, a_k, SFC_ALPHA_GRID, inner_folds,
                    edge_pair_idx, n_pairs,
                    prior=prior if use_prior else None,
                    top_k=tk,
                    use_ncr=model_type in ("A3", "A4", "A5", "A6"),
                    ratio_grid=NCR_RATIO_GRID if model_type in ("A3", "A4", "A5", "A6") else None,
                )
                if score > best_score:
                    best_score = score
                    best_tk = tk
                    best_params = params
        else:
            best_params, _ = _select_sfc_inner(
                X_fc_raw, X_sc_raw, y, a_k, SFC_ALPHA_GRID, inner_folds,
                edge_pair_idx, n_pairs,
                prior=None, top_k=None,
                use_ncr=False, ratio_grid=None,
            )

        selected = best_params["selected"]
        if len(selected) == 0:
            oof[v_local] = 0.0
            final_params = best_params
            continue

        # Fit on a_k, predict v_k
        X_a, X_v = build_sfc_features_from_raw(
            X_fc_raw, X_sc_raw, a_k, v_k, edge_pair_idx, n_pairs
        )
        X_a_sub = X_a[:, selected]
        X_v_sub = X_v[:, selected]

        scaler = StandardScaler()
        X_a_z = scaler.fit_transform(X_a_sub)
        X_v_z = scaler.transform(X_v_sub)
        X_a_z = np.nan_to_num(X_a_z, nan=0.0, posinf=0.0, neginf=0.0)
        X_v_z = np.nan_to_num(X_v_z, nan=0.0, posinf=0.0, neginf=0.0)

        y_a = y[a_k]
        y_mean_a = y_a.mean()
        y_std_a = max(y_a.std(), 1e-8)
        y_a_z = (y_a - y_mean_a) / y_std_a

        alpha = best_params["alpha"]
        ratio = best_params.get("ratio", 0.0)

        if ratio > 0:
            system_names_arr = _build_pair_system_names(n_pairs)
            L = build_pair_laplacian(prior, selected, system_names_arr)
            if L is not None:
                eigvals_L, eigvecs_L = np.linalg.eigh(L)
                eigvals_L = np.maximum(eigvals_L, 0.0)
                pred_z = ncr_expert_predict(X_a_z, y_a_z, X_v_z, alpha, ratio, eigvals_L, eigvecs_L)
                pred = pred_z * y_std_a + y_mean_a
            else:
                pred = ridge_expert_predict(X_a_z, y_a_z, X_v_z, alpha) * y_std_a + y_mean_a
        else:
            pred = ridge_expert_predict(X_a_z, y_a_z, X_v_z, alpha) * y_std_a + y_mean_a

        oof[v_local] = pred
        final_params = best_params

    if final_params is None:
        final_params = {"alpha": SFC_ALPHA_GRID[0], "top_k": None, "ratio": 0.0, "selected": np.array([], dtype=int)}

    return oof, final_params


# ── Final SFC expert fit ──────────────────────────────────────────────────────
def fit_sfc_expert_final(
    X_fc_raw: np.ndarray,
    X_sc_raw: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    prior: np.ndarray,
    best_params: dict,
    edge_pair_idx: np.ndarray = None,
    n_pairs: int = 45,
) -> Tuple[dict, np.ndarray]:
    """Fit SFC expert on full training data, return test prediction."""
    if edge_pair_idx is None:
        edge_pair_idx = get_edge_pair_indices()
    selected = best_params["selected"]

    fit_result = {
        "selected": selected,
        "alpha": best_params["alpha"],
        "ratio": best_params.get("ratio", 0.0),
        "beta_standardized": None,
    }

    if len(selected) == 0:
        fit_result["test_pred"] = np.zeros(len(test_idx))
        return fit_result, np.zeros(len(test_idx))

    # Build SFC features
    X_train, X_test = build_sfc_features_from_raw(
        X_fc_raw, X_sc_raw, train_idx, test_idx, edge_pair_idx, n_pairs
    )
    X_train_sub = X_train[:, selected]
    X_test_sub = X_test[:, selected]

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train_sub)
    X_test_z = scaler.transform(X_test_sub)
    X_train_z = np.nan_to_num(X_train_z, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_z = np.nan_to_num(X_test_z, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = y[train_idx]
    y_mean = y_train.mean()
    y_std = max(y_train.std(), 1e-8)
    y_train_z = (y_train - y_mean) / y_std

    alpha = best_params["alpha"]
    ratio = best_params.get("ratio", 0.0)

    fit_result["scaler_mean"] = scaler.mean_
    fit_result["scaler_std"] = scaler.scale_
    fit_result["y_mean"] = y_mean
    fit_result["y_std"] = y_std

    if ratio > 0:
        system_names_arr = _build_pair_system_names(n_pairs)
        L = build_pair_laplacian(prior, selected, system_names_arr)
        if L is not None:
            eigvals_L, eigvecs_L = np.linalg.eigh(L)
            eigvals_L = np.maximum(eigvals_L, 0.0)
            test_pred = ncr_expert_predict(
                X_train_z, y_train_z, X_test_z, alpha, ratio, eigvals_L, eigvecs_L
            ) * y_std + y_mean
            fit_result["eigvecs_L"] = eigvecs_L
            fit_result["eigvals_L"] = eigvals_L
        else:
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X_train_z, y_train_z)
            fit_result["beta_standardized"] = model.coef_
            test_pred = model.predict(X_test_z) * y_std + y_mean
    else:
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X_train_z, y_train_z)
        fit_result["beta_standardized"] = model.coef_
        test_pred = model.predict(X_test_z) * y_std + y_mean

    fit_result["test_pred"] = test_pred
    return fit_result, test_pred


# ── Prior loading ──────────────────────────────────────────────────────────────
def get_llm_prior(task_name: str, control_type: str = "matched") -> np.ndarray:
    """Load prior scores for a given task and control type."""
    priors_dir = PRIORS_DIR / "llm_network_interaction"

    # Map task names
    task_slug = "working_memory" if task_name in ("WM", "working_memory") else "fluid_intelligence"

    if control_type == "matched":
        fname = f"{task_slug}_pair_prior.csv"
    elif control_type == "cross_task":
        other = "fi" if task_slug == "working_memory" else "wm"
        fname = f"cross_task_{other}_pair_prior.csv"
    elif control_type == "shuffled":
        fname = f"{task_slug}_shuffled_pair_prior.csv"
    elif control_type == "random":
        fname = f"{task_slug}_random_pair_prior.csv"
    else:
        raise ValueError(f"Unknown control type: {control_type}")

    df = pd.read_csv(priors_dir / fname)
    return df["normalized_score"].values.astype(np.float64)


# ── Main evaluation for one split ─────────────────────────────────────────────
def evaluate_phase2e_fix_split(
    X_fc_raw: np.ndarray,
    X_sc_raw: np.ndarray,
    y: np.ndarray,
    seed: int,
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    prior: np.ndarray,
    roi_prior: np.ndarray,
) -> dict:
    """Evaluate A0-A6 models for one task on one outer split."""
    edge_pair_idx = get_edge_pair_indices()
    n_pairs = 45
    y_test = y[test_idx]
    y_train = y[train_idx]

    results = {}

    # --- A0: Corrected R0 baseline ---
    condition = CONDITIONS["R0"]
    oof_r0 = generate_crossfit_oof(
        X_fc_raw, X_sc_raw, y, train_idx, condition, roi_prior,
        seed, outer_fold, RIDGE_GRID, N_FUSION_FOLDS, N_INNER, N_ROI,
    )
    fusion_weights_r0, _ = search_fusion_weights(
        y_train, {"FP": oof_r0.fp_oof, "SC": oof_r0.sc_oof}, ["FP", "SC"],
    )
    fc_final_r0, sc_final_r0, fp_final_r0 = reselect_and_fit_final(
        X_fc_raw, X_sc_raw, y, train_idx, test_idx, condition, roi_prior,
        seed, outer_fold, RIDGE_GRID, N_FINAL_CV, N_ROI,
    )
    w_fp_r0 = fusion_weights_r0["FP"]
    w_sc_r0 = fusion_weights_r0["SC"]
    a0_test = w_fp_r0 * fp_final_r0.test_pred + w_sc_r0 * sc_final_r0.test_pred
    a0_oof = w_fp_r0 * oof_r0.fp_oof + w_sc_r0 * oof_r0.sc_oof

    results["A0"] = {
        "test_pred": a0_test,
        "oof": a0_oof,
        "metrics": prediction_metrics(y_test, a0_test),
        "oof_r": pearsonr(y_train, a0_oof)[0],
        "fusion_weights": fusion_weights_r0,
    }

    # --- A1: All-pair SFC Ridge (no prior) ---
    oof_a1, params_a1 = generate_sfc_expert_oof(
        X_fc_raw, X_sc_raw, y, train_idx,
        prior, seed, outer_fold,
        model_type="A1",
        edge_pair_idx=edge_pair_idx, n_pairs=n_pairs,
    )
    _, a1_test_pred = fit_sfc_expert_final(
        X_fc_raw, X_sc_raw, y, train_idx, test_idx,
        prior, params_a1, edge_pair_idx, n_pairs,
    )
    eta_a1, _ = hierarchical_fusion_weights(y_train, a0_oof, oof_a1, ETA_GRID)
    a1_fused_test = (1 - eta_a1) * a0_test + eta_a1 * a1_test_pred
    a1_fused_oof = (1 - eta_a1) * a0_oof + eta_a1 * oof_a1

    results["A1"] = {
        "test_pred": a1_fused_test,
        "expert_test": a1_test_pred,
        "oof": a1_fused_oof,
        "expert_oof": oof_a1,
        "metrics": prediction_metrics(y_test, a1_fused_test),
        "expert_metrics": prediction_metrics(y_test, a1_test_pred),
        "eta": eta_a1,
        "oof_r": pearsonr(y_train, a1_fused_oof)[0],
    }

    # --- A2: LLM-selected SFC Ridge (prior-selected, no NCR) ---
    oof_a2, params_a2 = generate_sfc_expert_oof(
        X_fc_raw, X_sc_raw, y, train_idx,
        prior, seed, outer_fold,
        model_type="A2",
        edge_pair_idx=edge_pair_idx, n_pairs=n_pairs,
    )
    _, a2_test_pred = fit_sfc_expert_final(
        X_fc_raw, X_sc_raw, y, train_idx, test_idx,
        prior, params_a2, edge_pair_idx, n_pairs,
    )
    eta_a2, _ = hierarchical_fusion_weights(y_train, a0_oof, oof_a2, ETA_GRID)
    a2_fused_test = (1 - eta_a2) * a0_test + eta_a2 * a2_test_pred
    a2_fused_oof = (1 - eta_a2) * a0_oof + eta_a2 * oof_a2

    results["A2"] = {
        "test_pred": a2_fused_test,
        "expert_test": a2_test_pred,
        "oof": a2_fused_oof,
        "expert_oof": oof_a2,
        "metrics": prediction_metrics(y_test, a2_fused_test),
        "expert_metrics": prediction_metrics(y_test, a2_test_pred),
        "eta": eta_a2,
        "oof_r": pearsonr(y_train, a2_fused_oof)[0],
        "selected_top_k": len(params_a2.get("selected", [])),
    }

    # --- A3-A6: NCR experts with different priors ---
    task_name = getattr(evaluate_phase2e_fix_split, '_current_task', 'working_memory')

    model_priors = {
        "A3": prior,
        "A4": get_llm_prior(task_name, "cross_task"),
        "A5": get_llm_prior(task_name, "shuffled"),
        "A6": get_llm_prior(task_name, "random"),
    }

    for model_id in ["A3", "A4", "A5", "A6"]:
        ctrl_prior = model_priors[model_id]
        oof_ai, params_ai = generate_sfc_expert_oof(
            X_fc_raw, X_sc_raw, y, train_idx,
            ctrl_prior, seed, outer_fold,
            model_type=model_id,
            edge_pair_idx=edge_pair_idx, n_pairs=n_pairs,
        )
        _, ai_test_pred = fit_sfc_expert_final(
            X_fc_raw, X_sc_raw, y, train_idx, test_idx,
            ctrl_prior, params_ai, edge_pair_idx, n_pairs,
        )
        eta_ai, _ = hierarchical_fusion_weights(y_train, a0_oof, oof_ai, ETA_GRID)
        ai_fused_test = (1 - eta_ai) * a0_test + eta_ai * ai_test_pred
        ai_fused_oof = (1 - eta_ai) * a0_oof + eta_ai * oof_ai

        results[model_id] = {
            "test_pred": ai_fused_test,
            "expert_test": ai_test_pred,
            "oof": ai_fused_oof,
            "expert_oof": oof_ai,
            "metrics": prediction_metrics(y_test, ai_fused_test),
            "expert_metrics": prediction_metrics(y_test, ai_test_pred),
            "eta": eta_ai,
            "oof_r": pearsonr(y_train, ai_fused_oof)[0],
            "selected_top_k": len(params_ai.get("selected", [])),
            "selected_alpha": params_ai.get("alpha"),
            "selected_ratio": params_ai.get("ratio"),
            "prior_type": model_id,
        }

    return results


# ── Decision rule ──────────────────────────────────────────────────────────────
def determine_decision(split_df: pd.DataFrame) -> str:
    """Determine PHASE2E_FIX_DECISION from split-level results."""
    # Aggregate to seed-level: mean pearson across folds for each (task, seed, model)
    seed_agg = split_df.groupby(["task", "seed", "model"]).agg(
        pearson_mean=("pearson", "mean")
    ).reset_index()

    wm = seed_agg[seed_agg["task"] == "WM"]
    fi = seed_agg[seed_agg["task"] == "FI"]

    # Check STRONG_GO
    strong_go = True
    for task_df in [wm, fi]:
        a0 = task_df[task_df["model"] == "A0"]
        a1 = task_df[task_df["model"] == "A1"]
        a3 = task_df[task_df["model"] == "A3"]
        a5 = task_df[task_df["model"] == "A5"]
        a6 = task_df[task_df["model"] == "A6"]

        if len(a0) == 0 or len(a3) == 0:
            strong_go = False
            continue

        deltas_a3_a0 = a3.set_index("seed")["pearson_mean"] - a0.set_index("seed")["pearson_mean"]
        if deltas_a3_a0.mean() < 0.003 or sum(deltas_a3_a0 > 0) < 3:
            strong_go = False
            continue

        if len(a1) > 0:
            deltas_a3_a1 = a3.set_index("seed")["pearson_mean"] - a1.set_index("seed")["pearson_mean"]
            if deltas_a3_a1.mean() <= 0 or sum(deltas_a3_a1 > 0) < 3:
                strong_go = False
                continue

        # A3 beats shuffled and random
        if len(a5) > 0:
            deltas_a3_a5 = a3.set_index("seed")["pearson_mean"] - a5.set_index("seed")["pearson_mean"]
            if deltas_a3_a5.mean() <= 0:
                strong_go = False
                continue
        if len(a6) > 0:
            deltas_a3_a6 = a3.set_index("seed")["pearson_mean"] - a6.set_index("seed")["pearson_mean"]
            if deltas_a3_a6.mean() <= 0:
                strong_go = False
                continue

    if strong_go:
        max_delta = 0
        for task_df in [wm, fi]:
            a0 = task_df[task_df["model"] == "A0"]
            a3 = task_df[task_df["model"] == "A3"]
            if len(a0) > 0 and len(a3) > 0:
                deltas = a3.set_index("seed")["pearson_mean"] - a0.set_index("seed")["pearson_mean"]
                max_delta = max(max_delta, deltas.mean())
        if max_delta >= 0.005:
            return "STRONG_GO"

    # Check PROMISING_GO
    promising = False
    for task_df in [wm, fi]:
        a0 = task_df[task_df["model"] == "A0"]
        a3 = task_df[task_df["model"] == "A3"]
        if len(a0) > 0 and len(a3) > 0:
            deltas_a3_a0 = a3.set_index("seed")["pearson_mean"] - a0.set_index("seed")["pearson_mean"]
            if deltas_a3_a0.mean() >= 0.003 and sum(deltas_a3_a0 > 0) >= 3:
                promising = True

    if promising:
        return "PROMISING_GO"

    # Check SFC_ONLY_GO
    sfc_only = False
    for task_df in [wm, fi]:
        a0 = task_df[task_df["model"] == "A0"]
        a1 = task_df[task_df["model"] == "A1"]
        if len(a0) > 0 and len(a1) > 0:
            deltas_a1_a0 = a1.set_index("seed")["pearson_mean"] - a0.set_index("seed")["pearson_mean"]
            if deltas_a1_a0.mean() > 0:
                sfc_only = True

    if sfc_only:
        return "SFC_ONLY_GO"

    return "NO_GO"


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    """Run the full Phase 2E-FIX pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("=" * 80)
    log.info("Phase 2E-FIX: Genuine LLM Interaction Prior + Correct Nested SFC Evaluation")
    log.info("=" * 80)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    COEFF_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    log.info("Loading data...")
    X_fc, X_sc, y_wm, y_fi = load_data()
    n_subjects = len(y_wm)
    log.info(f"  Subjects: {n_subjects}, FC edges: {X_fc.shape[1]}, SC edges: {X_sc.shape[1]}")

    roi_prior = load_roi_prior()

    # Step 1: Baseline audit
    log.info("")
    log.info("=" * 80)
    log.info("STEP 1: Baseline audit (seeds 0-9)")
    log.info("=" * 80)

    audit_seeds = list(range(10))
    audit_splits = make_outer_splits(n_subjects, audit_seeds, N_OUTER_FOLDS)

    wm_r0_results = []
    fi_r0_results = []

    condition = CONDITIONS["R0"]
    for seed, fold, train_idx, test_idx in audit_splits:
        for task_name, y in [("WM", y_wm), ("FI", y_fi)]:
            oof = generate_crossfit_oof(
                X_fc, X_sc, y, train_idx, condition, roi_prior,
                seed, fold, RIDGE_GRID, N_FUSION_FOLDS, N_INNER, N_ROI,
            )
            fw, _ = search_fusion_weights(
                y[train_idx], {"FP": oof.fp_oof, "SC": oof.sc_oof}, ["FP", "SC"],
            )

            fc_final, sc_final, fp_final = reselect_and_fit_final(
                X_fc, X_sc, y, train_idx, test_idx, condition, roi_prior,
                seed, fold, RIDGE_GRID, N_FINAL_CV, N_ROI,
            )
            w_fp = fw["FP"]
            w_sc = fw["SC"]
            fused_test = w_fp * fp_final.test_pred + w_sc * sc_final.test_pred

            metrics = prediction_metrics(y[test_idx], fused_test)
            if task_name == "WM":
                wm_r0_results.append({"seed": seed, "fold": fold, **metrics})
            else:
                fi_r0_results.append({"seed": seed, "fold": fold, **metrics})

    wm_r0_df = pd.DataFrame(wm_r0_results)
    fi_r0_df = pd.DataFrame(fi_r0_results)

    wm_r_mean = wm_r0_df["pearson"].mean()
    fi_r_mean = fi_r0_df["pearson"].mean()
    wm_rmse_mean = wm_r0_df["rmse"].mean()
    fi_rmse_mean = fi_r0_df["rmse"].mean()

    audit_pass = (
        abs(wm_r_mean - 0.263515) < 0.01 and
        abs(fi_r_mean - 0.370917) < 0.01 and
        abs(wm_rmse_mean - 11.292921) < 0.1 and
        abs(fi_rmse_mean - 4.566689) < 0.1
    )

    log.info(f"  WM fused r: {wm_r_mean:.6f} (expected ~0.263515, tol=0.01) -> {'PASS' if abs(wm_r_mean - 0.263515) < 0.01 else 'FAIL'}")
    log.info(f"  FI fused r: {fi_r_mean:.6f} (expected ~0.370917, tol=0.01) -> {'PASS' if abs(fi_r_mean - 0.370917) < 0.01 else 'FAIL'}")
    log.info(f"  WM fused rmse: {wm_rmse_mean:.6f} (expected ~11.292921, tol=0.1) -> {'PASS' if abs(wm_rmse_mean - 11.292921) < 0.1 else 'FAIL'}")
    log.info(f"  FI fused rmse: {fi_rmse_mean:.6f} (expected ~4.566689, tol=0.1) -> {'PASS' if abs(fi_rmse_mean - 4.566689) < 0.1 else 'FAIL'}")
    log.info(f"  Overall: {'PASS' if audit_pass else 'FAIL'}")

    # Save baseline audit
    with open(OUTPUT_DIR / "BASELINE_AUDIT.json", "w") as f:
        json.dump({
            "wm_pearson": float(wm_r_mean),
            "fi_pearson": float(fi_r_mean),
            "wm_rmse": float(wm_rmse_mean),
            "fi_rmse": float(fi_rmse_mean),
            "pass": bool(audit_pass),
        }, f, indent=2)

    if not audit_pass:
        log.error("Baseline audit FAILED. Stopping.")
        return

    # Step 2: System mapping
    log.info("")
    log.info("=" * 80)
    log.info("STEP 2: AAL116 → Yeo7 system mapping")
    log.info("=" * 80)

    mapping_df, qc = build_aal116_yeo7_mapping(
        str(REPO_ROOT / "inputs" / "atlases" / "AAL116.nii.gz"),
        str(REPO_ROOT / "inputs" / "atlases" / "AAL116_labels.csv"),
    )
    save_mapping(mapping_df, qc, PRIORS_DIR / "system_mapping")
    roi_system = roi_to_system_array(mapping_df)
    system_pairs_df = build_system_pair_list()

    log.info(f"  Systems: {qc['system_counts']}")
    log.info(f"  Pairs: {len(system_pairs_df)}")
    log.info(f"  Unresolved: {qc['n_unresolved']}")

    # Step 3: LLM priors
    log.info("")
    log.info("=" * 80)
    log.info("STEP 3: LLM network-interaction priors")
    log.info("=" * 80)

    from metascfc.phase2e_fix.llm_prior_generator_fix import (
        generate_prior_for_task,
        create_control_priors,
        compute_prior_diagnostics,
        freeze_priors,
        validate_frozen_priors,
    )

    priors_dir = PRIORS_DIR / "llm_network_interaction"
    priors_dir.mkdir(parents=True, exist_ok=True)

    # Check if priors already exist and are frozen
    if not (priors_dir / "FROZEN").exists():
        log.info("  Generating WM priors via Ollama REST API...")
        wm_df, wm_prov, wm_raw = generate_prior_for_task("working_memory")

        log.info("  Generating FI priors via Ollama REST API...")
        fi_df, fi_prov, fi_raw = generate_prior_for_task("fluid_intelligence")

        # Save priors
        wm_df.to_csv(priors_dir / "working_memory_pair_prior.csv", index=False)
        fi_df.to_csv(priors_dir / "fluid_intelligence_pair_prior.csv", index=False)

        # Save provenance
        with open(priors_dir / "working_memory_generation_provenance.json", "w") as f:
            json.dump(wm_prov, f, indent=2)
        with open(priors_dir / "fluid_intelligence_generation_provenance.json", "w") as f:
            json.dump(fi_prov, f, indent=2)

        # Save raw responses
        for key, text in wm_raw.items():
            with open(priors_dir / f"{key}.json", "w") as f:
                json.dump({"response": text}, f, indent=2)
        for key, text in fi_raw.items():
            with open(priors_dir / f"{key}.json", "w") as f:
                json.dump({"response": text}, f, indent=2)

        # Create control priors
        controls = create_control_priors(wm_df, fi_df)
        for name, ctrl_df in controls.items():
            ctrl_df.to_csv(priors_dir / f"{name}_pair_prior.csv", index=False)

        # Freeze
        all_files = list(priors_dir.glob("*.csv")) + list(priors_dir.glob("*.json"))
        freeze_priors(priors_dir, all_files)
        log.info("  Priors frozen.")
    else:
        log.info("  Priors already frozen. Validating...")
        validate_frozen_priors(priors_dir)
        wm_df = pd.read_csv(priors_dir / "working_memory_pair_prior.csv")
        fi_df = pd.read_csv(priors_dir / "fluid_intelligence_pair_prior.csv")

    # Diagnostics
    diag = compute_prior_diagnostics(wm_df, fi_df)
    log.info(f"  WM-FI Pearson: {diag['wm_fi_pearson']:.4f}")
    log.info(f"  WM-FI Spearman: {diag['wm_fi_spearman']:.4f}")
    log.info(f"  Top-5 overlap: {diag['top5_overlap']}")
    log.info(f"  Top-10 overlap: {diag['top10_overlap']}")

    # Load priors as arrays
    prior_wm = wm_df["normalized_score"].values.astype(np.float64)
    prior_fi = fi_df["normalized_score"].values.astype(np.float64)

    # Step 4: Main evaluation
    log.info("")
    log.info("=" * 80)
    log.info("STEP 4: Main pilot evaluation (A0-A6)")
    log.info("=" * 80)
    log.info(f"  Seeds: {PHASE2E_FIX_DEV_SEEDS}")
    log.info(f"  Outer folds: {N_OUTER_FOLDS}")
    log.info(f"  Total outer splits: {len(PHASE2E_FIX_DEV_SEEDS) * N_OUTER_FOLDS}")

    all_split_results = []
    seed_summary = []

    for task_name, y, prior in [("WM", y_wm, prior_wm), ("FI", y_fi, prior_fi)]:
        log.info(f"\n  Task: {task_name}")
        evaluate_phase2e_fix_split._current_task = task_name

        splits = make_outer_splits(n_subjects, PHASE2E_FIX_DEV_SEEDS, N_OUTER_FOLDS)
        for i, (seed, fold, train_idx, test_idx) in enumerate(splits):
            log.info(f"    [{i + 1}/{len(splits)}] seed={seed} fold={fold} train={len(train_idx)} test={len(test_idx)}")

            t0 = time.time()
            results = evaluate_phase2e_fix_split(
                X_fc, X_sc, y, seed, fold, train_idx, test_idx,
                prior, roi_prior,
            )
            elapsed = time.time() - t0

            for model_name, res in results.items():
                m = res["metrics"]
                all_split_results.append({
                    "task": task_name,
                    "seed": seed,
                    "fold": fold,
                    "model": model_name,
                    "pearson": m["pearson"],
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "eta": res.get("eta"),
                    "expert_pearson": res.get("expert_metrics", {}).get("pearson"),
                    "expert_rmse": res.get("expert_metrics", {}).get("rmse"),
                    "expert_mae": res.get("expert_metrics", {}).get("mae"),
                    "selected_top_k": res.get("selected_top_k"),
                    "selected_alpha": res.get("selected_alpha"),
                    "selected_ratio": res.get("selected_ratio"),
                    "prior_type": res.get("prior_type", "matched"),
                })

            log.info(f"      A0 r={results['A0']['metrics']['pearson']:.4f} A3 r={results['A3']['metrics']['pearson']:.4f} ({elapsed:.1f}s)")

        # Seed summary
        task_df = pd.DataFrame([r for r in all_split_results if r["task"] == task_name])
        for model in ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]:
            mdf = task_df[task_df["model"] == model]
            if len(mdf) > 0:
                seed_summary.append({
                    "task": task_name,
                    "model": model,
                    "pearson_mean": mdf["pearson"].mean(),
                    "pearson_std": mdf["pearson"].std(),
                    "rmse_mean": mdf["rmse"].mean(),
                    "mae_mean": mdf["mae"].mean(),
                    "eta_mean": mdf["eta"].mean() if mdf["eta"].notna().any() else None,
                    "n_seeds": len(mdf),
                })

    # Save results
    split_df = pd.DataFrame(all_split_results)
    split_df.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)

    seed_df = pd.DataFrame(seed_summary)
    seed_df.to_csv(OUTPUT_DIR / "seed_metrics.csv", index=False)

    # Model summary
    model_summary = []
    for _, row in seed_df.iterrows():
        model_summary.append(row.to_dict())
    pd.DataFrame(model_summary).to_csv(OUTPUT_DIR / "model_summary.csv", index=False)

    # Determine decision
    decision = determine_decision(split_df)

    # Print report
    log.info("")
    log.info("=" * 80)
    log.info("FINAL REPORT")
    log.info("=" * 80)

    log.info("\n## E. Corrected prediction results")
    for task_name in ["WM", "FI"]:
        log.info(f"\n  {task_name}:")
        task_data = seed_df[seed_df["task"] == task_name]
        for _, row in task_data.iterrows():
            log.info(f"    {row['model']}: r={row['pearson_mean']:.4f}±{row['pearson_std']:.4f}  rmse={row['rmse_mean']:.4f}")

    log.info(f"\n  PHASE2E_FIX_DECISION: {decision}")
    log.info(f"  STATUS: PHASE2E_FIX_COMPLETE")

    # Save decision
    with open(OUTPUT_DIR / "VALIDATION_REPORT.json", "w") as f:
        json.dump({
            "decision": str(decision),
            "status": "PHASE2E_FIX_COMPLETE",
            "baseline_audit": "PASS",
            "wm_fi_pearson": float(diag["wm_fi_pearson"]),
            "top5_overlap": int(diag["top5_overlap"]),
        }, f, indent=2)

    with open(OUTPUT_DIR / "COMPLETE", "w") as f:
        f.write(f"COMPLETE at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")

    # Zip outputs
    zip_path = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in OUTPUT_DIR.rglob("*"):
            if fp.is_file() and "checkpoints" not in str(fp):
                zf.write(fp, fp.relative_to(OUTPUT_DIR.parent))
    log.info(f"\n  Zipped outputs to {zip_path}")


if __name__ == "__main__":
    main()
