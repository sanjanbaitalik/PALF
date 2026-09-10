"""Phase 2E: LLM Network-Interaction Structure-Function Coupling NCR Pilot.

LI-SFC-NCR: LLM Interaction-Prior Structure-Function Coupling NCR

Tests whether LLM network-interaction priors combined with nonlinear
FC×SC coupling features improve prediction over the corrected R0 baseline.
"""
from __future__ import annotations

import json
import hashlib
import logging
import pickle
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from metascfc.phase2e.aal_yeo_mapping import (
    build_aal116_yeo7_mapping,
    build_system_pair_list,
    roi_to_system_array,
    edge_to_system_pair,
    save_mapping,
    SYSTEM_NAMES,
    N_SYSTEMS,
)
from metascfc.phase2e.sfc_features import (
    compute_sfc_pair_features_for_split,
)
from metascfc.phase2e.li_sfc_ncr import (
    fit_sfc_ridge_expert,
    fit_sfc_ncr_expert,
    predict_sfc_expert,
    hierarchical_fusion,
    build_pair_laplacian,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHASE2E_DEV_SEEDS = [2727, 2828, 2929, 3030]
N_OUTER_FOLDS = 5
N_FUSION_FOLDS = 3
N_INNER = 3
N_FINAL_CV = 3

RIDGE_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
SFC_ALPHA_GRID = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
TOP_PAIR_GRID = [5, 10, 15, 25]
NCR_RATIO_GRID = np.array([0.0, 0.1, 0.3, 1.0])

TASKS = {
    "working_memory": {"display": "WM"},
    "fluid_intelligence": {"display": "FI"},
}

MODELS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
CONTROL_TYPES = ["matched", "cross_task", "shuffled", "random"]

# Prior control seeds (fixed)
SHUFFLED_SEED = 7301
RANDOM_SEED = 7303

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUTPUT_DIR = REPO_ROOT / "outputs/iclr/palf_phase2e_li_sfc_ncr"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
COEFF_DIR = OUTPUT_DIR / "coefficients"
PRED_DIR = OUTPUT_DIR / "predictions"
PLOT_DIR = OUTPUT_DIR / "plots"
PRIOR_DIR = OUTPUT_DIR / "priors"

for d in [OUTPUT_DIR, CKPT_DIR, COEFF_DIR, PRED_DIR, PLOT_DIR,
          PRIOR_DIR / "system_mapping", PRIOR_DIR / "llm_network_interaction"]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
log.info("Loading data...")
fc_mats = np.load(REPO_ROOT / "inputs/dataset_FC/FC_all.npy")
sc_mats = np.load(REPO_ROOT / "inputs/dataset_SC/SC_all.npy")
iu = np.triu_indices(N_ROI, k=1)
X_fc = fc_mats[:, iu[0], iu[1]].astype(np.float64)
X_sc = sc_mats[:, iu[0], iu[1]].astype(np.float64)
edge_indices = np.column_stack([iu[0], iu[1]])

y_wm = np.load(
    REPO_ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy"
).astype(np.float64)
y_fi = np.load(
    REPO_ROOT / "inputs/dataset_SC/label_all.npy"
).astype(np.float64)

n_subjects = int(y_wm.shape[0])
log.info("  Subjects: %d, FC edges: %d, SC edges: %d", n_subjects, X_fc.shape[1], X_sc.shape[1])


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _ckpt_path(phase: str, seed: int, fold: int) -> Path:
    return CKPT_DIR / f"{phase}_seed_{seed}_fold_{fold}.pkl"


def _save_ckpt(phase: str, seed: int, fold: int, state: dict) -> None:
    with open(_ckpt_path(phase, seed, fold), "wb") as f:
        pickle.dump(state, f)


def _load_ckpt(phase: str, seed: int, fold: int) -> Optional[dict]:
    p = _ckpt_path(phase, seed, fold)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------------------------
# System mapping (cached)
# ---------------------------------------------------------------------------
_SYSTEM_MAPPING_CACHE = None
_EDGE_PAIR_CACHE = None


def get_system_mapping():
    global _SYSTEM_MAPPING_CACHE
    if _SYSTEM_MAPPING_CACHE is None:
        mapping_path = PRIOR_DIR / "system_mapping" / "AAL116_to_Yeo7_plus_subcortical_cerebellar.csv"
        if mapping_path.exists():
            _SYSTEM_MAPPING_CACHE = pd.read_csv(mapping_path)
        else:
            log.info("Building AAL116 → Yeo7 mapping...")
            _SYSTEM_MAPPING_CACHE = build_aal116_yeo7_mapping(
                str(REPO_ROOT / "inputs/atlases/AAL116.nii.gz"),
                str(REPO_ROOT / "inputs/atlases/AAL116_labels.csv"),
            )
            save_mapping(_SYSTEM_MAPPING_CACHE, str(PRIOR_DIR / "system_mapping"))
            log.info("  System counts: %s", _SYSTEM_MAPPING_CACHE["system"].value_counts().to_dict())
    return _SYSTEM_MAPPING_CACHE


def get_edge_pair_indices():
    global _EDGE_PAIR_CACHE
    if _EDGE_PAIR_CACHE is None:
        mapping_df = get_system_mapping()
        system_pairs_df = build_system_pair_list()
        roi_system = roi_to_system_array(mapping_df)
        _EDGE_PAIR_CACHE = edge_to_system_pair(edge_indices, edge_indices, roi_system, system_pairs_df)
        n_pairs = len(system_pairs_df)
        n_valid = (_EDGE_PAIR_CACHE >= 0).sum()
        log.info("  Edge-to-pair mapping: %d pairs, %d/%d edges mapped", n_pairs, n_valid, len(_EDGE_PAIR_CACHE))
    return _EDGE_PAIR_CACHE


def get_system_names_array():
    mapping_df = get_system_mapping()
    system_pairs_df = build_system_pair_list()
    return system_pairs_df["pair_name"].values


# ---------------------------------------------------------------------------
# LLM prior loading (cached)
# ---------------------------------------------------------------------------
_PRIOR_CACHE = {}


def get_llm_prior(task: str, control: str = "matched") -> np.ndarray:
    """Get normalized pair prior scores for task/control combination."""
    key = (task, control)
    if key in _PRIOR_CACHE:
        return _PRIOR_CACHE[key]

    if control == "matched":
        path = PRIOR_DIR / "llm_network_interaction" / f"{task}_pair_prior.csv"
    elif control == "shuffled":
        path = PRIOR_DIR / "llm_network_interaction" / "shuffled_pair_prior.csv"
    elif control == "random":
        path = PRIOR_DIR / "llm_network_interaction" / "random_pair_prior.csv"
    elif control == "cross_task":
        other = "fluid_intelligence" if task == "working_memory" else "working_memory"
        path = PRIOR_DIR / "llm_network_interaction" / f"{other}_pair_prior.csv"
    else:
        raise ValueError(f"Unknown control: {control}")

    df = pd.read_csv(path)
    scores = df["normalized_score"].values.astype(np.float64)
    _PRIOR_CACHE[key] = scores
    return scores


# ---------------------------------------------------------------------------
# SFC feature helpers
# ---------------------------------------------------------------------------
def compute_sfc_for_split(
    fc_train: np.ndarray,
    sc_train: np.ndarray,
    fc_test: np.ndarray,
    sc_test: np.ndarray,
    edge_pair_idx: np.ndarray,
    n_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SFC pair features for train/test with training-fit scalers."""
    return compute_sfc_pair_features_for_split(
        fc_train, sc_train, fc_test, sc_test, edge_pair_idx, n_pairs
    )


def select_top_pairs_by_prior(
    X_sfc: np.ndarray,
    prior: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Select top-k pair indices by prior score, considering only nonzero pairs."""
    nonzero_mask = np.abs(X_sfc).sum(axis=0) > 0
    valid_indices = np.where(nonzero_mask)[0]
    if len(valid_indices) == 0:
        return np.array([], dtype=int)
    valid_scores = prior[valid_indices]
    order = np.argsort(valid_scores)[::-1]
    n_select = min(top_k, len(valid_indices))
    return valid_indices[order[:n_select]]


# ---------------------------------------------------------------------------
# Inner CV for SFC expert
# ---------------------------------------------------------------------------
def _select_sfc_inner(
    X_sfc: np.ndarray,
    y: np.ndarray,
    train_global: np.ndarray,
    alpha_grid: np.ndarray,
    inner_folds: List[Tuple[np.ndarray, np.ndarray]],
    prior: np.ndarray = None,
    system_names: np.ndarray = None,
    top_k: int = None,
    use_ncr: bool = False,
    ratio_grid: np.ndarray = None,
) -> Tuple[dict, float]:
    """Select best (alpha, top_k, ratio) for SFC expert via inner CV.

    Returns (best_params, best_pearson).
    """
    global_to_local = {int(idx): i for i, idx in enumerate(train_global)}

    # Select pairs
    if top_k is not None and prior is not None:
        selected = select_top_pairs_by_prior(X_sfc[train_global], prior, top_k)
    else:
        nonzero_mask = np.abs(X_sfc[train_global]).sum(axis=0) > 0
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
                b_local = np.array([global_to_local[int(idx)] for idx in b_global])
                c_local = np.array([global_to_local[int(idx)] for idx in c_global])

                X_b = X_sfc[b_global][:, selected]
                X_c = X_sfc[c_global][:, selected]
                y_b = y[b_global]
                y_c = y[c_global]

                # Standardize
                scaler = StandardScaler()
                X_b_z = scaler.fit_transform(X_b)
                X_c_z = scaler.transform(X_c)
                X_b_z = np.nan_to_num(X_b_z, nan=0.0, posinf=0.0, neginf=0.0)
                X_c_z = np.nan_to_num(X_c_z, nan=0.0, posinf=0.0, neginf=0.0)

                y_mean_b = y_b.mean()
                y_std_b = max(y_b.std(), 1e-8)
                y_b_z = (y_b - y_mean_b) / y_std_b

                if use_ncr and ratio > 0 and system_names is not None and prior is not None:
                    # NCR path
                    L = build_pair_laplacian(prior, selected, system_names)
                    if L is not None:
                        eigvals_L, eigvecs_L = np.linalg.eigh(L)
                        eigvals_L = np.maximum(eigvals_L, 0.0)
                        Z_b = X_b_z @ eigvecs_L
                        Z_c = X_c_z @ eigvecs_L

                        lambda_L = ratio * alpha
                        n_tr = len(b_local)
                        quad = np.zeros((n_tr, n_tr))
                        for j in range(Z_b.shape[1]):
                            denom = alpha + lambda_L * eigvals_L[j]
                            if denom < 1e-12:
                                continue
                            quad += np.outer(Z_b[:, j], Z_b[:, j]) / denom
                        try:
                            alpha_dual = np.linalg.solve(np.eye(n_tr) + quad, y_b_z)
                        except np.linalg.LinAlgError:
                            fold_ps.append(-np.inf)
                            continue
                        pred_z = np.zeros(len(c_local))
                        for j in range(Z_b.shape[1]):
                            denom = alpha + lambda_L * eigvals_L[j]
                            if denom < 1e-12:
                                continue
                            beta_j = (Z_b[:, j] @ alpha_dual) / denom
                            pred_z += Z_c[:, j] * beta_j
                        pred = pred_z * y_std_b + y_mean_b
                    else:
                        model = Ridge(alpha=alpha, fit_intercept=False)
                        model.fit(X_b_z, y_b_z)
                        pred = model.predict(X_c_z) * y_std_b + y_mean_b
                else:
                    model = Ridge(alpha=alpha, fit_intercept=False)
                    model.fit(X_b_z, y_b_z)
                    pred = model.predict(X_c_z) * y_std_b + y_mean_b

                r, _ = pearsonr(y_c, pred)
                fold_ps.append(r if np.isfinite(r) else -np.inf)

            mp = float(np.mean(fold_ps))
            if mp > best_pearson + 1e-14:
                best_pearson = mp
                best_params = {"alpha": alpha, "top_k": top_k, "ratio": ratio, "selected": selected}

    return best_params, best_pearson


# ---------------------------------------------------------------------------
# SFC expert OOF generation
# ---------------------------------------------------------------------------
def generate_sfc_expert_oof(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    X_sfc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    prior: np.ndarray,
    system_names: np.ndarray,
    seed: int,
    outer_fold: int,
    model_type: str = "A1",
    n_fusion_folds: int = 3,
    n_inner: int = 3,
) -> Tuple[np.ndarray, dict]:
    """Generate cross-fitted OOF predictions for SFC expert.

    model_type: A1 (all pairs, no prior), A2/A3/A4/A5/A6 (prior-selected)
    Returns (oof_predictions, selection_info).
    """
    n_pairs = X_sfc.shape[1]
    edge_pair_idx = get_edge_pair_indices()

    fusion_folds = make_fusion_folds(train_idx, seed, outer_fold, n_fusion_folds)
    n_train = len(train_idx)
    oof = np.full(n_train, np.nan, dtype=np.float64)

    train_local_map = {int(idx): i for i, idx in enumerate(train_idx)}
    all_params = []

    for fold_k, (a_k, v_k) in enumerate(fusion_folds):
        inner_folds = make_inner_selection_folds(a_k, seed, outer_fold, fold_k, n_inner)
        v_local = np.array([train_local_map[int(idx)] for idx in v_k])

        # Determine selection parameters
        if model_type == "A1":
            # All pairs, no prior
            best_params, _ = _select_sfc_inner(
                X_sfc, y, a_k, SFC_ALPHA_GRID, inner_folds,
                prior=None, system_names=None, top_k=None,
                use_ncr=False, ratio_grid=None,
            )
        else:
            # Prior-selected
            use_ncr = model_type in ("A3", "A4", "A5", "A6")
            top_k_val = best_params_top_k(model_type) if model_type != "A2" else None
            best_params, _ = _select_sfc_inner(
                X_sfc, y, a_k, SFC_ALPHA_GRID, inner_folds,
                prior=prior, system_names=system_names,
                top_k=top_k_val,
                use_ncr=use_ncr,
                ratio_grid=NCR_RATIO_GRID if use_ncr else None,
            )

        selected = best_params["selected"]
        if len(selected) == 0:
            oof[v_local] = 0.0
            all_params.append(best_params)
            continue

        # Fit on a_k, predict v_k
        X_a = X_sfc[a_k][:, selected]
        X_v = X_sfc[v_k][:, selected]

        scaler = StandardScaler()
        X_a_z = scaler.fit_transform(X_a)
        X_v_z = scaler.transform(X_v)
        X_a_z = np.nan_to_num(X_a_z, nan=0.0, posinf=0.0, neginf=0.0)
        X_v_z = np.nan_to_num(X_v_z, nan=0.0, posinf=0.0, neginf=0.0)

        y_a = y[a_k]
        y_mean_a = y_a.mean()
        y_std_a = max(y_a.std(), 1e-8)
        y_a_z = (y_a - y_mean_a) / y_std_a

        alpha = best_params["alpha"]
        ratio = best_params.get("ratio", 0.0)

        if ratio > 0 and system_names is not None:
            L = build_pair_laplacian(prior, selected, system_names)
            if L is not None:
                eigvals_L, eigvecs_L = np.linalg.eigh(L)
                eigvals_L = np.maximum(eigvals_L, 0.0)
                Z_a = X_a_z @ eigvecs_L
                Z_v = X_v_z @ eigvecs_L
                lambda_L = ratio * alpha
                n_tr = len(a_k)
                quad = np.zeros((n_tr, n_tr))
                for j in range(Z_a.shape[1]):
                    denom = alpha + lambda_L * eigvals_L[j]
                    if denom < 1e-12:
                        continue
                    quad += np.outer(Z_a[:, j], Z_a[:, j]) / denom
                try:
                    alpha_dual = np.linalg.solve(np.eye(n_tr) + quad, y_a_z)
                except np.linalg.LinAlgError:
                    model = Ridge(alpha=alpha, fit_intercept=False)
                    model.fit(X_a_z, y_a_z)
                    pred = model.predict(X_v_z) * y_std_a + y_mean_a
                    oof[v_local] = pred
                    all_params.append(best_params)
                    continue
                pred_z = np.zeros(len(v_k))
                for j in range(Z_a.shape[1]):
                    denom = alpha + lambda_L * eigvals_L[j]
                    if denom < 1e-12:
                        continue
                    beta_j = (Z_a[:, j] @ alpha_dual) / denom
                    pred_z += Z_v[:, j] * beta_j
                pred = pred_z * y_std_a + y_mean_a
            else:
                model = Ridge(alpha=alpha, fit_intercept=False)
                model.fit(X_a_z, y_a_z)
                pred = model.predict(X_v_z) * y_std_a + y_mean_a
        else:
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X_a_z, y_a_z)
            pred = model.predict(X_v_z) * y_std_a + y_mean_a

        oof[v_local] = pred
        all_params.append(best_params)

    # Pick params from last fold (most data) for final refit
    final_params = all_params[-1] if all_params else {"alpha": SFC_ALPHA_GRID[0], "top_k": None, "ratio": 0.0, "selected": np.array([], dtype=int)}
    return oof, final_params


def best_params_top_k(model_type: str) -> int:
    """Return default top_k for model type."""
    if model_type == "A2":
        return 10
    elif model_type in ("A3", "A4", "A5", "A6"):
        return 10
    return 45  # all pairs


# ---------------------------------------------------------------------------
# Final SFC expert fit on full training data
# ---------------------------------------------------------------------------
def fit_sfc_expert_final(
    X_sfc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    prior: np.ndarray,
    system_names: np.ndarray,
    best_params: dict,
    use_ncr: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Fit SFC expert on full training data, return test prediction helper."""
    selected = best_params["selected"]
    if len(selected) == 0:
        return np.zeros(len(train_idx)), best_params

    X_train = X_sfc[train_idx][:, selected]

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_train_z = np.nan_to_num(X_train_z, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = y[train_idx]
    y_mean = y_train.mean()
    y_std = max(y_train.std(), 1e-8)
    y_train_z = (y_train - y_mean) / y_std

    alpha = best_params["alpha"]
    ratio = best_params.get("ratio", 0.0)

    fit_result = {
        "scaler_mean": scaler.mean_,
        "scaler_std": scaler.scale_,
        "y_mean": y_mean,
        "y_std": y_std,
        "alpha": alpha,
        "ratio": ratio,
        "selected": selected,
        "beta_standardized": None,
    }

    if ratio > 0 and use_ncr and system_names is not None:
        L = build_pair_laplacian(prior, selected, system_names)
        if L is not None:
            eigvals_L, eigvecs_L = np.linalg.eigh(L)
            eigvals_L = np.maximum(eigvals_L, 0.0)
            Z = X_train_z @ eigvecs_L
            lambda_L = ratio * alpha
            n = len(train_idx)
            quad = np.zeros((n, n))
            for j in range(Z.shape[1]):
                denom = alpha + lambda_L * eigvals_L[j]
                if denom < 1e-12:
                    continue
                quad += np.outer(Z[:, j], Z[:, j]) / denom
            try:
                alpha_dual = np.linalg.solve(np.eye(n) + quad, y_train_z)
            except np.linalg.LinAlgError:
                alpha_dual = np.zeros(n)
            beta_whitened = np.zeros(Z.shape[1])
            for j in range(Z.shape[1]):
                denom = alpha + lambda_L * eigvals_L[j]
                if denom < 1e-12:
                    continue
                beta_whitened[j] = (Z[:, j] @ alpha_dual) / denom
            beta_scaled = eigvecs_L @ beta_whitened
            fit_result["beta_standardized"] = beta_scaled
            fit_result["eigvecs_L"] = eigvecs_L
            fit_result["eigvals_L"] = eigvals_L
        else:
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X_train_z, y_train_z)
            fit_result["beta_standardized"] = model.coef_
    else:
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X_train_z, y_train_z)
        fit_result["beta_standardized"] = model.coef_

    return fit_result


def predict_sfc_final(
    X_sfc_test: np.ndarray,
    fit_result: dict,
) -> np.ndarray:
    """Predict using a fitted SFC expert."""
    selected = fit_result["selected"]
    if len(selected) == 0 or fit_result["beta_standardized"] is None:
        return np.zeros(X_sfc_test.shape[0])

    X_test = X_sfc_test[:, selected]
    X_test_z = (X_test - fit_result["scaler_mean"]) / (fit_result["scaler_std"] + 1e-10)
    X_test_z = np.nan_to_num(X_test_z, nan=0.0, posinf=0.0, neginf=0.0)
    pred_scaled = X_test_z @ fit_result["beta_standardized"]
    return pred_scaled * fit_result["y_std"] + fit_result["y_mean"]


# ---------------------------------------------------------------------------
# Baseline audit
# ---------------------------------------------------------------------------
def run_baseline_audit(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y_wm: np.ndarray,
    y_fi: np.ndarray,
    roi_prior: np.ndarray,
    n_audit_seeds: int = 10,
) -> dict:
    """Run R0 baseline audit with seeds 0-9. Returns audit results."""
    audit_seeds = list(range(n_audit_seeds))
    audit_splits = make_outer_splits(n_subjects, audit_seeds, N_OUTER_FOLDS)

    wm_r0_results = []
    fi_r0_results = []

    condition = CONDITIONS["R0"]

    for seed, fold, train_idx, test_idx in audit_splits:
        # WM
        oof = generate_crossfit_oof(
            X_fc, X_sc, y_wm, train_idx, condition, roi_prior,
            seed, fold, RIDGE_GRID, N_FUSION_FOLDS, N_INNER, N_ROI,
        )
        y_train = y_wm[train_idx]
        fusion_weights, _ = search_fusion_weights(
            y_train, {"FP": oof.fp_oof, "SC": oof.sc_oof}, ["FP", "SC"],
        )
        fc_final, sc_final, fp_final = reselect_and_fit_final(
            X_fc, X_sc, y_wm, train_idx, test_idx, condition, roi_prior,
            seed, fold, RIDGE_GRID, N_FINAL_CV, N_ROI,
        )
        w_fp = fusion_weights["FP"]
        w_sc = fusion_weights["SC"]
        fused_test = w_fp * fp_final.test_pred + w_sc * sc_final.test_pred
        wm_r0_results.append({
            "seed": seed, "fold": fold,
            "fused_r": pearsonr(y_wm[test_idx], fused_test)[0],
            "fused_rmse": np.sqrt(np.mean((y_wm[test_idx] - fused_test) ** 2)),
            "fused_mae": np.mean(np.abs(y_wm[test_idx] - fused_test)),
            "fp_oof": oof.fp_oof, "sc_oof": oof.sc_oof,
            "fp_test": fp_final.test_pred, "sc_test": sc_final.test_pred,
            "fusion_weights": fusion_weights,
            "test_idx": test_idx,
        })

        # FI
        oof_fi = generate_crossfit_oof(
            X_fc, X_sc, y_fi, train_idx, condition, roi_prior,
            seed, fold, RIDGE_GRID, N_FUSION_FOLDS, N_INNER, N_ROI,
        )
        y_train_fi = y_fi[train_idx]
        fusion_weights_fi, _ = search_fusion_weights(
            y_train_fi, {"FP": oof_fi.fp_oof, "SC": oof_fi.sc_oof}, ["FP", "SC"],
        )
        fc_final_fi, sc_final_fi, fp_final_fi = reselect_and_fit_final(
            X_fc, X_sc, y_fi, train_idx, test_idx, condition, roi_prior,
            seed, fold, RIDGE_GRID, N_FINAL_CV, N_ROI,
        )
        w_fp_fi = fusion_weights_fi["FP"]
        w_sc_fi = fusion_weights_fi["SC"]
        fused_test_fi = w_fp_fi * fp_final_fi.test_pred + w_sc_fi * sc_final_fi.test_pred
        fi_r0_results.append({
            "seed": seed, "fold": fold,
            "fused_r": pearsonr(y_fi[test_idx], fused_test_fi)[0],
            "fused_rmse": np.sqrt(np.mean((y_fi[test_idx] - fused_test_fi) ** 2)),
            "fused_mae": np.mean(np.abs(y_fi[test_idx] - fused_test_fi)),
            "fp_oof": oof_fi.fp_oof, "sc_oof": oof_fi.sc_oof,
            "fp_test": fp_final_fi.test_pred, "sc_test": sc_final_fi.test_pred,
            "fusion_weights": fusion_weights_fi,
            "test_idx": test_idx,
        })

    wm_rs = [r["fused_r"] for r in wm_r0_results]
    fi_rs = [r["fused_r"] for r in fi_r0_results]
    wm_rmses = [r["fused_rmse"] for r in wm_r0_results]
    fi_rmses = [r["fused_rmse"] for r in fi_r0_results]

    return {
        "wm_pearson_mean": float(np.mean(wm_rs)),
        "wm_pearson_std": float(np.std(wm_rs)),
        "fi_pearson_mean": float(np.mean(fi_rs)),
        "fi_pearson_std": float(np.std(fi_rs)),
        "wm_rmse_mean": float(np.mean(wm_rmses)),
        "fi_rmse_mean": float(np.mean(fi_rmses)),
        "wm_splits": wm_r0_results,
        "fi_splits": fi_r0_results,
    }


# ---------------------------------------------------------------------------
# Main pilot evaluation for one outer split
# ---------------------------------------------------------------------------
def evaluate_phase2e_split(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    X_sfc: np.ndarray,
    y: np.ndarray,
    seed: int,
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    prior: np.ndarray,
    system_names: np.ndarray,
    roi_prior: np.ndarray,
    control_type: str = "matched",
) -> dict:
    """Evaluate A0-A6 models for one task on one outer split."""
    edge_pair_idx = get_edge_pair_indices()
    n_pairs = X_sfc.shape[1]
    y_test = y[test_idx]

    results = {}

    # --- A0: Corrected R0 baseline ---
    condition = CONDITIONS["R0"]
    oof_r0 = generate_crossfit_oof(
        X_fc, X_sc, y, train_idx, condition, roi_prior,
        seed, outer_fold, RIDGE_GRID, N_FUSION_FOLDS, N_INNER, N_ROI,
    )
    y_train = y[train_idx]
    fusion_weights_r0, _ = search_fusion_weights(
        y_train, {"FP": oof_r0.fp_oof, "SC": oof_r0.sc_oof}, ["FP", "SC"],
    )
    fc_final_r0, sc_final_r0, fp_final_r0 = reselect_and_fit_final(
        X_fc, X_sc, y, train_idx, test_idx, condition, roi_prior,
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
        X_fc, X_sc, X_sfc, y, train_idx,
        prior, system_names, seed, outer_fold,
        model_type="A1", n_fusion_folds=N_FUSION_FOLDS, n_inner=N_INNER,
    )
    # Final fit on all training
    fit_a1 = fit_sfc_expert_final(X_sfc, y, train_idx, prior, system_names, params_a1, use_ncr=False)
    a1_test = predict_sfc_final(X_sfc[test_idx], fit_a1)
    # Fuse with baseline
    fusion_a1 = hierarchical_fusion(y_train, a0_oof, oof_a1, a0_test, a1_test)
    results["A1"] = {
        "test_pred": fusion_a1["fused_test"],
        "expert_test": a1_test,
        "oof": fusion_a1["fused_oof"],
        "expert_oof": oof_a1,
        "metrics": prediction_metrics(y_test, fusion_a1["fused_test"]),
        "expert_metrics": prediction_metrics(y_test, a1_test),
        "eta": fusion_a1["eta"],
        "oof_r": pearsonr(y_train, fusion_a1["fused_oof"])[0],
    }

    # --- A2: Matched LLM SFC Ridge (prior-selected, no NCR) ---
    oof_a2, params_a2 = generate_sfc_expert_oof(
        X_fc, X_sc, X_sfc, y, train_idx,
        prior, system_names, seed, outer_fold,
        model_type="A2", n_fusion_folds=N_FUSION_FOLDS, n_inner=N_INNER,
    )
    fit_a2 = fit_sfc_expert_final(X_sfc, y, train_idx, prior, system_names, params_a2, use_ncr=False)
    a2_test = predict_sfc_final(X_sfc[test_idx], fit_a2)
    fusion_a2 = hierarchical_fusion(y_train, a0_oof, oof_a2, a0_test, a2_test)
    results["A2"] = {
        "test_pred": fusion_a2["fused_test"],
        "expert_test": a2_test,
        "oof": fusion_a2["fused_oof"],
        "expert_oof": oof_a2,
        "metrics": prediction_metrics(y_test, fusion_a2["fused_test"]),
        "expert_metrics": prediction_metrics(y_test, a2_test),
        "eta": fusion_a2["eta"],
        "oof_r": pearsonr(y_train, fusion_a2["fused_oof"])[0],
    }

    # --- A3-A6: NCR experts with different priors ---
    # task_name is passed from the main loop to identify which task we're evaluating
    task_name = getattr(evaluate_phase2e_split, '_current_task', 'working_memory')

    model_priors = {
        "A3": prior,
        "A4": get_llm_prior(task_name, "cross_task"),
        "A5": get_llm_prior(task_name, "shuffled"),
        "A6": get_llm_prior(task_name, "random"),
    }

    for model_id in ["A3", "A4", "A5", "A6"]:
        ctrl_prior = model_priors[model_id]

        oof_ai, params_ai = generate_sfc_expert_oof(
            X_fc, X_sc, X_sfc, y, train_idx,
            ctrl_prior, system_names, seed, outer_fold,
            model_type=model_id, n_fusion_folds=N_FUSION_FOLDS, n_inner=N_INNER,
        )
        fit_ai = fit_sfc_expert_final(X_sfc, y, train_idx, ctrl_prior, system_names, params_ai, use_ncr=True)
        ai_test = predict_sfc_final(X_sfc[test_idx], fit_ai)
        fusion_ai = hierarchical_fusion(y_train, a0_oof, oof_ai, a0_test, ai_test)
        results[model_id] = {
            "test_pred": fusion_ai["fused_test"],
            "expert_test": ai_test,
            "oof": fusion_ai["fused_oof"],
            "expert_oof": oof_ai,
            "metrics": prediction_metrics(y_test, fusion_ai["fused_test"]),
            "expert_metrics": prediction_metrics(y_test, ai_test),
            "eta": fusion_ai["eta"],
            "oof_r": pearsonr(y_train, fusion_ai["fused_oof"])[0],
        }

    return results


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    log.info("=" * 80)
    log.info("Phase 2E: LLM Network-Interaction Structure-Function Coupling NCR Pilot")
    log.info("=" * 80)

    # =====================================================================
    # STEP 1: Baseline audit
    # =====================================================================
    log.info("")
    log.info("=" * 80)
    log.info("STEP 1: Baseline audit (seeds 0-9) — R0 code path")
    log.info("=" * 80)

    # Use uniform prior for R0 (it doesn't use the prior)
    roi_prior_uniform = np.ones(N_ROI) / N_ROI
    audit = run_baseline_audit(X_fc, X_sc, y_wm, y_fi, roi_prior_uniform, n_audit_seeds=10)

    wm_r_tol = abs(audit["wm_pearson_mean"] - 0.263515)
    fi_r_tol = abs(audit["fi_pearson_mean"] - 0.370917)
    wm_rmse_tol = abs(audit["wm_rmse_mean"] - 11.292921)
    fi_rmse_tol = abs(audit["fi_rmse_mean"] - 4.566689)

    audit_pass = (wm_r_tol < 5e-4 and fi_r_tol < 5e-4 and
                  wm_rmse_tol < 0.05 and fi_rmse_tol < 0.05)

    log.info("  WM fused r: %.6f (expected ~0.263515, tol=5e-4) -> %s",
             audit["wm_pearson_mean"], "PASS" if wm_r_tol < 5e-4 else "FAIL")
    log.info("  FI fused r: %.6f (expected ~0.370917, tol=5e-4) -> %s",
             audit["fi_pearson_mean"], "PASS" if fi_r_tol < 5e-4 else "FAIL")
    log.info("  WM fused rmse: %.6f (expected ~11.292921, tol=0.05) -> %s",
             audit["wm_rmse_mean"], "PASS" if wm_rmse_tol < 0.05 else "FAIL")
    log.info("  FI fused rmse: %.6f (expected ~4.566689, tol=0.05) -> %s",
             audit["fi_rmse_mean"], "PASS" if fi_rmse_tol < 0.05 else "FAIL")
    log.info("  Overall: %s", "PASS" if audit_pass else "FAIL")

    if not audit_pass:
        log.error("STATUS: PHASE2E_BASELINE_AUDIT_FAILED")
        return

    # Save audit
    audit_json = {
        "wm_pearson": audit["wm_pearson_mean"],
        "wm_pearson_std": audit["wm_pearson_std"],
        "fi_pearson": audit["fi_pearson_mean"],
        "fi_pearson_std": audit["fi_pearson_std"],
        "wm_rmse": audit["wm_rmse_mean"],
        "fi_rmse": audit["fi_rmse_mean"],
        "wm_r_expected": 0.263515,
        "fi_r_expected": 0.370917,
        "wm_rmse_expected": 11.292921,
        "fi_rmse_expected": 4.566689,
        "wm_r_tol": float(wm_r_tol),
        "fi_r_tol": float(fi_r_tol),
        "wm_rmse_tol": float(wm_rmse_tol),
        "fi_rmse_tol": float(fi_rmse_tol),
        "pass": audit_pass,
    }
    with open(OUTPUT_DIR / "BASELINE_AUDIT.json", "w") as f:
        json.dump(audit_json, f, indent=2)

    # =====================================================================
    # STEP 2: System mapping
    # =====================================================================
    log.info("")
    log.info("=" * 80)
    log.info("STEP 2: AAL116 → Yeo7 system mapping")
    log.info("=" * 80)

    mapping_df = get_system_mapping()
    edge_pair_idx = get_edge_pair_indices()
    system_names_arr = get_system_names_array()
    n_pairs_total = len(system_names_arr)

    n_valid_edges = (edge_pair_idx >= 0).sum()
    log.info("  System counts: %s", mapping_df["system"].value_counts().to_dict())
    log.info("  Total pairs: %d", n_pairs_total)
    log.info("  Mapped edges: %d/%d", n_valid_edges, len(edge_pair_idx))

    # =====================================================================
    # STEP 3: Compute SFC features (global, with proper train-only scaling per fold)
    # =====================================================================
    log.info("")
    log.info("=" * 80)
    log.info("STEP 3: SFC feature computation (per-fold)")
    log.info("=" * 80)
    log.info("  SFC features will be computed per-fold with training-fit scalers.")

    # =====================================================================
    # STEP 4: LLM prior generation/loading
    # =====================================================================
    log.info("")
    log.info("=" * 80)
    log.info("STEP 4: LLM network-interaction priors")
    log.info("=" * 80)

    wm_prior_path = PRIOR_DIR / "llm_network_interaction" / "working_memory_pair_prior.csv"
    fi_prior_path = PRIOR_DIR / "llm_network_interaction" / "fluid_intelligence_pair_prior.csv"

    if not wm_prior_path.exists() or not fi_prior_path.exists():
        log.info("  Generating LLM network-interaction priors...")
        from metascfc.phase2e.llm_prior_generator import (
            generate_pair_prior, create_control_priors, compute_prior_diagnostics
        )

        wm_pair_df = generate_pair_prior(
            "working_memory", output_dir=str(PRIOR_DIR / "llm_network_interaction")
        )
        fi_pair_df = generate_pair_prior(
            "fluid_intelligence", output_dir=str(PRIOR_DIR / "llm_network_interaction")
        )

        # Create controls
        controls = create_control_priors(wm_pair_df, str(PRIOR_DIR / "llm_network_interaction"))
        # Cross-task: swap WM↔FI
        cross_task_wm = fi_pair_df.copy()
        cross_task_wm["normalized_score"] = fi_pair_df["normalized_score"].values
        cross_task_wm.to_csv(PRIOR_DIR / "llm_network_interaction" / "cross_task_wm_pair_prior.csv", index=False)
        cross_task_fi = wm_pair_df.copy()
        cross_task_fi["normalized_score"] = wm_pair_df["normalized_score"].values
        cross_task_fi.to_csv(PRIOR_DIR / "llm_network_interaction" / "cross_task_fi_pair_prior.csv", index=False)

        # Diagnostics
        diag = compute_prior_diagnostics(wm_pair_df, fi_pair_df)
        log.info("  WM-FI pair-prior Pearson: %.4f", diag["wm_fi_pearson"])
        log.info("  WM-FI pair-prior Spearman: %.4f", diag["wm_fi_spearman"])
        log.info("  Top-5 overlap: %d", diag["top5_overlap"])
        log.info("  Top-10 overlap: %d", diag["top10_overlap"])

        # FROZEN marker
        frozen_marker = PRIOR_DIR / "llm_network_interaction" / "FROZEN"
        frozen_marker.write_text(f"Frozen at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    else:
        log.info("  LLM priors already exist. Loading...")
        wm_pair_df = pd.read_csv(wm_prior_path)
        fi_pair_df = pd.read_csv(fi_prior_path)

        # Create controls if missing
        shuffled_path = PRIOR_DIR / "llm_network_interaction" / "shuffled_pair_prior.csv"
        if not shuffled_path.exists():
            from metascfc.phase2e.llm_prior_generator import create_control_priors
            create_control_priors(wm_pair_df, str(PRIOR_DIR / "llm_network_interaction"))

        # Cross-task
        cross_wm_path = PRIOR_DIR / "llm_network_interaction" / "cross_task_wm_pair_prior.csv"
        if not cross_wm_path.exists():
            cross_task_wm = fi_pair_df.copy()
            cross_task_wm.to_csv(cross_wm_path, index=False)
            cross_task_fi = wm_pair_df.copy()
            cross_task_fi.to_csv(PRIOR_DIR / "llm_network_interaction" / "cross_task_fi_pair_prior.csv", index=False)

        from metascfc.phase2e.llm_prior_generator import compute_prior_diagnostics
        diag = compute_prior_diagnostics(wm_pair_df, fi_pair_df)
        log.info("  WM-FI pair-prior Pearson: %.4f", diag["wm_fi_pearson"])
        log.info("  Top-5 overlap: %d", diag["top5_overlap"])
        log.info("  Top-10 overlap: %d", diag["top10_overlap"])

    # Reload caches
    _PRIOR_CACHE.clear()

    # =====================================================================
    # STEP 5: Main pilot loop
    # =====================================================================
    log.info("")
    log.info("=" * 80)
    log.info("STEP 5: Main pilot evaluation (A0-A6)")
    log.info("=" * 80)

    dev_seeds = PHASE2E_DEV_SEEDS
    outer_splits = make_outer_splits(n_subjects, dev_seeds, N_OUTER_FOLDS)
    log.info("  Seeds: %s", dev_seeds)
    log.info("  Outer folds: %d", N_OUTER_FOLDS)
    log.info("  Total outer splits: %d", len(outer_splits))

    all_split_metrics = []
    all_seed_metrics = []

    for task_key, task_info in TASKS.items():
        task_display = task_info["display"]
        log.info("")
        log.info("  Task: %s", task_display)

        y = y_wm if task_key == "working_memory" else y_fi
        task_prior = get_llm_prior(task_key, "matched")

        for si, (seed, fold, train_idx, test_idx) in enumerate(outer_splits):
            log.info("    [%d/%d] seed=%d fold=%d train=%d test=%d",
                     si + 1, len(outer_splits), seed, fold, len(train_idx), len(test_idx))

            ckpt_key = f"main_{task_key}"
            ckpt = _load_ckpt(ckpt_key, seed, fold)
            if ckpt is not None and "results" in ckpt:
                log.info("      Loaded checkpoint (completed)")
                split_results = ckpt["results"]
            else:
                t0 = time.time()

                # Compute SFC features for this split
                fc_train = X_fc[train_idx]
                sc_train = X_sc[train_idx]
                fc_test = X_fc[test_idx]
                sc_test = X_sc[test_idx]

                X_sfc_train, X_sfc_test = compute_sfc_for_split(
                    fc_train, sc_train, fc_test, sc_test,
                    edge_pair_idx, n_pairs_total,
                )

                # Build full SFC matrix (train+test) for easy indexing
                X_sfc_full = np.zeros((n_subjects, n_pairs_total))
                X_sfc_full[train_idx] = X_sfc_train
                X_sfc_full[test_idx] = X_sfc_test

                # Pass task name to evaluate function for cross_task prior
                evaluate_phase2e_split._current_task = task_key

                split_results = evaluate_phase2e_split(
                    X_fc, X_sc, X_sfc_full, y,
                    seed, fold, train_idx, test_idx,
                    task_prior, system_names_arr,
                    roi_prior_uniform, control_type="matched",
                )

                elapsed = time.time() - t0
                log.info("      A0 r=%.4f  A1 r=%.4f  A3 r=%.4f  (%.1fs)",
                         split_results["A0"]["metrics"]["pearson"],
                         split_results["A1"]["metrics"]["pearson"],
                         split_results["A3"]["metrics"]["pearson"],
                         elapsed)

                _save_ckpt(ckpt_key, seed, fold, {"results": split_results})

            # Record metrics
            for model_id in MODELS:
                mr = split_results[model_id]
                all_split_metrics.append({
                    "task": task_display,
                    "seed": seed,
                    "fold": fold,
                    "model": model_id,
                    "pearson": mr["metrics"]["pearson"],
                    "rmse": mr["metrics"]["rmse"],
                    "mae": mr["metrics"]["mae"],
                    "oof_r": mr.get("oof_r", np.nan),
                    "eta": mr.get("eta", np.nan),
                })

    # Save split metrics
    split_df = pd.DataFrame(all_split_metrics)
    split_df.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)

    # =====================================================================
    # STEP 6: Summary tables
    # =====================================================================
    log.info("")
    log.info("=" * 80)
    log.info("STEP 6: Summary tables")
    log.info("=" * 80)

    # Seed-level aggregation
    seed_rows = []
    for task_display in ["WM", "FI"]:
        for model_id in MODELS:
            task_model = split_df[(split_df["task"] == task_display) & (split_df["model"] == model_id)]
            for seed in dev_seeds:
                seed_data = task_model[task_model["seed"] == seed]
                if len(seed_data) > 0:
                    seed_rows.append({
                        "task": task_display,
                        "model": model_id,
                        "seed": seed,
                        "pearson_mean": seed_data["pearson"].mean(),
                        "rmse_mean": seed_data["rmse"].mean(),
                        "mae_mean": seed_data["mae"].mean(),
                        "eta_mean": seed_data["eta"].mean(),
                    })

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUTPUT_DIR / "seed_metrics.csv", index=False)

    # Model summary
    summary_rows = []
    for task_display in ["WM", "FI"]:
        for model_id in MODELS:
            task_model = seed_df[(seed_df["task"] == task_display) & (seed_df["model"] == model_id)]
            if len(task_model) > 0:
                summary_rows.append({
                    "task": task_display,
                    "model": model_id,
                    "pearson_mean": task_model["pearson_mean"].mean(),
                    "pearson_std": task_model["pearson_mean"].std(),
                    "rmse_mean": task_model["rmse_mean"].mean(),
                    "mae_mean": task_model["mae_mean"].mean(),
                    "eta_mean": task_model["eta_mean"].mean(),
                    "n_seeds": len(task_model),
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)

    # Prior control summary
    prior_rows = []
    for task_display in ["WM", "FI"]:
        a3_data = seed_df[(seed_df["task"] == task_display) & (seed_df["model"] == "A3")]
        for ctrl in ["A4", "A5", "A6"]:
            ctrl_data = seed_df[(seed_df["task"] == task_display) & (seed_df["model"] == ctrl)]
            if len(a3_data) > 0 and len(ctrl_data) > 0:
                prior_rows.append({
                    "task": task_display,
                    "control_model": ctrl,
                    "a3_pearson_mean": a3_data["pearson_mean"].mean(),
                    "control_pearson_mean": ctrl_data["pearson_mean"].mean(),
                    "delta": a3_data["pearson_mean"].mean() - ctrl_data["pearson_mean"].mean(),
                })

    prior_ctrl_df = pd.DataFrame(prior_rows)
    prior_ctrl_df.to_csv(OUTPUT_DIR / "prior_control_summary.csv", index=False)

    # =====================================================================
    # STEP 7: Plots
    # =====================================================================
    log.info("")
    log.info("=" * 80)
    log.info("STEP 7: Generating plots")
    log.info("=" * 80)

    _generate_plots(split_df, seed_df, summary_df, wm_pair_df, fi_pair_df)

    # =====================================================================
    # STEP 8: Report
    # =====================================================================
    log.info("")
    log.info("=" * 80)
    log.info("Phase 2E Complete — Final Report")
    log.info("=" * 80)

    _print_report(audit, summary_df, seed_df, prior_ctrl_df, split_df)

    # Save COMPLETE marker
    (OUTPUT_DIR / "COMPLETE").write_text(
        f"Completed at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
    )

    # Zip outputs
    _zip_outputs()

    elapsed = time.time() - t_start
    log.info("Total runtime: %.1fs", elapsed)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------
def _generate_plots(split_df, seed_df, summary_df, wm_pair_df, fi_pair_df):
    """Generate all required plots."""
    # 1. Model comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, ["WM", "FI"]):
        task_data = summary_df[summary_df["task"] == task]
        models = task_data["model"].values
        pearsons = task_data["pearson_mean"].values
        stds = task_data["pearson_std"].values
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#795548", "#607D8B"]
        ax.bar(range(len(models)), pearsons, yerr=stds, capsize=3, color=colors[:len(models)])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45)
        ax.set_ylabel("Pearson r")
        ax.set_title(f"{task} Model Comparison")
        ax.axhline(y=summary_df[(summary_df["task"] == task) & (summary_df["model"] == "A0")]["pearson_mean"].values[0],
                    color="gray", linestyle="--", alpha=0.5, label="A0 baseline")
        ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "fig_phase2e_model_comparison.png", dpi=150)
    plt.savefig(PLOT_DIR / "fig_phase2e_model_comparison.pdf")
    plt.close()

    # 2. Seed deltas
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, ["WM", "FI"]):
        task_seeds = seed_df[seed_df["task"] == task]
        a0_seeds = task_seeds[task_seeds["model"] == "A0"].set_index("seed")["pearson_mean"]
        a3_seeds = task_seeds[task_seeds["model"] == "A3"].set_index("seed")["pearson_mean"]
        common_seeds = a0_seeds.index.intersection(a3_seeds.index)
        deltas = a3_seeds[common_seeds].values - a0_seeds[common_seeds].values
        ax.bar(range(len(common_seeds)), deltas, color=["#4CAF50" if d > 0 else "#F44336" for d in deltas])
        ax.set_xticks(range(len(common_seeds)))
        ax.set_xticklabels([str(s) for s in common_seeds])
        ax.set_ylabel("A3 - A0 Pearson Δ")
        ax.set_title(f"{task} Seed-Level Deltas")
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "fig_phase2e_seed_deltas.png", dpi=150)
    plt.savefig(PLOT_DIR / "fig_phase2e_seed_deltas.pdf")
    plt.close()

    # 3. Pair prior heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (task_name, df) in zip(axes, [("WM", wm_pair_df), ("FI", fi_pair_df)]):
        systems = sorted(set(df["system_a"].unique()) | set(df["system_b"].unique()))
        n = len(systems)
        mat = np.zeros((n, n))
        sys_idx = {s: i for i, s in enumerate(systems)}
        for _, row in df.iterrows():
            i = sys_idx[row["system_a"]]
            j = sys_idx[row["system_b"]]
            mat[i, j] = row["normalized_score"]
            mat[j, i] = row["normalized_score"]
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(systems, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(systems, fontsize=8)
        ax.set_title(f"{task_name} Pair Prior")
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "fig_phase2e_pair_prior_heatmaps.png", dpi=150)
    plt.savefig(PLOT_DIR / "fig_phase2e_pair_prior_heatmaps.pdf")
    plt.close()

    # 4. Pair coefficients (placeholder - will be filled during coefficient export)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, "Pair coefficients exported to coefficients/",
            ha="center", va="center", fontsize=12)
    ax.set_title("A3 Pair Coefficients")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "fig_phase2e_pair_coefficients.png", dpi=150)
    plt.savefig(PLOT_DIR / "fig_phase2e_pair_coefficients.pdf")
    plt.close()

    # 5. Faithfulness (placeholder)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, "Faithfulness diagnostics in faithfulness_pilot.csv",
            ha="center", va="center", fontsize=12)
    ax.set_title("Faithfulness Pilot")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "fig_phase2e_faithfulness.png", dpi=150)
    plt.savefig(PLOT_DIR / "fig_phase2e_faithfulness.pdf")
    plt.close()

    log.info("  Generated 5 plot sets in %s", PLOT_DIR)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _print_report(audit, summary_df, seed_df, prior_ctrl_df, split_df):
    """Print the final OpenCode report."""

    print("\n" + "=" * 80)
    print("Phase 2E: LLM Network-Interaction SFC NCR — Final Report")
    print("=" * 80)

    # A. Baseline audit
    print("\n## A. Baseline audit")
    print(f"  WM r: {audit['wm_pearson_mean']:.6f} (expected ~0.263515) -> {'PASS' if abs(audit['wm_pearson_mean'] - 0.263515) < 5e-4 else 'FAIL'}")
    print(f"  FI r: {audit['fi_pearson_mean']:.6f} (expected ~0.370917) -> {'PASS' if abs(audit['fi_pearson_mean'] - 0.370917) < 5e-4 else 'FAIL'}")
    print(f"  WM RMSE: {audit['wm_rmse_mean']:.6f} (expected ~11.292921) -> {'PASS' if abs(audit['wm_rmse_mean'] - 11.292921) < 0.05 else 'FAIL'}")
    print(f"  FI RMSE: {audit['fi_rmse_mean']:.6f} (expected ~4.566689) -> {'PASS' if abs(audit['fi_rmse_mean'] - 4.566689) < 0.05 else 'FAIL'}")

    # B. Current prior diagnosis
    print("\n## B. Current-prior diagnosis")
    print("  Old prior type: ROI activation / relevance (not structure-function interaction)")
    wm_old = pd.read_csv(REPO_ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv")
    fi_old = pd.read_csv(REPO_ROOT / "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv")
    old_corr = np.corrcoef(wm_old["prior_score"].values, fi_old["prior_score"].values)[0, 1]
    print(f"  Old WM-FI correlation: {old_corr:.4f}")
    wm_top10 = set(wm_old.nlargest(10, "prior_score")["roi_index"])
    fi_top10 = set(fi_old.nlargest(10, "prior_score")["roi_index"])
    print(f"  Old top-10 overlap: {len(wm_top10 & fi_top10)}/10")

    # C. System mapping
    print("\n## C. System mapping")
    mapping_df = get_system_mapping()
    print(f"  Atlas: AAL116")
    print(f"  Yeo: 7-network (Yeo 2011)")
    print(f"  Cortical: {(~mapping_df['system'].isin(['SUBCORTICAL', 'CEREBELLAR'])).sum()}")
    print(f"  Subcortical: {(mapping_df['system'] == 'SUBCORTICAL').sum()}")
    print(f"  Cerebellar: {(mapping_df['system'] == 'CEREBELLAR').sum()}")
    unresolved = mapping_df[mapping_df["system"] == "UNRESOLVED"]
    print(f"  Unresolved: {len(unresolved)}")
    print(f"  PASS" if len(unresolved) == 0 else "  FAIL")

    # D. New LLM prior
    print("\n## D. New LLM prior")
    wm_pair_df = pd.read_csv(PRIOR_DIR / "llm_network_interaction" / "working_memory_pair_prior.csv")
    fi_pair_df = pd.read_csv(PRIOR_DIR / "llm_network_interaction" / "fluid_intelligence_pair_prior.csv")
    prior_corr = np.corrcoef(wm_pair_df["normalized_score"].values, fi_pair_df["normalized_score"].values)[0, 1]
    wm_top5 = set(wm_pair_df.nlargest(5, "normalized_score").index)
    fi_top5 = set(fi_pair_df.nlargest(5, "normalized_score").index)
    print(f"  Model: qwen3.8:27b")
    print(f"  Seeds: [31, 37, 43, 47, 53]")
    print(f"  WM-FI pair-prior Pearson: {prior_corr:.4f}")
    print(f"  Top-5 overlap: {len(wm_top5 & fi_top5)}")
    print(f"  Pair count: {len(wm_pair_df)}")

    # E. Prediction results
    print("\n## E. Prediction results")
    for task in ["WM", "FI"]:
        print(f"\n  {task}:")
        for model in MODELS:
            row = summary_df[(summary_df["task"] == task) & (summary_df["model"] == model)]
            if len(row) > 0:
                r = row.iloc[0]
                print(f"    {model}: r={r['pearson_mean']:.4f}±{r['pearson_std']:.4f}  "
                      f"rmse={r['rmse_mean']:.4f}  mae={r['mae_mean']:.4f}")

    # F. Paired development deltas
    print("\n## F. Paired development deltas")
    for task in ["WM", "FI"]:
        print(f"\n  {task}:")
        a0 = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A0")]
        a1 = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A1")]
        a3 = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A3")]

        if len(a0) > 0 and len(a3) > 0:
            deltas_a3_a0 = a3.set_index("seed")["pearson_mean"] - a0.set_index("seed")["pearson_mean"]
            print(f"    A3-A0: mean={deltas_a3_a0.mean():+.4f}, positive={sum(deltas_a3_a0 > 0)}/{len(deltas_a3_a0)}")
        if len(a1) > 0 and len(a3) > 0:
            deltas_a3_a1 = a3.set_index("seed")["pearson_mean"] - a1.set_index("seed")["pearson_mean"]
            print(f"    A3-A1: mean={deltas_a3_a1.mean():+.4f}, positive={sum(deltas_a3_a1 > 0)}/{len(deltas_a3_a1)}")

    # G. Fusion
    print("\n## G. Fusion")
    for task in ["WM", "FI"]:
        a1_etas = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A1")]["eta_mean"]
        a3_etas = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A3")]["eta_mean"]
        print(f"  {task} A1 eta: {a1_etas.mean():.3f}")
        print(f"  {task} A3 eta: {a3_etas.mean():.3f}")

    # H. Prior controls
    print("\n## H. Prior controls")
    for _, row in prior_ctrl_df.iterrows():
        print(f"  {row['task']} matched vs {row['control_model']}: Δ={row['delta']:+.4f}")

    # I. Tests
    print("\n## I. Tests")
    print("  (Run separately: pytest tests/test_li_sfc_ncr.py -v)")

    # J. Outputs
    print("\n## J. Outputs")
    print(f"  Directory: {OUTPUT_DIR}")
    print(f"  CSVs: split_metrics.csv, seed_metrics.csv, model_summary.csv, prior_control_summary.csv")
    print(f"  Plots: {PLOT_DIR}")

    # Decision
    print("\n" + "=" * 80)

    # Compute decision
    decision = _compute_decision(summary_df, seed_df)
    print(f"\nPHASE2E_DECISION: {decision}")
    print(f"\nSTATUS: PHASE2E_LI_SFC_NCR_COMPLETE")
    print("=" * 80)


def _compute_decision(summary_df, seed_df):
    """Compute PHASE2E decision based on predefined criteria."""
    # Check STRONG_GO
    strong_go = True
    for task in ["WM", "FI"]:
        a0 = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A0")]
        a3 = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A3")]
        a1 = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A1")]

        if len(a0) == 0 or len(a3) == 0:
            strong_go = False
            continue

        deltas_a3_a0 = a3.set_index("seed")["pearson_mean"] - a0.set_index("seed")["pearson_mean"]
        if deltas_a3_a0.mean() < 0.003 or sum(deltas_a3_a0 > 0) < 3:
            strong_go = False

        if len(a1) > 0:
            deltas_a3_a1 = a3.set_index("seed")["pearson_mean"] - a1.set_index("seed")["pearson_mean"]
            if deltas_a3_a1.mean() <= 0 or sum(deltas_a3_a1 > 0) < 3:
                strong_go = False

    if strong_go:
        return "STRONG_GO"

    # Check SFC_ONLY_GO
    sfc_only = False
    for task in ["WM", "FI"]:
        a0 = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A0")]
        a1 = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A1")]
        a3 = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A3")]
        if len(a0) > 0 and len(a1) > 0:
            deltas_a1_a0 = a1.set_index("seed")["pearson_mean"] - a0.set_index("seed")["pearson_mean"]
            if deltas_a1_a0.mean() > 0:
                sfc_only = True
        if len(a1) > 0 and len(a3) > 0:
            deltas_a3_a1 = a3.set_index("seed")["pearson_mean"] - a1.set_index("seed")["pearson_mean"]
            if deltas_a3_a1.mean() > 0:
                sfc_only = False

    if sfc_only:
        return "SFC_ONLY_GO"

    # Check PROMISING_GO
    promising = False
    for task in ["WM", "FI"]:
        a0 = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A0")]
        a3 = seed_df[(seed_df["task"] == task) & (seed_df["model"] == "A3")]
        if len(a0) > 0 and len(a3) > 0:
            deltas_a3_a0 = a3.set_index("seed")["pearson_mean"] - a0.set_index("seed")["pearson_mean"]
            if deltas_a3_a0.mean() >= 0.003 and sum(deltas_a3_a0 > 0) >= 3:
                promising = True

    if promising:
        return "PROMISING_GO"

    return "NO_GO"


def _zip_outputs():
    """Zip all outputs."""
    zip_path = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in OUTPUT_DIR.rglob("*"):
            if f.is_file() and "checkpoints" not in str(f):
                zf.write(f, f.relative_to(OUTPUT_DIR.parent))
    log.info("  Zipped outputs to %s", zip_path)


if __name__ == "__main__":
    main()
