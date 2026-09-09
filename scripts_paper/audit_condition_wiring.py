#!/usr/bin/env python3
"""Condition-wiring audit for PALF cross-fitted ablation.

Traces the exact code path for R0-R3 conditions and verifies that
anisotropic D and network L_p actually reach the solver.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from metascfc.experiments.palf_crossfit_ablation import (
    CONDITIONS,
    build_condition_cache,
    GAMMA_FIXED,
    DIAGONAL_EPSILON,
    TOP_K,
    N_ROI,
    N_EDGE,
    C_SCALE,
    _select_fc_only_params,
    _fit_and_predict_fc_only_on_subset,
    _select_alpha_ridge,
    _fit_and_predict_ridge_on_subset,
    make_inner_selection_folds,
    RIDGE_GRID,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    build_msancr_cache,
    _solve_msancr_kernel,
    _predict_msancr,
    compute_diagonal_penalty,
    lift_roi_to_edge,
)

OUTPUT_BASE = REPO_ROOT / "outputs/iclr/palf_crossfit_ablation_v1"


def arr_hash(a):
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def load_checkpoint(task):
    with open(OUTPUT_BASE / task / "checkpoint.pkl", "rb") as f:
        return pickle.load(f)


def audit_one_split():
    """Audit a single split: WM seed=0 fold=0, all four conditions."""
    # Load data
    fc_mats = np.load(REPO_ROOT / "inputs/dataset_FC/FC_all.npy")
    sc_mats = np.load(REPO_ROOT / "inputs/dataset_SC/SC_all.npy")
    iu = np.triu_indices(116, k=1)
    X_fc = fc_mats[:, iu[0], iu[1]].astype(np.float64)
    X_sc = sc_mats[:, iu[0], iu[1]].astype(np.float64)
    y_wm = np.load(REPO_ROOT / "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy").astype(np.float64)

    # Load prior
    prior_df = pd.read_csv(REPO_ROOT / "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv")
    roi_prior = prior_df["prior_score"].values.astype(np.float64)

    # Get the split indices from checkpoint
    result = load_checkpoint("working_memory")
    # Find seed=0, fold=0 for each condition
    splits_by_cond = {}
    for s in result.splits:
        if s.seed == 0 and s.outer_fold == 0:
            splits_by_cond[s.condition_id] = s

    print("=" * 80)
    print("CONDITION-WIRING AUDIT: WM seed=0, fold=0")
    print("=" * 80)

    audit_results = {}

    for cond_id in ["R0", "R1", "R2", "R3"]:
        condition = CONDITIONS[cond_id]
        split = splits_by_cond[cond_id]

        print(f"\n{'─' * 60}")
        print(f"CONDITION: {cond_id} — {condition.name}")
        print(f"{'─' * 60}")

        # Build cache (same path as OOF and final refit)
        cache = build_condition_cache(roi_prior, condition)

        # 1. Verify condition parameters
        use_aniso = condition.use_anisotropy
        use_network = condition.use_network
        gamma = GAMMA_FIXED if use_aniso else 0.0
        epsilon = DIAGONAL_EPSILON if use_aniso else 1.0
        lambda_l_grid = condition.lambda_l_grid

        print(f"  use_anisotropy: {use_aniso}")
        print(f"  use_network:    {use_network}")
        print(f"  gamma:          {gamma}")
        print(f"  epsilon:        {epsilon}")
        print(f"  lambda_l_grid:  {lambda_l_grid}")

        # 2. Compute D and q
        q = lift_roi_to_edge(roi_prior, n_rois=N_ROI, rule="prod")
        D = compute_diagonal_penalty(q, gamma=gamma, epsilon=epsilon)
        D_inv_sqrt = np.sqrt(1.0 / D)

        print(f"\n  q:  min={q.min():.6f} max={q.max():.6f} mean={q.mean():.6f} std={q.std():.6f}")
        print(f"  D:  min={D.min():.6f} max={D.max():.6f} mean={D.mean():.6f} std={D.std():.6f}")
        print(f"  ||D - I||_F:    {np.linalg.norm(D - 1.0):.10f}")
        print(f"  D[0:20]:        {D[:20]}")

        # 3. Cache contents
        print(f"\n  Cache D_inv_sqrt:  min={cache.D_inv_sqrt.min():.6f} max={cache.D_inv_sqrt.max():.6f}")
        print(f"  Cache n_active:    {cache.n_active}")
        print(f"  Cache active_indices shape: {cache.active_indices.shape}")
        print(f"  generalized_mu:   {cache.generalized_mu[:5]}...")

        # 4. Laplacian
        if hasattr(cache, 'active_laplacian') and cache.active_laplacian is not None:
            L_norm = np.linalg.norm(cache.active_laplacian)
            print(f"  active_laplacian norm: {L_norm:.10f}")
        elif hasattr(cache, 'generalized_mu'):
            print(f"  generalized_mu:    min={cache.generalized_mu.min():.6e} max={cache.generalized_mu.max():.6e}")
        else:
            print(f"  (no Laplacian info available)")

        # 5. Selected hyperparameters from checkpoint
        fc_alpha = split.fc_final.selected_params.get("alpha")
        sc_alpha = split.sc_final.selected_params.get("alpha")
        fp_params = split.fp_final.selected_params
        print(f"\n  Selected FC alpha: {fc_alpha}")
        print(f"  Selected SC alpha: {sc_alpha}")
        print(f"  Selected FP lambda_fc: {fp_params.get('lambda_fc')}")
        print(f"  Selected FP lambda_l:  {fp_params.get('lambda_l')}")

        # 6. Prediction hashes
        print(f"\n  FC OOF hash:       {arr_hash(split.fc_oof)}")
        print(f"  SC OOF hash:       {arr_hash(split.sc_oof)}")
        print(f"  FP OOF hash:       {arr_hash(split.fp_oof)}")
        print(f"  FC test_pred hash: {arr_hash(split.fc_test_pred)}")
        print(f"  SC test_pred hash: {arr_hash(split.sc_test_pred)}")
        print(f"  FP test_pred hash: {arr_hash(split.fp_test_pred)}")
        print(f"  Fused test hash:   {arr_hash(split.fused_test_pred)}")
        print(f"  Equal-weight hash: {arr_hash(split.equal_weight_pred)}")

        # 7. Fusion weights
        print(f"  Fusion weights:    {split.fusion_weights}")

        # 8. Store for cross-condition comparison
        audit_results[cond_id] = {
            "use_anisotropy": use_aniso,
            "use_network": use_network,
            "gamma": gamma,
            "epsilon": epsilon,
            "q": q,
            "D": D,
            "D_inv_sqrt": cache.D_inv_sqrt,
            "active_laplacian_norm": np.linalg.norm(cache.active_laplacian) if cache.active_laplacian is not None else 0.0,
            "n_active": cache.n_active,
            "lambda_l_grid": list(lambda_l_grid),
            "fc_alpha": fc_alpha,
            "sc_alpha": sc_alpha,
            "fp_lambda_fc": fp_params.get("lambda_fc"),
            "fp_lambda_l": fp_params.get("lambda_l"),
            "fp_oof_hash": arr_hash(split.fp_oof),
            "fp_test_hash": arr_hash(split.fp_test_pred),
            "fused_test_hash": arr_hash(split.fused_test_pred),
            "fusion_weights": split.fusion_weights,
            "D_first20": D[:20],
        }

    # Cross-condition comparisons
    print(f"\n{'=' * 80}")
    print("CROSS-CONDITION COMPARISONS")
    print(f"{'=' * 80}")

    # R0 vs R1: D should differ, lambda_l same (both 0)
    d_diff_01 = np.linalg.norm(audit_results["R0"]["D"] - audit_results["R1"]["D"])
    print(f"\n  ||D_R0 - D_R1||_F = {d_diff_01:.10f}")
    print(f"  FP OOF  R0==R1: {audit_results['R0']['fp_oof_hash'] == audit_results['R1']['fp_oof_hash']}")
    print(f"  FP test R0==R1: {audit_results['R0']['fp_test_hash'] == audit_results['R1']['fp_test_hash']}")

    # R2 vs R3: D should differ, lambda_l same (both nonzero)
    d_diff_23 = np.linalg.norm(audit_results["R2"]["D"] - audit_results["R3"]["D"])
    print(f"\n  ||D_R2 - D_R3||_F = {d_diff_23:.10f}")
    print(f"  FP OOF  R2==R3: {audit_results['R2']['fp_oof_hash'] == audit_results['R3']['fp_oof_hash']}")
    print(f"  FP test R2==R3: {audit_results['R2']['fp_test_hash'] == audit_results['R3']['fp_test_hash']}")

    # R0 vs R2: lambda_l should differ (0 vs >0), D same
    print(f"\n  R0 lambda_l_grid: {audit_results['R0']['lambda_l_grid']}")
    print(f"  R2 lambda_l_grid: {audit_results['R2']['lambda_l_grid']}")
    print(f"  R0 FP lambda_l:   {audit_results['R0']['fp_lambda_l']}")
    print(f"  R2 FP lambda_l:   {audit_results['R2']['fp_lambda_l']}")
    print(f"  FP OOF  R0==R2: {audit_results['R0']['fp_oof_hash'] == audit_results['R2']['fp_oof_hash']}")
    print(f"  FP test R0==R2: {audit_results['R0']['fp_test_hash'] == audit_results['R2']['fp_test_hash']}")

    # R1 vs R3: lambda_l should differ, D same
    print(f"\n  R1 FP lambda_l:   {audit_results['R1']['fp_lambda_l']}")
    print(f"  R3 FP lambda_l:   {audit_results['R3']['fp_lambda_l']}")
    print(f"  FP OOF  R1==R3: {audit_results['R1']['fp_oof_hash'] == audit_results['R3']['fp_oof_hash']}")
    print(f"  FP test R1==R3: {audit_results['R1']['fp_test_hash'] == audit_results['R3']['fp_test_hash']}")

    # Check if any FP predictions are identical
    print(f"\n  ALL FP test hashes identical: "
          f"{len(set(audit_results[c]['fp_test_hash'] for c in ['R0','R1','R2','R3'])) == 1}")

    # Now test the actual solver with different D values
    print(f"\n{'=' * 80}")
    print("SOLVER-level verification")
    print(f"{'=' * 80}")

    # Use the R3 selected hyperparams on a small subset
    train_idx = splits_by_cond["R3"].train_idx
    test_idx = splits_by_cond["R3"].test_idx
    lambda_fc = audit_results["R3"]["fp_lambda_fc"]
    lambda_l = audit_results["R3"]["fp_lambda_l"]

    from sklearn.preprocessing import StandardScaler

    # Build R0 cache (D=I) and R3 cache (D=D(q;0.5))
    cache_r0 = build_condition_cache(roi_prior, CONDITIONS["R0"])
    cache_r3 = build_condition_cache(roi_prior, CONDITIONS["R3"])

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_fc[train_idx])
    X_test_z = scaler.transform(X_fc[test_idx])
    y_mean = float(y_wm[train_idx].mean())
    y_std = max(float(y_wm[train_idx].std()), 1e-8)
    y_train_z = (y_wm[train_idx] - y_mean) / y_std

    # Solve with R0 cache
    alpha_r0, _ = _solve_msancr_kernel(
        X_train_z, np.zeros_like(X_train_z), y_train_z, cache_r0,
        lambda_fc, 1.0, lambda_l, fc_only=True,
    )
    pred_r0 = _predict_msancr(
        X_test_z, np.zeros_like(X_test_z),
        X_train_z, np.zeros_like(X_train_z),
        alpha_r0, cache_r0, lambda_fc, 1.0, lambda_l, fc_only=True,
    ) * y_std + y_mean

    # Solve with R3 cache
    alpha_r3, _ = _solve_msancr_kernel(
        X_train_z, np.zeros_like(X_train_z), y_train_z, cache_r3,
        lambda_fc, 1.0, lambda_l, fc_only=True,
    )
    pred_r3 = _predict_msancr(
        X_test_z, np.zeros_like(X_test_z),
        X_train_z, np.zeros_like(X_train_z),
        alpha_r3, cache_r3, lambda_fc, 1.0, lambda_l, fc_only=True,
    ) * y_std + y_mean

    print(f"\n  With same lambda_fc={lambda_fc}, lambda_l={lambda_l}:")
    print(f"  R0 (D=I) pred[0:5]:       {pred_r0[:5]}")
    print(f"  R3 (D=D(q)) pred[0:5]:    {pred_r3[:5]}")
    print(f"  Max |pred_R0 - pred_R3|:   {np.max(np.abs(pred_r0 - pred_r3)):.2e}")
    print(f"  ||pred_R0 - pred_R3||_F:   {np.linalg.norm(pred_r0 - pred_r3):.2e}")
    print(f"  alpha R0 hash: {arr_hash(alpha_r0)}")
    print(f"  alpha R3 hash: {arr_hash(alpha_r3)}")
    print(f"  Alpha differ: {not np.allclose(alpha_r0, alpha_r3)}")

    # Check D_inv_sqrt from caches
    print(f"\n  R0 cache D_inv_sqrt[0:10]: {cache_r0.D_inv_sqrt[:10]}")
    print(f"  R3 cache D_inv_sqrt[0:10]: {cache_r3.D_inv_sqrt[:10]}")
    print(f"  D_inv_sqrt differ: {not np.allclose(cache_r0.D_inv_sqrt, cache_r3.D_inv_sqrt)}")

    # Save audit results
    audit_path = OUTPUT_BASE / "condition_wiring_audit.json"
    save_data = {}
    for c in ["R0", "R1", "R2", "R3"]:
        r = audit_results[c]
        save_data[c] = {
            "use_anisotropy": r["use_anisotropy"],
            "use_network": r["use_network"],
            "gamma": r["gamma"],
            "epsilon": r["epsilon"],
            "D_first20": r["D_first20"].tolist(),
            "fp_lambda_fc": r["fp_lambda_fc"],
            "fp_lambda_l": r["fp_lambda_l"],
            "fp_oof_hash": r["fp_oof_hash"],
            "fp_test_hash": r["fp_test_hash"],
            "fused_test_hash": r["fused_test_hash"],
            "fusion_weights": r["fusion_weights"],
        }
    save_data["cross_condition"] = {
        "D_R0_R1_diff": float(d_diff_01),
        "D_R2_R3_diff": float(d_diff_23),
        "FP_test_R0_R1_identical": audit_results["R0"]["fp_test_hash"] == audit_results["R1"]["fp_test_hash"],
        "FP_test_R2_R3_identical": audit_results["R2"]["fp_test_hash"] == audit_results["R3"]["fp_test_hash"],
        "all_FP_test_identical": len(set(audit_results[c]["fp_test_hash"] for c in ["R0","R1","R2","R3"])) == 1,
        "solver_R0_R3_max_diff": float(np.max(np.abs(pred_r0 - pred_r3))),
        "solver_R0_R3_alpha_diff": bool(not np.allclose(alpha_r0, alpha_r3)),
    }
    with open(audit_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Audit results saved to {audit_path}")

    return audit_results


if __name__ == "__main__":
    audit_one_split()
