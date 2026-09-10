
================================================================================
Phase 2D-FIX: Prior-Selected Subspace NCR Expert Fusion Pilot — Final Report
================================================================================

## A. Baseline correctness
  git HEAD: ce362ab7b865ee9185694a733204a86650444d9f
  Code path: R0 from palf_crossfit_ablation.py (F1 FIX)
  WM fused r: 0.263515 (expected ~0.263515, tol=5e-4) -> PASS
  FI fused r: 0.370917 (expected ~0.370917, tol=5e-4) -> PASS
  WM fused rmse: 11.2929 (expected ~11.2929, tol=0.05) -> PASS
  FI fused rmse: 4.5667 (expected ~4.5667, tol=0.05) -> PASS
  Overall: PASS

## B. Development setup
  Seeds (FIX): [1717, 1818, 1919, 2020]
  Audit seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Outer folds: 5
  Fusion folds: 3
  Inner folds: 3
  Subjects: 412
  Output dir: /home/genaicoe/Documents/Sanjan/iclr/metaSFC_extends/outputs/iclr/palf_phase2d_fix_ps_ncr_expert_fusion
  Runtime: 1443.2s

## C. Expert masks (independent FC/SC — F4 FIX)
  WM:
    FC mask families:
      roi_incident: 16/20
      direct_topk: 4/20
    FC mean mask size: 220
    SC mask families:
      roi_incident: 18/20
      direct_topk: 2/20
    SC mean mask size: 36
  FI:
    FC mask families:
      roi_incident: 17/20
      direct_topk: 3/20
    FC mean mask size: 50
    SC mask families:
      roi_incident: 14/20
      direct_topk: 6/20
    SC mean mask size: 224

## D. Prediction results
  WM:
    Strong no-prior baseline (R0): r=0.2414 rmse=11.3432 mae=9.2430
    Matched prior Ridge expert fusion (Ridge): r=0.2488
    Matched prior Ridge expert fusion (NCR): r=0.2556 rmse=11.2178 mae=9.1068
    Matched PS-NCR-EF (Ridge): r=0.2488
    Matched PS-NCR-EF (NCR): r=0.2556 rmse=11.2178 mae=9.1068
    Cross-task PS-NCR-EF: r=0.2428 rmse=11.3251 mae=9.2085
    Shuffled PS-NCR-EF: r=0.2554 rmse=11.2631 mae=9.1715
    Random PS-NCR-EF: r=0.2336 rmse=11.3332 mae=9.2498
  FI:
    Strong no-prior baseline (R0): r=0.3555 rmse=4.6026 mae=3.8172
    Matched prior Ridge expert fusion (Ridge): r=0.3517
    Matched prior Ridge expert fusion (NCR): r=0.3524 rmse=4.6000 mae=3.8179
    Matched PS-NCR-EF (Ridge): r=0.3517
    Matched PS-NCR-EF (NCR): r=0.3524 rmse=4.6000 mae=3.8179
    Cross-task PS-NCR-EF: r=0.3481 rmse=4.6135 mae=3.8329
    Shuffled PS-NCR-EF: r=0.3567 rmse=4.5913 mae=3.8156
    Random PS-NCR-EF: r=0.3665 rmse=4.5806 mae=3.8043

## E. Matched prediction gain
  WM:
    seed 1717: delta=+0.0080
    seed 1818: delta=+0.0251
    seed 1919: delta=+0.0238
    seed 2020: delta=-0.0001
    Mean: +0.0142, Median: +0.0159, Positive: 3/4
  FI:
    seed 1717: delta=-0.0076
    seed 1818: delta=+0.0003
    seed 1919: delta=-0.0058
    seed 2020: delta=+0.0008
    Mean: -0.0031, Median: -0.0027, Positive: 2/4

## F. NCR contribution
  WM:
    PS-NCR-EF minus Ridge delta: mean=+0.0068
    Positive seeds: 10/20
    Laplacian ratio frequencies: {np.float64(0.3): 3, np.float64(1.0): 12, np.float64(0.0): 4, np.float64(0.1): 1}
  FI:
    PS-NCR-EF minus Ridge delta: mean=+0.0007
    Positive seeds: 8/20
    Laplacian ratio frequencies: {np.float64(0.1): 6, np.float64(0.0): 11, np.float64(0.3): 1, np.float64(1.0): 2}

## G. Prior specificity
  WM:
    matched - cross_task: +0.0128
    matched - shuffled: +0.0002
    matched - random: +0.0220
  FI:
    matched - cross_task: +0.0043
    matched - shuffled: -0.0043
    matched - random: -0.0141

## H. Expert mechanism
  WM:
    Expert FC weight: 0.7850
    Expert SC weight: 0.2150
    Final alpha: 0.3675
  FI:
    Expert FC weight: 0.3350
    Expert SC weight: 0.6650
    Final alpha: 0.0975

## I. Reconstruction validation (F5 FIX)
  Coefficient files exported: 160
  Max expert reconstruction error (NCR): 4.63e-12
  Max final reconstruction error (NCR): 1.15e-12
  Max expert reconstruction error (Ridge): 7.11e-14
  Max final reconstruction error (Ridge): 4.26e-14
  Reconstruction validation: PASS

## J. Forensic report (F1-F6)
  F1: FIXED — R0 from palf_crossfit_ablation.py via generate_crossfit_oof + reselect_and_fit_final
  F2: FIXED — R0 handles target centering/scaling correctly; no custom _fit_and_predict_ridge_on_subset with fit_intercept=False
  F3: FIXED — N/A
  F4: FIXED — generate_expert_crossfit_oof_fixed with _select_best_mask_for_modality per modality
  F5: FIXED — validate_expert_reconstruction and validate_final_reconstruction from corrected module
  F6: FIXED — {"pearson_tolerance": "5e-4", "rmse_tolerance": "0.05", "wm_expected_pearson": 0.263515, "fi_expected_pearson": 0.370917, "wm_expected_rmse": 11.2929, "fi_expected_rmse": 4.5667}

## K. Tests
  (Run separately: pytest tests/test_prior_subspace_expert_fusion_fix.py)

## L. Outputs
  Directory: /home/genaicoe/Documents/Sanjan/iclr/metaSFC_extends/outputs/iclr/palf_phase2d_fix_ps_ncr_expert_fusion
  CSVs: split_metrics.csv, seed_metrics.csv, prior_control_summary.csv
  JSONs: BASELINE_AUDIT.json, FORENSIC_REPORT.json, diagnostics.json
  Plots: /home/genaicoe/Documents/Sanjan/iclr/metaSFC_extends/outputs/iclr/palf_phase2d_fix_ps_ncr_expert_fusion/plots
  Coefficients: /home/genaicoe/Documents/Sanjan/iclr/metaSFC_extends/outputs/iclr/palf_phase2d_fix_ps_ncr_expert_fusion/coefficients

================================================================================
PHASE2D_FIX_DECISION: NO_GO

STATUS: PHASE2D_FIX_PS_NCR_EXPERT_FUSION_COMPLETE