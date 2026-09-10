
================================================================================
Phase 2D: Prior-Selected Subspace NCR Expert Fusion Pilot — Final Report
================================================================================

## A. Baseline correctness
  git HEAD: 2fa288c2b6954c53bec8650dc4d85ab800f7bcd1
  WM fused r: 0.257854 (expected ~0.2635) -> PASS
  FI fused r: 0.365037 (expected ~0.3709) -> PASS
  Overall: PASS

## B. Development setup
  Seeds: [1313, 1414, 1515, 1616]
  Outer folds: 5
  Fusion folds: 3
  Inner folds: 3
  Subjects: 412
  Prior files: working_memory_contrastive_qwen3, fluid_intelligence_contrastive_qwen3
  Runtime: 3043.1s

## C. Expert masks
  WM:
    roi_incident: 11/20
    direct_topk: 9/20
    Mean mask size: 362
    Mean Laplacian ratio: 0.170
  FI:
    roi_incident: 13/20
    direct_topk: 7/20
    Mean mask size: 148
    Mean Laplacian ratio: 0.105

## D. Prediction results
  WM:
    Strong no-prior baseline: r=0.2795 rmse=111.7670 mae=111.1787
    Matched prior Ridge expert fusion: r=0.2751
    Matched prior Ridge expert fusion: r=0.2749 rmse=85.4959 mae=84.5527
    Matched PS-NCR-EF: r=0.2751
    Matched PS-NCR-EF: r=0.2749 rmse=85.4959 mae=84.5527
    Cross-task PS-NCR-EF: r=0.2673 rmse=83.9181 mae=82.9123
    Shuffled PS-NCR-EF: r=0.2786 rmse=83.6334 mae=82.7571
    Random PS-NCR-EF: r=0.2807 rmse=82.6221 mae=81.7510
  FI:
    Strong no-prior baseline: r=0.3562 rmse=17.4855 mae=16.8697
    Matched prior Ridge expert fusion: r=0.3534
    Matched prior Ridge expert fusion: r=0.3536 rmse=13.8933 mae=13.1268
    Matched PS-NCR-EF: r=0.3534
    Matched PS-NCR-EF: r=0.3536 rmse=13.8933 mae=13.1268
    Cross-task PS-NCR-EF: r=0.3562 rmse=15.0910 mae=14.3741
    Shuffled PS-NCR-EF: r=0.3603 rmse=14.0359 mae=13.2490
    Random PS-NCR-EF: r=0.3535 rmse=14.6586 mae=13.9117

## E. Matched prediction gain
  WM:
    seed 1313: delta=-0.0268
    seed 1414: delta=+0.0047
    seed 1515: delta=+0.0062
    seed 1616: delta=-0.0026
    Mean: -0.0046, Median: +0.0010, Positive: 2/4
  FI:
    seed 1313: delta=+0.0069
    seed 1414: delta=+0.0077
    seed 1515: delta=-0.0035
    seed 1616: delta=-0.0215
    Mean: -0.0026, Median: +0.0017, Positive: 2/4

## F. NCR contribution
  WM:
    PS-NCR-EF minus Ridge delta: mean=+nan
    Positive seeds: 0/4
    Laplacian ratio frequencies: {}
  FI:
    PS-NCR-EF minus Ridge delta: mean=+nan
    Positive seeds: 0/4
    Laplacian ratio frequencies: {}

## G. Prior specificity
  WM:
    matched - cross_task: +0.0076
    matched - shuffled: -0.0037
    matched - random: -0.0058
  FI:
    matched - cross_task: -0.0026
    matched - shuffled: -0.0067
    matched - random: +0.0001

## H. Expert mechanism
  WM:
    Expert FC weight: 0.7475
    Expert SC weight: 0.2525
    Final alpha: 0.2400
    Mean pred corr: 0.6209
    Mean error corr: 0.9034
  FI:
    Expert FC weight: 0.3225
    Expert SC weight: 0.6775
    Final alpha: 0.2250
    Mean pred corr: 0.6186
    Mean error corr: 0.9251

## I. Coefficient validation
  Files exported: 160
  Full edge dimensions: (6670,)
    seed_1414_fold_3_fluid_intelligence_shuffled.npz: beta_fc shape=(6670,), selected_edges=565
    seed_1414_fold_4_fluid_intelligence_shuffled.npz: beta_fc shape=(6670,), selected_edges=1620
    seed_1616_fold_0_working_memory_matched.npz: beta_fc shape=(6670,), selected_edges=1105
    seed_1515_fold_1_fluid_intelligence_cross_task.npz: beta_fc shape=(6670,), selected_edges=1620
    seed_1515_fold_0_working_memory_cross_task.npz: beta_fc shape=(6670,), selected_edges=300
  Max primal reconstruction error: see split_metrics.csv

## J. Tests
  (Run separately: pytest tests/test_prior_subspace_expert_fusion.py)

## K. Outputs
  Directory: /home/genaicoe/Documents/Sanjan/iclr/metaSFC_extends/outputs/iclr/palf_phase2d_ps_ncr_expert_fusion
  ZIP: /home/genaicoe/Documents/Sanjan/iclr/metaSFC_extends/outputs/iclr/palf_phase2d_ps_ncr_expert_fusion/phase2d_all_outputs.zip
  CSVs: split_metrics.csv, seed_metrics.csv, expert_selection.csv, prior_control_summary.csv
  Plots: /home/genaicoe/Documents/Sanjan/iclr/metaSFC_extends/outputs/iclr/palf_phase2d_ps_ncr_expert_fusion/plots
  Coefficients: /home/genaicoe/Documents/Sanjan/iclr/metaSFC_extends/outputs/iclr/palf_phase2d_ps_ncr_expert_fusion/coefficients

================================================================================
PHASE2D_DECISION: NO_GO

STATUS: PHASE2D_PS_NCR_EXPERT_FUSION_COMPLETE