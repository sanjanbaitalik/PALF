# Phase 4 — FINAL CONFIRMATION REPORT

Target: Working Memory (ListSort_Unadj). Frozen method: PS-NCR-EF.
Independent holdout: 98 subjects (one-time evaluation).

## A. Runtime
- Historical planning estimate: 2-4 h (upper 4-6 h)
- Measured ETA: ~1.02 h (see RUNTIME_ESTIMATE.md)
- Actual: Stage A 349 s; Stage B/C 880 s; Stage D ~0 s; Stages E/F/G 10 s; total ~21 min

## B. Pre-open integrity
- dev=412, holdout=98, intersection=0, SHA verified
- FI holdout labels never loaded; holdout never accessed before unlock
- R0 audit PASS: WM r=0.2635147736, RMSE=11.2929210027
- Phase-2D-FIX audit PASS (valid module; ratio=0==Ridge err=0; recon 2.8e-14)

## C. Frozen 412 model
- matched NCR: FC cfg=['roi_incident', 15, 0.1, 1.0], SC cfg=['roi_incident', 5, 0.1, 1.0]
- v_fc=0.65, alpha=0.25
- pooled dev OOF: expert r=0.2214, final r=0.2823 (R0 pooled 0.2742)
- freeze SHA256=2026-09-13 05:40:41

## D. Frozen WM biomarker top-10 (development only)
```
 rank  roi_index_1based           roi_name  importance
    1                 7      Frontal_Mid_L    0.288025
    2                 8      Frontal_Mid_R    0.279384
    3                 4      Frontal_Sup_R    0.274871
    4                 3      Frontal_Sup_L    0.264199
    5                60     Parietal_Sup_R    0.255176
    6                65          Angular_L    0.237403
    7                64    SupraMarginal_R    0.205708
    8                81     Temporal_Sup_L    0.198299
    9                12 Frontal_Inf_Oper_R    0.195525
   10                61     Parietal_Inf_L    0.171279
```
ranking SHA256=990ac6a2ee1752b5f3662d7718a52e3e3fe5b36bd8078a11be49fed800fd7412

## E. Holdout prediction
```
        model  pearson      rmse      mae
           R0 0.196358 11.578882 9.626418
matched_ridge 0.189768 11.527454 9.529871
  matched_ncr 0.178386 11.610512 9.625823
    cross_ncr 0.197194 11.461659 9.451031
 shuffled_ncr 0.203284 11.534141 9.628174
   random_ncr 0.196501 11.585027 9.655216
```

## F. Primary prediction inference
- observed delta_r = -0.0180
- bootstrap 95% CI = [-0.0442, +0.0086]
- one-sided lower bound = -0.0442
- fraction <= 0 = 0.8624
- Williams/Steiger t=-1.154, p=0.252
- delta_RMSE=+0.0296 CI [-0.1053, 0.1699]
- delta_MAE=-0.0024 CI [-0.1305, 0.1241]
PREDICTION_CONFIRMATION: FAIL

## G. Biomarker faithfulness
- unmasked RMSE=11.6105
- top10 delta_RMSE=-0.3798; top5=-0.0436
- bottom5=-0.1494; bottom10=-0.0855
- random10 mean=+0.0191 p95=+0.8599; empirical p=0.8681
- random5 mean=+0.0116; empirical p=0.5105
- bootstrap sensitivity mean=-0.3945 CI [-0.5986, -0.1992]
BIOMARKER_CONFIRMATION: FAIL

## H. Ranking controls (same matched predictor)
      ranking  top10_delta_rmse
      matched         -0.379819
   cross_task          0.275162
     shuffled          0.646737
       random          0.731866
matched_ridge          0.226082

## I. Development stability context (DEVELOPMENT ONLY)
prior_type  n_valid_fits  fc_abs_edge_spearman  sc_abs_edge_spearman  fc_top10_roi_jaccard  sc_top10_roi_jaccard
   matched            18                 0.753                 0.790                 0.590                 0.528
cross_task            19                 0.733                 0.626                 0.582                 0.498
  shuffled            17                 0.184                 0.172                 0.187                 0.173
    random            15                 0.166                 0.137                 0.156                 0.146

## J. Decisions
PREDICTION_CONFIRMATION: FAIL
BIOMARKER_CONFIRMATION: FAIL
PHASE4_DECISION: WM_CONFIRMATION_FAILED

## Technical-failure disclosure
All fixes restore the predeclared frozen computation. No model, mask, hyperparameter, fusion weight, ranking, endpoint, threshold, randomization seed, or statistical test was changed. The buggy first pass is archived and disclosed. The corrected evaluation is reported exactly as obtained, whatever the outcome.

Events: V1_validator_shape_bug, V2_final_map_formula_bug, V3_missing_iu_variable

The first pass (buggy map formula) gave delta_r=-0.0138; the corrected frozen formula gave delta_r=-0.0180. Both are negative; the conclusion is unchanged. Buggy outputs are archived in _state/.

STATUS: PHASE4_WM_CONFIRMATION_COMPLETE
