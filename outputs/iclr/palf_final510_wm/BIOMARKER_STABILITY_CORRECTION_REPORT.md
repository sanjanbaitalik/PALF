# Biomarker stability correction report (reporting-only)

## What was wrong

`stability_for_maps()` in `scripts_paper/final510_wm_pilot.py` computed `fc_top10_roi_jaccard` and `sc_top10_roi_jaccard` from index 2 of the map tuple, which is the *multimodal* ROI-importance array. FC and SC top-10 ROI Jaccard were therefore identical to the multimodal value by construction (verified in biomarker_stability.csv). In addition, folds with v_FC = 1 (inactive SC) produced all-zero SC vectors, and the old code propagated NaN Spearman values instead of excluding them.

## What was corrected

- FC metrics now use `I_FC[i] = sum_j |C_FC[i,j]|` only.
- SC metrics now use `I_SC[i] = sum_j |C_SC[i,j]|` only.
- Multimodal metrics use `I_MULTI = I_FC + I_SC` (unchanged definition).
- Inactive modality folds are excluded per modality; all-zero maps never produce rankings. ABSTAINED folds (expert alpha = 0) are excluded from all stability metrics.
- Tie rule (documented): descending importance, then ascending AAL116 canonical ROI index (`np.lexsort`). The old `argsort(...)[-k:]` had an implicit tie behaviour favouring larger indices.
- Added `multimodal_sign_consistency` on `C_MULTI = C_FC + C_SC` (previously not reported) and per-metric mean/median/SD, n_pairs, and n_excluded_inactive.

## Which metrics changed

| model | old FC t10 J | new FC t10 J | old SC t10 J | new SC t10 J | old MULTI t10 J | new MULTI t10 J |
|---|---|---|---|---|---|---|
| R-MATCHED | 0.4796 | 0.6294 | 0.4796 | 0.4098 | 0.4796 | 0.4796 |
| R-CROSS | 0.5138 | 0.6081 | 0.5138 | 0.4552 | 0.5138 | 0.5138 |
| R-SHUFFLED | 0.5492 | 0.6417 | 0.5492 | 0.5556 | 0.5492 | 0.5492 |
| R-RANDOM | 0.5018 | 0.4778 | 0.5018 | 0.4054 | 0.5018 | 0.5018 |
| N-MATCHED | 0.3875 | 0.5062 | 0.3875 | 0.3683 | 0.3875 | 0.3875 |
| N-CROSS | 0.4419 | 0.4766 | 0.4419 | 0.3746 | 0.4419 | 0.4419 |
| N-SHUFFLED | 0.5492 | 0.5846 | 0.5492 | 0.5195 | 0.5492 | 0.5492 |
| N-RANDOM | 0.4392 | 0.4296 | 0.4392 | 0.3661 | 0.4392 | 0.4392 |

FC/SC top-10 ROI Jaccard change (previously equal to multimodal), and SC edge Spearman is now a finite number over active folds instead of NaN. Edge Jaccard, sign consistency, and multimodal metrics keep their definitions; multimodal values are unchanged in expectation.

## Which metrics did not change

- All prediction metrics (primary_metrics.csv, seed_metrics.csv, fold_metrics.csv) and all model/hyperparameter selections.
- Faithfulness (biomarker_faithfulness.csv): it uses the combined `I_MULTI` ranking, which was already computed correctly.

## Source coefficient files

`outputs/iclr/palf_final510_wm/_state/folds/fold_seed*_f*.pkl` (25 saved outer-fold checkpoints). No model was retrained; coefficients were read via `coefficient_maps` from the frozen checkpoints.

## Inactive-fold handling

FC inactive: expert abstained (alpha = 0). SC inactive: alpha = 0 or v_FC = 1. Excluded per modality from edge Spearman, edge Jaccard, top-10 ROI Jaccard, and sign consistency. Multimodal stability remains valid whenever the combined map is non-zero.

## Confirmation

Predictions were not rerun; no model selection, hyperparameter search, prior change, seed change, or fold change was performed. The 98-subject holdout was not accessed.
