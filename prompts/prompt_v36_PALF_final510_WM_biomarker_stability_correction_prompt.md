# PALF — FINAL PRE-DRAFT BIOMARKER STABILITY CORRECTION
## 510-subject Working-Memory study — reporting-only correction
## NO new model search, NO prediction rerun, NO holdout access

### OBJECTIVE

Perform one small, strictly reporting-level correction to the final 510-subject
Working-Memory evidence package before manuscript drafting.

The purpose is to correct the modality-specific biomarker stability calculations
so that:

1. FC top-10 ROI Jaccard is computed from the FC coefficient map only.
2. SC top-10 ROI Jaccard is computed from the SC coefficient map only.
3. Multimodal top-10 ROI Jaccard remains computed from the combined FC+SC
   importance map.
4. Zero/inactive modality maps are handled explicitly rather than producing
   misleading NaN stability statistics.
5. All corrected biomarker tables/plots/report references are regenerated from
   the already-saved final 510 outer-fold coefficients.

THIS IS NOT A NEW EXPERIMENT.

Do NOT rerun model selection.
Do NOT change any model, hyperparameter, seed, fold, prior, mask, fusion rule,
or prediction result.
Do NOT access the 98-subject holdout.
Do NOT inspect holdout labels/features/metrics.
Do NOT modify any historical Phase 2/3/4/OpenChallenge/scaling outputs.

The final 510 predictive results are already frozen.

---

# 1. SOURCE OF TRUTH

Work from the existing final 510 package:

```text
outputs/iclr/palf_final510_wm/
```

and, if necessary, the corresponding source code and saved coefficient files.

The existing final 510 evidence package contains:
- 510-subject nested-CV results;
- saved outer-fold coefficient maps;
- biomarker stability outputs;
- biomarker faithfulness outputs;
- paper-ready figures/tables.

Use the existing saved outer-fold coefficients.

DO NOT retrain models merely to obtain coefficients.

If a coefficient required for the corrected calculation is genuinely missing,
STOP and report exactly what is missing rather than inventing/reconstructing it
from predictions.

---

# 2. HOLDOUT INTEGRITY — ABSOLUTE REQUIREMENT

The 98-subject holdout remains permanently sealed.

Verify before starting:

```text
data_splits/phase3_holdout_98.txt
```

expected:

```text
n = 98
unique = 98
SHA256 =
89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425
```

The correction must use ONLY:
- final 510 development/nested-CV saved coefficients;
- existing training/outer-fold metadata;
- existing 510 biomarker artifacts.

No holdout FC.
No holdout SC.
No holdout labels.
No holdout predictions.
No holdout-derived rankings.

Create/update:

```text
outputs/iclr/palf_final510_wm/BIOMARKER_CORRECTION_SEAL.json
```

containing:
- holdout hash;
- holdout count;
- statement that no holdout data were accessed;
- timestamp;
- files read;
- files written.

---

# 3. BUG TO CORRECT

Locate the current implementation of the biomarker stability calculation,
including the function currently responsible for the stability maps (expected
to be similar to):

```text
stability_for_maps()
```

The current implementation appears to calculate:

```text
fc_top10_roi_jaccard
sc_top10_roi_jaccard
multimodal_top10_roi_jaccard
```

from the same multimodal ROI-importance array.

That is incorrect for the FC/SC-specific quantities.

The corrected implementation MUST distinguish three separate rankings.

---

# 4. CORRECT ROI IMPORTANCE DEFINITIONS

For every valid outer-fold model coefficient set, first obtain the validated
primal edge coefficient matrices:

```text
C_FC
C_SC
```

with:
- 116 × 116 shape;
- symmetric representation where appropriate;
- zero diagonal;
- corresponding to the final semantic expert contribution;
- no R0 coefficient mixing unless the existing biomarker definition explicitly
  and validly includes it (the current final 510 definition is the semantic
  expert biomarker).

For modality m:

```text
I_i^m = sum_{j != i} |C_m[i,j]|
```

Therefore:

```text
I_FC = incident absolute FC coefficient importance
I_SC = incident absolute SC coefficient importance
```

For multimodal importance:

```text
I_MULTI = I_FC + I_SC
```

Do not normalize these independently unless the existing frozen biomarker
definition already specifies a normalization. Preserve the existing intended
definition.

---

# 5. TOP-10 ROI RANKINGS

For each valid fold:

```text
top10_FC    = top 10 ROIs ranked by I_FC
top10_SC    = top 10 ROIs ranked by I_SC
top10_MULTI = top 10 ROIs ranked by I_MULTI
```

Use deterministic tie handling.

Document the tie rule actually used by the existing implementation. If a tie
rule is not explicit, implement a deterministic rule based on:
1. descending importance;
2. AAL116 canonical ROI index/name as deterministic secondary key.

Do not tune the tie rule.

---

# 6. JACCARD CALCULATION

For two valid fold rankings A and B:

```text
Jaccard(A,B) = |A ∩ B| / |A ∪ B|
```

Compute separately:

```text
FC top10 ROI Jaccard
SC top10 ROI Jaccard
multimodal top10 ROI Jaccard
```

For every pair of valid outer-fold coefficient maps that the existing stability
pipeline compares.

Do not change the existing pairwise-vs-reference aggregation convention.
Preserve the existing convention unless it is demonstrably inconsistent with
the saved artifact structure; if changed, document it.

---

# 7. ZERO / INACTIVE MODALITY HANDLING

Some selected models may have:

```text
v_FC = 1
```

which makes the final SC expert contribution exactly zero.

Similarly, a model could have:

```text
v_FC = 0
```

which makes FC contribution exactly zero.

An inactive modality does NOT have a meaningful biomarker ranking for that fold.

Therefore:

### If FC coefficient map is exactly zero/inactive:

```text
FC stability = invalid for that fold
```

Exclude that fold from FC-specific Spearman/Jaccard aggregation.

### If SC coefficient map is exactly zero/inactive:

```text
SC stability = invalid for that fold
```

Exclude that fold from SC-specific Spearman/Jaccard aggregation.

### Multimodal map

If:

```text
I_MULTI != 0
```

the multimodal ranking remains valid.

If the entire multimodal expert contribution is zero, mark the fold:

```text
ABSTAINED
```

and exclude it from biomarker stability.

Do NOT convert an all-zero map into an arbitrary ranking.

---

# 8. EDGE SPEARMAN

Recompute the existing edge-level stability metrics consistently.

For each valid modality:

```text
Spearman(abs(C_FC_fold_a), abs(C_FC_fold_b))
Spearman(abs(C_SC_fold_a), abs(C_SC_fold_b))
```

Preserve the existing edge-vector definition:
- upper triangle / 6670 edges;
- absolute coefficients;
- same edge ordering.

If a modality is inactive in a fold, exclude that pair from that modality's
Spearman calculation.

Report:

```text
n_valid_pairs
n_excluded_inactive
mean
median
SD
```

where practical.

Do not silently report a mean over an undefined quantity.

---

# 9. TOP-100 / TOP-300 EDGE JACCARD

Recompute the existing edge stability metrics separately for FC and SC using
the correct modality-specific coefficient vectors.

For each valid fold pair:

```text
top100_FC
top300_FC
top100_SC
top300_SC
```

Then aggregate using the existing convention.

If the existing final package already has these values from correct
modality-specific code, verify them and preserve them; otherwise recompute.

Do not alter the definition merely to improve results.

---

# 10. SIGN CONSISTENCY

Recompute/verify sign consistency using the modality-specific coefficient
matrices.

Use the existing intended definition.

If a modality is inactive for a fold, exclude it from modality-specific sign
consistency.

Do not assign artificial signs to zero coefficients.

---

# 11. MODELS / CONTROLS

At minimum recompute corrected stability for:

```text
R-MATCHED
R-CROSS
R-SHUFFLED
R-RANDOM
N-MATCHED
N-CROSS
N-SHUFFLED
N-RANDOM
```

using the saved final 510 outer-fold coefficients.

Do not change the model definitions.

Do not select the best control.

Do not drop an unfavorable control.

Do not change which folds are included except for the explicit inactive-modality
rule above.

---

# 12. PRIMARY CORRECTED OUTPUT

Create:

```text
outputs/iclr/palf_final510_wm/biomarker_stability_corrected.csv
```

Minimum columns:

```text
model
task
n_valid_outer_folds
n_inactive_outer_folds
fc_edge_spearman
sc_edge_spearman
fc_top100_edge_jaccard
sc_top100_edge_jaccard
fc_top300_edge_jaccard
sc_top300_edge_jaccard
fc_top10_roi_jaccard
sc_top10_roi_jaccard
multimodal_top10_roi_jaccard
multimodal_top20_roi_jaccard
fc_sign_consistency
sc_sign_consistency
multimodal_sign_consistency
```

Add mean/median/SD columns where the existing reporting convention supports
them.

Create:

```text
outputs/iclr/palf_final510_wm/BIOMARKER_STABILITY_CORRECTION_REPORT.md
```

Explain:
- what was wrong;
- what was corrected;
- which metrics changed;
- which metrics did not change;
- exact source coefficient files;
- inactive-fold handling;
- confirmation that predictions were not rerun.

---

# 13. REPLACE PAPER-READY BIOMARKER TABLE

Regenerate:

```text
outputs/iclr/palf_final510_wm/paper_ready/table_main_wm_biomarkers.csv
outputs/iclr/palf_final510_wm/paper_ready/table_main_wm_biomarkers.tex
```

using the corrected stability values.

Do not change:
- prediction metrics;
- primary model ranking;
- prior-control definitions;
- faithfulness calculations unless they depend directly on the corrected
  ROI-ranking implementation.

If a paper-ready table contains FC/SC top-10 Jaccard values, they MUST now be
modality-specific.

---

# 14. BIOMARKER PLOTS

Regenerate only the plots that depend on corrected stability metrics.

At minimum:

```text
Fig 5 / biomarker stability
```

must use:
- correct FC stability;
- correct SC stability;
- correct multimodal stability.

If existing figure numbers differ, preserve the repository's current naming
scheme.

Generate both:

```text
PNG
PDF
```

and SVG if the existing paper-ready figure pipeline uses SVG.

Do not manually alter axes or omit controls because their results are
unfavorable.

---

# 15. TOP-10 BIOMARKER LIST

The final top-10 WM biomarker list should continue to use the intended
multimodal ranking:

```text
I_MULTI = I_FC + I_SC
```

unless the existing paper definition explicitly states otherwise.

Do NOT replace the final multimodal biomarker list merely because the corrected
FC/SC-specific rankings change.

Verify that the reported top-10 names trace directly to the saved final
510-fold coefficient-derived biomarker map.

Create/update:

```text
paper_ready/wm_top10_biomarkers.csv
paper_ready/wm_top10_biomarkers.md
```

with:
- ROI;
- atlas index;
- FC importance;
- SC importance;
- multimodal importance;
- rank.

---

# 16. FAITHFULNESS — DO NOT RERUN UNLESS NECESSARY

The existing final 510 faithfulness analysis is conceptually separate from
the FC/SC Jaccard bug.

Do NOT rerun the expensive faithfulness experiment merely because stability
was corrected.

However, inspect whether the reported faithfulness ranking was generated from
the same incorrect FC/SC Jaccard helper.

If faithfulness uses the multimodal ranking:

```text
I_MULTI
```

and that ranking is already correct, leave faithfulness unchanged.

If and ONLY IF the saved faithfulness ranking itself used the incorrect
modality-specific calculation, stop and report the dependency before changing
anything.

Do not automatically rerun.

---

# 17. IMPORTANT: NO PREDICTION CHANGES

The following files/results are frozen and MUST NOT change:

```text
primary_metrics.csv
seed_metrics.csv
fold_metrics.csv
R0 audit
R-MATCHED prediction results
R-CROSS prediction results
R-SHUFFLED prediction results
R-RANDOM prediction results
N-MATCHED prediction results
N-CROSS prediction results
N-SHUFFLED prediction results
N-RANDOM prediction results
```

If any prediction result changes, STOP.

This task is a biomarker-reporting correction only.

---

# 18. NO HOLDOUT ACCESS

Before and after execution, verify:

```text
98-subject holdout access log
```

remains clean.

Search newly created files for holdout IDs.

Required:

```text
no holdout subject IDs
no holdout feature arrays
no holdout labels
no holdout predictions
```

If any appear, STOP and report.

---

# 19. TESTS

Add:

```text
tests/test_biomarker_stability_correction.py
```

Minimum tests:

### T1
FC ROI importance uses FC coefficients only.

### T2
SC ROI importance uses SC coefficients only.

### T3
Multimodal ROI importance equals FC + SC importance.

### T4
FC and SC top-10 rankings can differ.

### T5
Jaccard implementation is correct.

### T6
All-zero modality produces no arbitrary ranking.

### T7
Inactive FC/SC fold is excluded from corresponding modality stability.

### T8
Multimodal stability remains valid when one modality is inactive.

### T9
6670-edge ordering unchanged.

### T10
Coefficient reconstruction remains exact.

### T11
No model fitting is invoked by the correction script.

### T12
No prediction file is modified.

### T13
No holdout access.

### T14
Paper-ready table equals corrected CSV.

### T15
Paper-ready stability figure uses corrected values.

### T16
Top-10 multimodal ROI list traces to saved coefficients.

Run:
- targeted correction tests;
- full repository test suite.

---

# 20. SANITY CHECKS

Print a before/after table for every corrected model:

```text
MODEL
old_fc_top10_jaccard
new_fc_top10_jaccard
old_sc_top10_jaccard
new_sc_top10_jaccard
old_multimodal_top10_jaccard
new_multimodal_top10_jaccard
```

Expected interpretation:

- FC-specific values may change;
- SC-specific values may change;
- multimodal values should remain unchanged if the previous multimodal
  calculation was already based on the correct combined map.

Do NOT force the new values to match any expected direction.

---

# 21. CLAIM AUDIT

Update:

```text
paper_ready/SAFE_CLAIMS.md
paper_ready/UNSAFE_CLAIMS.md
```

ONLY where the corrected biomarker stability values affect claims.

Do not upgrade the biomarker claim tier automatically.

The previous final evidence freeze classified the biomarker evidence as:

```text
BIOMARKER_LEVEL: B0
```

Do not change B0 unless the corrected calculations genuinely satisfy the
pre-existing claim criteria.

In particular, do not convert:
- stability -> faithfulness;
- stability -> causal relevance;
- stability -> validated biomarker;
- candidate biomarker -> clinically validated biomarker.

---

# 22. FINAL EVIDENCE AUDIT

Create:

```text
outputs/iclr/palf_final510_wm/BIOMARKER_CORRECTION_FINAL_AUDIT.md
```

It must state:

1. final cohort remains 510;
2. 98-subject holdout remained sealed;
3. no prediction models were retrained;
4. no hyperparameters changed;
5. no prior changed;
6. no CV split changed;
7. no seed changed;
8. only biomarker stability reporting was corrected;
9. FC/SC modality-specific ROI Jaccard is now correct;
10. multimodal ranking is preserved;
11. all corrected values trace to saved coefficients;
12. tests pass.

---

# 23. RUNTIME

Before starting, print:

```text
BIOMARKER_CORRECTION_ESTIMATED_RUNTIME: <h:mm>
```

Because this is a reporting-only correction, expected runtime should normally be
much shorter than a model-training run.

Track:

```text
Stage A — source/audit
Stage B — corrected coefficient-derived stability
Stage C — tables
Stage D — plots
Stage E — tests/audit
```

At completion print:

```text
BIOMARKER_CORRECTION_ACTUAL_RUNTIME: <h:mm>
```

and create:

```text
outputs/iclr/palf_final510_wm/BIOMARKER_CORRECTION_RUNTIME.md
```

---

# 24. DO NOT DO

Do NOT:

- rerun the 510 predictive experiment;
- rerun model selection;
- rerun hyperparameter search;
- change prior generation;
- change prior controls;
- change R0;
- access the 98 holdout;
- inspect holdout performance;
- alter faithfulness merely to improve its result;
- select favorable folds;
- remove unfavorable controls;
- redefine stability after seeing corrected numbers;
- change claim thresholds;
- change the paper's primary metric;
- manufacture a positive biomarker result.

This is a correctness correction, not an optimization exercise.

---

# 25. FINAL REPORT FORMAT

Print:

## A. Runtime
Estimated and actual.

## B. Integrity
510 cohort + holdout seal.

## C. Bug
Exact implementation location and explanation.

## D. Correction
FC / SC / multimodal importance definitions.

## E. Before vs after
All affected stability values.

## F. Corrected biomarker stability
R-MATCHED / R-CROSS / R-SHUFFLED / R-RANDOM /
N-MATCHED / N-CROSS / N-SHUFFLED / N-RANDOM.

## G. Faithfulness
State explicitly whether it was unchanged.

## H. Top WM biomarkers
Final multimodal top-10.

## I. Claim level
P level unchanged.
Biomarker level recomputed only if justified by the pre-existing criteria.

## J. Tests
Targeted + full suite.

## K. Outputs
List all corrected files.

Finally print exactly:

```text
STATUS: FINAL510_WM_BIOMARKER_STABILITY_CORRECTION_COMPLETE
```

If any holdout access, prediction change, coefficient mismatch, or unexplained
dependency is detected, print:

```text
STATUS: BIOMARKER_CORRECTION_BLOCKED
```

and do not proceed to paper drafting.
