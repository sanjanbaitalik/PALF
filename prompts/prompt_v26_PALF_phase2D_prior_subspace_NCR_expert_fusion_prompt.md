# PALF ICLR 2027 — Phase 2D: Prior-Selected Subspace NCR Expert Fusion Pilot

## Executive decision

Do NOT continue the Phase-2C residual-NCR implementation and do NOT use its
reported FAIL as final evidence against residual/multi-task ideas.

A code audit found that the Phase-2C runner did not faithfully implement the
intended protocol. Important issues include:

1. `scripts_paper/phase2c_multitask_residual_ncr_pilot.py` fits StandardScaler
   on the whole outer-training set before inner-CV model selection.
2. It builds residual targets from in-sample full-training base predictions
   instead of leakage-safe base OOF predictions.
3. It averages WM/FI selected FC/SC Ridge alphas before fitting the base model.
4. `select_eta()` expects base predictions, but the runner passes standardized
   residual targets as the `base_oof` argument.
5. Individual prior and individual no-prior eta selection use the same OOF
   residual prediction array.
6. The implemented lambda2 grid is `[0, 0.1, 1, 10]`, not the requested grid.
7. The MT prior branch hard-codes lambda2=1 rather than selecting it.
8. The coefficient export uses a hard-coded NCR setting and therefore does not
   necessarily correspond to the selected predictive model.

Therefore Phase 2D must use the validated fully-cross-fitted same-solver PALF
evaluation utilities as the scientific backbone and must NOT reuse the Phase-2C
runner.

The new hypothesis is based on the strongest remaining empirical signal:

> The LLM prior appears to identify informative subnetworks, but global
> prior-weighted penalties distort an already strong full-connectome predictor.
> Use the prior to construct a separate low-dimensional expert instead, and
> fuse that expert with the untouched strong baseline.

Working name:

```text
PS-NCR-EF
Prior-Selected Network-Constrained Expert Fusion
```

This combines the safest part of Prior-Selected Subspace Stacking with NCR as
the proposed prior-aware expert.

---

# 1. Scientific requirements

The proposed model must satisfy all of the following.

1. Preserve the current strong same-solver no-prior late-fusion predictor
   exactly as a nested special case.
2. The prior must enter through feature/subnetwork selection and, optionally,
   NCR inside the independent expert — not by modifying the baseline.
3. FC and SC prior experts must be tested because Fluid Intelligence is
   SC-dominant in the current final fusion.
4. NCR must be directly ablated against restricted Ridge on the exact same
   prior-selected subspace.
5. All model/hyperparameter/fusion selection must be outer-test clean.
6. Save true primal expert coefficients for later biomarker validation.
7. Do not use Phase-1 differential stacking or the Phase-2C residual model.

---

# 2. Correct baseline

Use the CURRENT corrected same-solver no-prior pipeline from:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

and the same scientific semantics as:

```text
outputs/iclr/palf_same_solver_fusion_corrected_v2/
```

The baseline is:

```text
generalized same-solver no-prior FC
+
SC Ridge
+
fully cross-fitted convex FP+SC fusion
```

Do not replace it with ordinary FC Ridge.

Do not average hyperparameters across tasks.

Do not weaken its grid.

## Frozen correctness audit

Before any Phase-2D development run, add an audit mode that evaluates the
existing frozen seeds 0–9 using the current final code path and checks that the
aggregated corrected baseline is approximately:

```text
Working Memory fused r = 0.263515
Fluid Intelligence fused r = 0.370917
```

This audit is correctness-only.

Do NOT use seeds 0–9 for Phase-2D development or model choice.

If the audit does not reproduce the current baseline to the expected numerical
tolerance, stop with:

```text
STATUS: PHASE2D_BASELINE_AUDIT_FAILED
```

---

# 3. Fresh development seeds

Do not reuse any previous development seeds:

```text
0–9
101, 202, 303, 404
505, 606, 707, 808
909, 1010, 1111, 1212
```

Use exactly:

```text
DEV_SEEDS = [1313, 1414, 1515, 1616]
```

with:

```text
5 outer folds
3 fusion OOF folds
3 inner parameter-selection folds
3 final reselection folds
```

Use the exact same outer train/test indices for baseline and every prior/control
expert for a target.

Do not change this seed set after looking at results.

---

# 4. Frozen priors

Use only the existing frozen Qwen priors:

```text
outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv
outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv
```

Do not call an LLM.

Do not regenerate priors.

For edge `(i,j)` define:

\[
q_{ij} = p_i p_j.
\]

Use the AAL116 upper-triangle ordering already used by the repository.

---

# 5. Prior-selected edge masks

Test exactly two mask families.

## 5A. Direct top-K edge mask

Rank all 6670 undirected edges by:

```text
q_ij = p_i * p_j
```

and retain the top:

```text
K_EDGE_GRID = [100, 300, 600, 1200]
```

Call:

```text
direct_topk
```

---

## 5B. Top-M ROI incident-edge mask

Take the top-M ROIs by `p_i`.

Retain every FC/SC edge incident to at least one selected ROI.

Use:

```text
M_ROI_GRID = [5, 10, 15]
```

Expected edge counts:

```text
M=5  -> 565
M=10 -> 1105
M=15 -> 1590
```

Call:

```text
roi_incident
```

Assert the counts under the repository's undirected AAL116 edge convention.

Do not add further mask types after seeing outer-test results.

---

# 6. Independent prior experts

For each target and selected mask `E`, fit two independent experts:

```text
FC expert: X_F[:, E]
SC expert: X_S[:, E]
```

The experts predict the ORIGINAL cognitive target, not residuals.

This is crucial.

The strong baseline remains completely untouched.

---

# 7. Restricted-Ridge expert ablation

For a selected expert feature matrix `X_E`, fit:

\[
\hat\beta
=
\arg\min_\beta
\|y-X_E\beta\|_2^2
+
\lambda_R\|\beta\|_2^2.
\]

Search:

```text
RIDGE_EXPERT_GRID = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0
]
```

Call this:

```text
Prior-selected Ridge expert
```

This is the primary ablation that isolates whether hard prior selection alone
is useful.

---

# 8. Prior-selected NCR expert — proposed expert

On the SAME selected edge subset, build the induced line-graph Laplacian from
the current target prior.

Reuse the repository's tested line-graph construction semantics from:

```text
src/metascfc/models/iclr_backbones/network_constrained_ridge.py
```

but restrict the penalty to the selected expert edge coordinates.

Fit:

\[
\hat\beta
=
\arg\min_\beta
\left[
\|y-X_E\beta\|_2^2
+
\lambda_R\|\beta\|_2^2
+
\lambda_L\beta^\top L_E\beta
\right].
\]

Parameterize:

```text
lambda_L = ratio * lambda_R
```

with:

```text
LAPLACIAN_RATIO_GRID = [0.0, 0.1, 0.3, 1.0]
```

`ratio=0` must reproduce the restricted-Ridge expert exactly.

Call:

```text
Prior-selected NCR expert
```

Hard gates:

- identical selected feature set between Ridge and NCR expert;
- identical preprocessing;
- identical target;
- same selection folds;
- ratio=0 equivalence tested numerically.

---

# 9. Expert parameter selection

Within every training scope, candidate selection must be based only on its
inner-validation subjects.

For each modality separately, select:

```text
mask_family
mask_size
lambda_R
laplacian_ratio
```

using:

1. highest pooled Pearson r;
2. RMSE tie-break;
3. MAE tie-break;
4. lower Laplacian ratio;
5. smaller mask;
6. lambda closest to 1;
7. deterministic final tie-break.

Do not use outer-test prediction to choose:
- direct_topk vs roi_incident;
- FC vs SC expert;
- Ridge vs NCR;
- mask size;
- penalties.

---

# 10. Fully cross-fitted expert OOF predictions

Use the existing PALF cross-fitting design.

For each outer-training set:

1. create the same 3 fusion OOF folds used by the final PALF protocol;
2. for each held-out fusion fold:
   - perform expert hyperparameter selection using only its analysis subset;
   - fit the selected expert on that analysis subset;
   - predict the held-out fusion fold;
3. concatenate the expert OOF predictions.

Generate:

```text
fc_expert_oof
sc_expert_oof
```

for Ridge and NCR expert variants.

All feature standardization is fit inside the corresponding analysis fold.

Never fit StandardScaler on the entire outer-training set before expert inner
CV.

---

# 11. Hierarchical convex fusion

Do NOT use unconstrained multi-column stacking.

Phase 1 showed that a more flexible meta-stack can be anti-predictive.

Instead use only the repository's already-tested 2-branch convex fusion logic.

## Stage A — fuse the two prior experts

Using outer-training OOF predictions, choose:

\[
\hat y_{expert}
=
v\,\hat y_{FCexpert}
+
(1-v)\,\hat y_{SCexpert}
\]

with:

```text
v in [0, 0.05, ..., 1]
```

using the same criterion as `search_fusion_weights`:

1. Pearson
2. RMSE
3. MAE

Save:

```text
expert_w_FC
expert_w_SC
```

## Stage B — fuse strong baseline with prior expert

Let:

```text
base_oof = corrected no-prior late-fusion OOF prediction
expert_oof = Stage-A expert OOF prediction
```

Select:

\[
\hat y_{final}
=
(1-\alpha)\hat y_{base}
+
\alpha \hat y_{expert}
\]

with:

```text
alpha in [0, 0.05, ..., 1]
```

again using the same 2-branch convex search.

`alpha=0` exactly reproduces the strong no-prior baseline.

Call the final proposed model:

```text
PS-NCR-EF
```

Do not add a separate residual stage.

Do not add cross-task prediction columns in Phase 2D.

---

# 12. Conservative expert-use diagnostic

Because the prior expert is an optional add-on, report:

```text
fraction alpha = 0
fraction 0 < alpha <= 0.25
fraction alpha > 0.25
mean alpha
median alpha
```

by target.

Also report:

```text
corr(base_oof_error, expert_oof_error)
corr(base_oof, expert_oof)
```

The method has a plausible ensemble mechanism only if the expert is:
- individually predictive;
- not perfectly redundant with the base model.

Do not use these outer-test diagnostics to redefine the method.

---

# 13. Prior controls

Run the exact same PS-NCR-EF architecture under four frozen prior identities:

```text
matched
cross_task
shuffled
random
```

Requirements:

- same outer splits;
- same search grids;
- same fusion procedure;
- same number of selected edges;
- control prior only changes edge ranking / induced line graph.

Use one fixed shuffled and one fixed random prior per task during the pilot.

Do not select the best random prior.

Primary prediction comparison:

```text
matched PS-NCR-EF
vs
strong no-prior baseline
```

Prior-specificity comparisons:

```text
matched
vs
cross_task
vs
shuffled
vs
random
```

---

# 14. Optional data-driven screening control

Do NOT run this until the matched pilot is complete.

If the matched model passes at least `PROMISING_GO`, add one capacity-matched
supervised screening control:

For each expert training scope, rank edges by:

```text
abs(corr(X_edge, y))
```

computed only on the relevant inner-training data.

Use the same selected edge cardinality and same Ridge/NCR/fusion machinery.

Call:

```text
train-only marginal-screening expert
```

This is a strong capacity control, not the primary no-prior baseline.

It must not influence matched-prior model selection.

---

# 15. Multi-task analysis

Do not force multi-task coefficients in the first Phase-2D pilot.

The advisor's individual-vs-multi-task question will be tested only after the
individual-task PS-NCR-EF model is shown to be useful.

If Phase 2D passes, Phase 2D.1 will add:

```text
shared-prior expert
task-specific-prior expert
```

as a pre-registered ablation using the same low-dimensional expert mechanism.

Do not spend the current pilot budget on the already-failed shared/contrast
residual architecture.

---

# 16. Fresh final refit

After expert OOF parameter selection:

1. reselect the expert configuration on the complete outer-training set using
   only its 3-fold CV;
2. fit the selected FC expert on all outer-training subjects;
3. fit the selected SC expert on all outer-training subjects;
4. predict the outer-test set once;
5. use previously selected `expert_w_FC`, `expert_w_SC`, and `alpha`;
6. evaluate the untouched outer test.

No test-driven refit.

---

# 17. True primal coefficient export

For every final expert fit save:

```text
selected_edge_indices
beta_expert_standardized
beta_expert_original_units
scaler_mean
scaler_scale
y_mean
y_scale
```

Embed the expert coefficients back into a full 6670-edge vector with zeros
outside the selected subspace.

Validate:

```text
direct model prediction
==
prediction reconstructed from saved primal coefficients + intercept
```

to:

```text
max_abs_error <= 1e-8
```

for both FC and SC experts.

For the final PS-NCR-EF linear map, also export:

```text
beta_FC_expert_weighted = alpha * expert_w_FC * beta_FC_expert
beta_SC_expert_weighted = alpha * expert_w_SC * beta_SC_expert
```

Do not claim full-model biomarkers yet unless the current same-solver baseline
primal coefficients are independently validated.

The expert map itself is a valid prior-specific biomarker object.

---

# 18. Prediction outputs

Create:

```text
outputs/iclr/palf_phase2d_ps_ncr_expert_fusion/
```

with:

```text
COMPLETE
VALIDATION_REPORT.json
RUN_REPORT.md
BASELINE_AUDIT.json

split_metrics.csv
seed_metrics.csv
expert_selection.csv
prior_control_summary.csv

coefficients/
predictions/
plots/
```

Zip:

```text
outputs/iclr/palf_phase2d_ps_ncr_expert_fusion.zip
```

Do not overwrite old outputs.

---

# 19. Required split-level rows

For each:

```text
task / seed / outer_fold / prior_type / expert_type
```

save:

```text
baseline_pearson
baseline_rmse
baseline_mae

fc_expert_pearson
sc_expert_pearson
expert_fused_pearson

final_pearson
final_rmse
final_mae

delta_final_vs_baseline

selected_mask_family_fc
selected_mask_size_fc
selected_lambda_fc_expert
selected_laplacian_ratio_fc

selected_mask_family_sc
selected_mask_size_sc
selected_lambda_sc_expert
selected_laplacian_ratio_sc

expert_w_FC
expert_w_SC
alpha_base_vs_expert

base_expert_prediction_corr
base_expert_error_corr
```

---

# 20. Seed-level summary

Average five outer folds within each development seed.

For each target report:

```text
Strong baseline
Matched prior Ridge expert fusion
Matched PS-NCR-EF
Cross-task PS-NCR-EF
Shuffled PS-NCR-EF
Random PS-NCR-EF
```

For matched PS-NCR-EF vs baseline report:

- four seed deltas;
- mean delta;
- median delta;
- positive seeds / 4.

For NCR vs restricted-Ridge expert report:

- mean delta;
- positive seeds / 4.

For matched vs each prior control report the same descriptive pilot deltas.

No final significance claim from four development seeds.

---

# 21. Predefined pilot decision

## STRONG_GO

Return:

```text
PHASE2D_DECISION: STRONG_GO
```

if BOTH targets satisfy:

```text
matched PS-NCR-EF mean final delta >= +0.003
positive seeds >= 3/4
```

AND at least one target satisfies:

```text
mean delta >= +0.005
```

AND:

```text
matched mean final r > shuffled mean final r
matched mean final r > random mean final r
```

for both targets.

---

## PROMISING_GO

Return:

```text
PHASE2D_DECISION: PROMISING_GO
```

if:

Working Memory:

```text
mean delta >= +0.004
positive seeds >= 3/4
```

and Fluid Intelligence:

```text
mean delta >= 0
```

with matched prior at least numerically better than shuffled/random overall.

OR vice versa.

---

## RIDGE_EXPERT_GO

Return:

```text
PHASE2D_DECISION: RIDGE_EXPERT_GO
```

if prior-selected restricted Ridge expert fusion improves prediction but NCR
does not.

This would mean hard prior subspace selection works, but the NCR penalty is not
the source of gain.

Do not claim NCR superiority under this outcome.

---

## NO_GO

Return:

```text
PHASE2D_DECISION: NO_GO
```

if:

- matched final prediction is not positive relative to the strong baseline on
  either target;
- or matched is not better than shuffled/random;
- or expert selection is unstable and outer performance is systematically
  negative.

---

# 22. Plots

Generate:

```text
plots/fig_phase2d_model_comparison.pdf/png
plots/fig_phase2d_seed_deltas.pdf/png
plots/fig_phase2d_expert_weights.pdf/png
plots/fig_phase2d_prior_controls.pdf/png
plots/fig_phase2d_mask_selection.pdf/png
```

Figure 1:
baseline vs Ridge expert fusion vs PS-NCR-EF.

Figure 2:
seed-level matched PS-NCR-EF minus baseline.

Figure 3:
expert FC/SC weights and final alpha.

Figure 4:
matched/cross-task/shuffled/random final prediction.

Figure 5:
selected mask family/size and Laplacian ratio.

---

# 23. Tests

Add:

```text
tests/test_prior_subspace_expert_fusion.py
```

Minimum tests:

1. direct top-K mask has exact requested cardinality.
2. ROI-incident mask has expected cardinality.
3. mask is deterministic for a frozen prior.
4. shuffled/random controls preserve cardinality.
5. Ridge expert StandardScaler is fit only on analysis/train indices.
6. NCR ratio=0 reproduces restricted Ridge expert.
7. induced subspace Laplacian is PSD within tolerance.
8. FC and SC expert feature indices are identical for the same prior/mask.
9. OOF expert prediction for a subject is generated without fitting on that
   subject.
10. baseline branch is current generalized same-solver no-prior, not ordinary
    FC Ridge.
11. `alpha=0` reproduces baseline predictions exactly.
12. `expert_w_FC + expert_w_SC = 1`.
13. final model uses only OOF-selected fusion weights.
14. outer-test target never enters mask/penalty/fusion selection.
15. primal coefficient reconstruction reproduces predictions.
16. coefficient vector length is 6670.
17. fresh development seeds are used.
18. previous output directories remain unchanged.
19. checkpoint restart does not duplicate split rows.
20. frozen correctness audit reproduces current baseline before pilot launch.

Run targeted tests and then full suite.

Do not hide pre-existing unrelated failures.

---

# 24. Implementation reuse map

Reuse, rather than rewrite:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

for:
- outer/fusion/inner split logic;
- final same-solver baseline;
- prediction metrics;
- 2-branch convex fusion semantics.

Reuse:

```text
src/metascfc/models/iclr_backbones/network_constrained_ridge.py
```

for NCR mathematical machinery / prior edge graph semantics.

Reuse useful mask/feature-selection ideas from:

```text
src/metascfc/diagnostics/conditional_prior_signal.py
```

but do NOT reuse its older residual-evaluation harness as the final protocol.

Add preferably:

```text
src/metascfc/experiments/prior_subspace_expert_fusion.py
scripts_paper/phase2d_prior_subspace_ncr_expert_pilot.py
tests/test_prior_subspace_expert_fusion.py
```

Do not build on:

```text
scripts_paper/phase2c_multitask_residual_ncr_pilot.py
```

for scientific evaluation.

---

# 25. Final OpenCode report

Print exactly:

## A. Baseline correctness
- git HEAD
- frozen baseline audit WM
- frozen baseline audit FI
- PASS/FAIL

## B. Development setup
- seeds
- folds
- subjects
- prior files
- runtime

## C. Expert masks
For each target:
- mask-family frequencies
- selected sizes
- exact edge counts

## D. Prediction results
For each target:

```text
Strong no-prior baseline
Matched prior Ridge expert fusion
Matched PS-NCR-EF
Cross-task PS-NCR-EF
Shuffled PS-NCR-EF
Random PS-NCR-EF
```

Report Pearson, RMSE, MAE.

## E. Matched prediction gain
For each target:
- four seed deltas PS-NCR-EF minus baseline
- mean
- median
- positive seeds / 4

## F. NCR contribution
For each target:
- PS-NCR-EF minus Ridge-expert-fusion delta
- positive seeds / 4
- selected Laplacian ratio frequencies

## G. Prior specificity
For each target:
- matched minus cross-task
- matched minus shuffled
- matched minus random

## H. Expert mechanism
For each target:
- expert FC weight
- expert SC weight
- final alpha
- fraction alpha=0
- base/expert prediction correlation
- base/expert error correlation

## I. Coefficient validation
- files exported
- full edge dimensions
- maximum primal reconstruction error

## J. Tests
- targeted
- full suite
- pre-existing failures

## K. Outputs
- directory
- ZIP
- CSVs
- plots
- coefficient path

Then print exactly one:

```text
PHASE2D_DECISION: STRONG_GO
```

```text
PHASE2D_DECISION: PROMISING_GO
```

```text
PHASE2D_DECISION: RIDGE_EXPERT_GO
```

or

```text
PHASE2D_DECISION: NO_GO
```

according to the predefined rule.

Finally:

```text
STATUS: PHASE2D_PS_NCR_EXPERT_FUSION_COMPLETE
```

If any validation gate fails:

```text
STATUS: PHASE2D_NOT_VALID
```

and do not interpret prediction results.

---

# 26. Stop after this pilot

Do not run final 10-seed inference or rewrite the paper after Phase 2D.

If Phase 2D is STRONG/PROMISING GO:

1. freeze the method;
2. add individual-vs-shared/specific multi-task expert ablation;
3. run a fresh 10-seed final evaluation;
4. perform matched/shuffled/random/cross-task inference;
5. run biomarker stability;
6. run held-out top-vs-random/bottom perturbation;
7. update the paper.

If Phase 2D is RIDGE_EXPERT_GO, do not force NCR into the headline without an
NCR-specific gain.

If Phase 2D is NO_GO, stop introducing additional prediction architectures on
the same cohort.
