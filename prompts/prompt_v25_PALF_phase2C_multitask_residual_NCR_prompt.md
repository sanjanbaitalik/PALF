# PALF ICLR 2027 — Phase 2C: Prior-Guided Multi-Task Residual NCR Pilot

## 0. Why this is the next experiment

The previous direct-prior formulations have now failed under matched evaluation:

- Full PALF does not beat the corrected same-solver no-prior late-fusion model.
- Differential stacking is worse on both tasks.
- Matched adaptive anisotropy/network regularization is worse on both tasks.
- PALF-MPR multi-penalty feature grouping is strongly worse for Working Memory
  and only marginally positive for Fluid Intelligence.

Do NOT continue tuning those formulations.

The remaining high-upside direction is the one explicitly suggested by the
advisor:

> investigate individual-task prediction against multi-task prediction.

The current codebase already contains:
- `src/metascfc/models/mt_ncr.py`
- `scripts/95_run_multitask_ncr.py`
- leakage-safe residual-branch helpers in
  `src/metascfc/diagnostics/conditional_prior_signal.py`
- `scripts/101_audit_conditional_prior_signal.py`

However, do NOT reuse the old multi-task NCR implementation as final evidence
without correction. It has several methodological limitations:
- direct l2,1 support sharing is too rigid;
- it selects using mean RMSE rather than the final Pearson objective;
- it uses a simple train/validation/test pattern rather than the final
  fully-cross-fitted PALF protocol;
- the existing original-coordinate coefficient conversion in `mt_ncr.py`
  must be audited carefully before any biomarker claim.

Phase 2C introduces a new model that preserves the strongest late-fusion
predictor and asks whether the LLM prior can explain its *remaining errors*
through multi-task residual learning.

Working name:

```text
MT-RNCR
Multi-Task Residual Network-Constrained Ridge
```

The key scientific idea is:

> Use individual-task late fusion as the strong backbone, then use the two
> correlated cognitive targets jointly to learn a shared residual factor and a
> task-contrast residual factor, with LLM-derived shared and contrast priors.

This directly tests:
1. no-prior vs prior,
2. individual-task vs multi-task,
3. whether prior-guided residual information improves final prediction.

Do not edit the manuscript yet.

---

# 1. Fresh development seeds only

Do NOT use any previously inspected seeds:

```text
0–9
101, 202, 303, 404
505, 606, 707, 808
```

Use exactly:

```text
dev_seeds = [909, 1010, 1111, 1212]
```

with five outer folds.

Both cognitive targets must use the same subject partition for the same
development seed/fold.

Total:

```text
4 seeds × 5 folds = 20 paired outer partitions
2 targets -> 40 target evaluations per model family
```

Do not change seeds after seeing results.

---

# 2. Empirical rationale that must be recorded, not tuned

Before model fitting, compute and save:

```text
corr(y_WM, y_FI)
corr(prior_WM, prior_FI)
top10 prior overlap
top20 prior overlap
```

using the full frozen arrays/prior CSVs only as descriptive dataset properties.

Expected approximate values from the current repository:

```text
target Pearson correlation ≈ 0.339
prior Pearson correlation ≈ 0.794
top-10 prior intersection ≈ 8/10
```

Do not use these values for hyperparameter selection.

They motivate a shared/contrast multi-task decomposition:
the targets have moderate shared variance and the two LLM priors have strong
shared semantic structure.

Save to:

```text
DATASET_RELATIONSHIP_AUDIT.json
```

---

# 3. Strong base predictor

Use the current corrected no-prior late-fusion methodology as the backbone for
BOTH tasks.

For each outer split, independently for WM and FI:

```text
generalized no-prior FC
+
SC Ridge
+
fully cross-fitted convex FP+SC fusion
```

Use the current final same-solver fitting protocol and grids.

Call the resulting model:

```text
Base late fusion
```

The Phase-2C method is not allowed to weaken or replace this base predictor.
It learns only a residual correction.

For each outer-training set, generate fully cross-fitted base predictions:

```text
base_oof_WM
base_oof_FI
```

so that every residual target is computed from a prediction that did not train
on that subject.

For the outer test, fit the normal final base model and save:

```text
base_test_WM
base_test_FI
```

---

# 4. Construct leakage-safe residual targets

Within each outer-training set:

```text
r_WM = y_WM - base_oof_WM
r_FI = y_FI - base_oof_FI
```

Do NOT use in-sample base predictions.

For multi-task modeling, standardize the residual targets using statistics
computed only from the relevant residual-model training fold.

Define standardized residuals:

\[
z_{WM} = \frac{r_{WM}-\mu_{WM}}{\sigma_{WM}},
\qquad
z_{FI} = \frac{r_{FI}-\mu_{FI}}{\sigma_{FI}}.
\]

Then form an orthogonal shared/contrast decomposition:

\[
z_{shared}
=
\frac{z_{WM}+z_{FI}}{\sqrt{2}},
\]

\[
z_{contrast}
=
\frac{z_{WM}-z_{FI}}{\sqrt{2}}.
\]

This is the multi-task representation.

After prediction:

\[
\hat z_{WM}
=
\frac{\hat z_{shared}+\hat z_{contrast}}{\sqrt{2}},
\]

\[
\hat z_{FI}
=
\frac{\hat z_{shared}-\hat z_{contrast}}{\sqrt{2}}.
\]

Convert predicted residuals back to each target's original units using only the
training-fold residual mean/std.

---

# 5. Construct frozen LLM multi-task priors

Use the two existing frozen contrastive Qwen priors:

```text
p_WM
p_FI
```

Do not call an LLM.

Do not regenerate a prior.

## 5A. Shared prior

Define:

\[
p_{shared,i}
=
\sqrt{(p_{WM,i}+\epsilon_p)(p_{FI,i}+\epsilon_p)}
\]

with:

```text
epsilon_p = 1e-6
```

Then min-max normalize to `[0,1]`.

This emphasizes ROIs considered relevant by both task priors.

## 5B. Contrast prior

Define:

\[
p_{contrast,i}
=
|p_{WM,i}-p_{FI,i}|.
\]

Min-max normalize to `[0,1]`.

This emphasizes ROIs that differentiate the two semantic task descriptions.

The sign of task contrast comes from the signed target:

```text
z_WM - z_FI
```

not from the prior magnitude.

Save:

```text
priors/shared_prior.csv
priors/contrast_prior.csv
```

including ROI index, ROI label, and normalized score.

---

# 6. Network-Constrained Ridge residual model

Use **FC edges only** for the residual NCR branches.

Rationale:
- the base predictor already includes both FC and SC;
- the LLM semantic prior is an ROI/FC prior;
- the residual branch should test incremental semantic FC information rather
  than re-fit the entire multimodal predictor.

For each residual factor:

\[
\hat\beta
=
\arg\min_{\beta}
\left\{
\|r-X_F\beta\|_2^2
+
c\lambda_1\|\beta\|_2^2
+
c\lambda_2\beta^\top L_p\beta
\right\}.
\]

Use the existing NCR line-graph implementation.

Use:

```text
top_k = 10
laplacian_weighting = binary
laplacian_normalization = sym
couple_modalities = false
```

Use exactly:

```text
lambda1_grid = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0
]

lambda2_grid = [
    0.0,
    0.03,
    0.1,
    0.5,
    1.0,
    2.0,
    5.0
]
```

`lambda2=0` is the no-prior NCR special case.

---

# 7. Four model families to evaluate

The Phase-2C pilot must compare exactly these four.

## Model A — Base late fusion

No residual correction.

```text
prediction = base prediction
```

---

## Model B — Individual-task residual NCR, no prior

For each task independently:

```text
residual target = y_task - base_oof_task
```

Fit FC residual Ridge/NCR with:

```text
lambda2 = 0
```

Tune `lambda1`.

Add the residual prediction to the base using a shrinkage coefficient `eta`.

This tests whether residual learning alone helps.

---

## Model C — Individual-task residual NCR + matched task prior

For each task independently:
- WM residual uses `p_WM`;
- FI residual uses `p_FI`.

Tune:

```text
lambda1
lambda2
eta
```

This tests whether the task-specific prior improves over individual no-prior
residual learning.

---

## Model D — MT-RNCR: multi-task shared/contrast prior residual NCR

Fit:

```text
shared residual NCR using p_shared
contrast residual NCR using p_contrast
```

Reconstruct WM/FI residual predictions from shared/contrast predictions.

Then add the reconstructed residual to the base.

This is the proposed method.

---

# 8. Architecture-matched multi-task no-prior control

For Model D, also compute an exact architecture-matched control:

```text
MT residual no-prior
```

Use the same:
- shared/contrast residual targets;
- FC features;
- fitting scopes;
- eta logic;

but force:

```text
lambda2_shared = 0
lambda2_contrast = 0
```

This control is mandatory.

The crucial prior comparison is:

```text
MT-RNCR
vs
MT residual no-prior
```

not merely MT-RNCR vs Base late fusion.

---

# 9. Residual shrinkage / safety gate

For every residual model use:

\[
\hat y_{final}
=
\hat y_{base}
+
\eta \hat r.
\]

Search:

```text
eta_grid = [0.0, 0.25, 0.50, 0.75, 1.0]
```

`eta=0` exactly recovers the base late-fusion predictor.

Eta must be chosen using only outer-training cross-fitted predictions.

Do not choose eta from the outer-test target.

---

# 10. End-to-end inner selection criterion

Do not select residual models by residual RMSE alone.

The method's final objective is cognitive prediction.

Within the outer-training set, generate cross-fitted corrected predictions for
both tasks.

## Single-task Models B/C

Select the candidate maximizing final task Pearson correlation:

```text
corr(y_task, base_oof_task + eta * residual_oof_task)
```

Tie-break:
1. lower RMSE
2. lower MAE
3. eta closer to 0
4. lambda2 closer to 0
5. simpler candidate

## Multi-task Model D and MT no-prior

Selection must treat the two tasks symmetrically.

For each candidate compute:

```text
r_WM
r_FI
```

on cross-fitted outer-training predictions.

Use the mean Fisher-z score:

\[
score
=
\frac{
\operatorname{atanh}(clip(r_{WM}))
+
\operatorname{atanh}(clip(r_{FI}))
}{2}.
\]

Clip Pearson r to:

```text
[-0.999999, 0.999999]
```

before `atanh`.

Tie-break:
1. mean normalized RMSE across the two tasks
2. mean normalized MAE
3. smaller total eta usage
4. smaller lambda2 values
5. smaller lambda1 values

Do not optimize one task at the expense of the other after seeing outer-test
performance.

---

# 11. Eta for the multi-task model

Allow task-specific residual amplitudes:

```text
eta_WM
eta_FI
```

both chosen from the same predefined grid:

```text
[0.0, 0.25, 0.50, 0.75, 1.0]
```

The shared/contrast NCR hyperparameters are shared across the pair, while the
final residual amplitude may differ by task.

This lets FI fall back toward the strong SC-dominant base predictor if the
residual correction is not useful.

Do not add any other task-specific grid.

---

# 12. Cross-fitting hierarchy

This is critical.

For every outer split:

## Step 1 — base OOF
Create leakage-safe base predictions for all outer-training subjects.

## Step 2 — residual model cross-fitting
Using only the outer-training set:
- split into residual-model inner folds;
- on each residual inner fold:
  - fit FC preprocessing on residual-inner-train only;
  - construct residual standardization on residual-inner-train only;
  - fit candidate NCR;
  - predict residual-inner-validation;
- concatenate residual OOF predictions.

## Step 3 — candidate and eta selection
Use only these residual OOF predictions and base OOF predictions.

## Step 4 — final residual refit
After candidate selection:
- fit residual model on the complete outer-training set;
- use OOF base residual targets, not in-sample base residuals;
- predict the outer-test FC data.

## Step 5 — outer test
Evaluate once:

```text
base_test + eta * residual_test
```

The outer-test labels must never enter Steps 1–4.

---

# 13. Correct coefficient conversion

Audit the old `src/metascfc/models/mt_ncr.py` coefficient conversion.

For standardized feature:

\[
x_z = \frac{x-\mu_x}{\sigma_x}
\]

and standardized target:

\[
y_z = \frac{y-\mu_y}{\sigma_y},
\]

if `beta_z` is the standardized coefficient, the original-unit coefficient is:

\[
\boxed{
\beta_{original}
=
\beta_z
\frac{\sigma_y}{\sigma_x}
}
\]

not:

```text
beta_z * sigma_x / sigma_y
```

Do not silently reuse the old conversion.

Add a synthetic test that verifies primal coefficients reproduce predictions.

Do not overwrite historical outputs generated by the old code.

---

# 14. Coefficients to export

For every MT-RNCR outer fit save:

```text
beta_shared
beta_contrast
```

in standardized residual-target units.

Reconstruct task-specific incremental coefficient maps:

\[
\beta_{WM}^{inc}
=
\frac{\beta_{shared}+\beta_{contrast}}{\sqrt{2}},
\]

\[
\beta_{FI}^{inc}
=
\frac{\beta_{shared}-\beta_{contrast}}{\sqrt{2}}.
\]

Also save original-unit forms with the appropriate residual target scaling.

Required dimensions:

```text
6670 FC edges
```

Save:

```text
coefficients/
  <seed>_<fold>_shared.npz
  <seed>_<fold>_contrast.npz
  <seed>_<fold>_wm_incremental.npz
  <seed>_<fold>_fi_incremental.npz
```

Do NOT make final biomarker claims yet.

These files are for the next phase if prediction succeeds.

---

# 15. Lightweight biomarker-readiness diagnostics

Prediction is the Phase-2C decision criterion, but compute inexpensive
diagnostics so we know whether a biomarker phase is viable.

Across outer fits compute:

## Shared component
- pairwise Spearman correlation of `abs(beta_shared)`
- top-100 edge Jaccard

## Task incremental maps
For WM and FI separately:
- pairwise Spearman of `abs(beta_task_inc)`
- top-100 edge Jaccard
- top-10 ROI Jaccard after edge-to-node saliency aggregation

Do not call these biomarkers yet.

Call them:

```text
coefficient stability diagnostics
```

Do not use them to choose prediction hyperparameters.

---

# 16. Data-derived sanity checks

Save and assert:

```text
n_subjects = 412
n_rois = 116
n_fc_edges = 6670
```

All 412 subjects must have both target labels.

The same subject order must be used for WM and FI.

For every paired outer split:

```text
train_idx_WM == train_idx_FI
test_idx_WM == test_idx_FI
```

---

# 17. Efficient implementation plan

Prefer adding:

```text
src/metascfc/models/multitask_residual_ncr.py
scripts_paper/phase2c_multitask_residual_ncr_pilot.py
configs/iclr/palf_phase2c_mt_residual_ncr.yaml
tests/test_multitask_residual_ncr.py
```

Reuse tested pieces from:

```text
src/metascfc/diagnostics/conditional_prior_signal.py
src/metascfc/models/iclr_backbones/network_constrained_ridge.py
```

Do not copy the old `mt_ncr.py` wholesale.

Cache:
- base OOF predictions;
- base test predictions;
- shared/contrast prior Laplacians;
- Laplacian eigendecompositions;
- completed outer splits.

Make the pilot restartable.

Print ETA after two outer splits.

---

# 18. Output directory

Create only:

```text
outputs/iclr/palf_phase2c_mt_residual_ncr_pilot/
```

with:

```text
COMPLETE
VALIDATION_REPORT.json
RUN_REPORT.md
DATASET_RELATIONSHIP_AUDIT.json

split_metrics.csv
seed_metrics.csv
selection_summary.csv
stability_diagnostics.csv

priors/
coefficients/
plots/
```

Zip to:

```text
outputs/iclr/palf_phase2c_mt_residual_ncr_pilot.zip
```

Do not modify previous result directories.

---

# 19. Required split-level metrics

For every target / seed / fold / model save:

```text
pearson
rmse
mae
```

Models:

```text
base_late_fusion
individual_residual_no_prior
individual_residual_prior
mt_residual_no_prior
mt_rncr
```

For residual models also save:

```text
eta
lambda1
lambda2
```

For MT models save:

```text
lambda1_shared
lambda2_shared
lambda1_contrast
lambda2_contrast
eta_WM
eta_FI
```

---

# 20. Seed-level report

Average the five outer folds within each development seed.

For each target report mean Pearson for:

```text
Base late fusion
Individual residual no-prior
Individual residual + matched prior
MT residual no-prior
MT-RNCR
```

Then report four key paired deltas:

## Prior effect in individual-task learning
```text
Individual prior - Individual no-prior
```

## Multi-task effect without prior
```text
MT no-prior - Individual no-prior
```

## Prior effect in multi-task learning
```text
MT-RNCR - MT no-prior
```

## Final improvement
```text
MT-RNCR - Base late fusion
```

For every delta:
- four seed values
- mean
- median
- positive seeds / 4

No final p-value claim with four development seeds.

---

# 21. Predefined decision rules

## STRONG_GO

Print:

```text
PHASE2C_DECISION: STRONG_GO
```

only if BOTH targets satisfy:

```text
MT-RNCR mean r > Base late fusion mean r
MT-RNCR - Base >= +0.003
positive final-improvement seeds >= 3/4
```

AND BOTH targets satisfy:

```text
MT-RNCR mean r > MT residual no-prior mean r
positive prior-effect seeds >= 3/4
```

This is the desired outcome:
multi-task prior improves both prediction and the architecture-matched no-prior
control.

---

## PROMISING_GO

Print:

```text
PHASE2C_DECISION: PROMISING_GO
```

if:

- MT-RNCR beats Base on both targets with >=3/4 positive seeds;
- at least one target has improvement >= +0.003;
- and the other is positive but smaller;
- and the multi-task prior effect is positive on average for both tasks.

---

## MULTITASK_ONLY_GO

Print:

```text
PHASE2C_DECISION: MULTITASK_ONLY_GO
```

if multi-task residual learning improves prediction but:

```text
MT-RNCR <= MT residual no-prior
```

for one or both tasks.

This means multi-task learning helped but the LLM prior did not.

Do not present it as prior success.

---

## NO_GO

Print:

```text
PHASE2C_DECISION: NO_GO
```

if:
- MT-RNCR fails to improve both targets;
- or the architecture-matched prior effect is nonpositive for either target;
- or the method is unstable across development seeds.

---

# 22. Required figures

Generate:

```text
plots/fig_phase2c_five_model_comparison.pdf/png
plots/fig_phase2c_seed_deltas.pdf/png
plots/fig_phase2c_prior_vs_multitask_effect.pdf/png
plots/fig_phase2c_coefficient_stability.pdf/png
```

Figure 1:
five-model prediction comparison.

Figure 2:
seed-level MT-RNCR minus Base deltas.

Figure 3:
separate bars for:
- individual prior effect
- multi-task effect
- multi-task prior effect

Figure 4:
coefficient stability diagnostics.

---

# 23. Tests

Add at least:

1. shared residual transform is invertible.
2. shared/contrast reconstruction exactly recovers synthetic WM/FI residuals.
3. shared prior is finite and normalized.
4. contrast prior is finite and normalized.
5. outer WM/FI split indices are identical.
6. base OOF prediction for a subject is generated without training on that
   subject.
7. residual targets use OOF base predictions only.
8. lambda2=0 removes NCR prior penalty.
9. MT no-prior uses identical architecture to MT-RNCR except Laplacian penalty.
10. eta=0 exactly recovers base prediction.
11. outer-test target is never passed to residual selection.
12. original-coordinate coefficient conversion uses
    `beta_std * y_scale / x_scale`.
13. saved primal coefficients reproduce direct residual predictions.
14. coefficient vector length is 6670.
15. no previous development/final seeds are used.
16. model selection uses Fisher-z mean for the paired MT objective.
17. stability diagnostics are not used in prediction selection.
18. historical result directories are untouched.

Run targeted tests and full suite.

---

# 24. Final OpenCode report

Print exactly:

## A. Code audit
- old `mt_ncr.py` issues found
- whether coefficient scaling issue was confirmed
- new files added
- git HEAD

## B. Dataset relationship
- target Pearson correlation
- LLM prior Pearson correlation
- prior top-10 and top-20 overlap

## C. Development setup
- seeds
- folds
- subjects
- runtime
- validation gates

## D. Prediction results
For each target:

```text
Base late fusion
Individual residual no-prior
Individual residual prior
MT residual no-prior
MT-RNCR
```

Pearson, RMSE, MAE.

## E. Decomposition of gains
For each target print:

```text
Individual prior effect
MT no-prior - individual no-prior
MT prior effect
Final MT-RNCR - Base effect
```

with 4 seed deltas and positive seed counts.

## F. Selected residual hyperparameters
- eta frequencies by task/model
- lambda1/lambda2 frequencies
- fraction eta=0
- fraction lambda2=0

## G. Coefficient readiness
- number of coefficient files
- reconstruction max error
- shared coefficient stability
- WM incremental stability
- FI incremental stability

## H. Tests
- targeted
- full suite
- pre-existing failures

## I. Outputs
- directory
- ZIP
- CSVs
- priors
- coefficients
- figures

Then exactly one:

```text
PHASE2C_DECISION: STRONG_GO
```

```text
PHASE2C_DECISION: PROMISING_GO
```

```text
PHASE2C_DECISION: MULTITASK_ONLY_GO
```

or

```text
PHASE2C_DECISION: NO_GO
```

Finally:

```text
STATUS: PHASE2C_MT_RESIDUAL_NCR_PILOT_COMPLETE
```

If any validity gate fails:

```text
STATUS: PHASE2C_NOT_VALID
```

and do not interpret performance.

---

# 25. Stop after the pilot

Do not run final seeds and do not rewrite the paper after Phase 2C.

If Phase 2C is a STRONG/PROMISING GO, the next phase will:

1. freeze the method,
2. run a fresh 10-seed final evaluation,
3. run matched/cross-task/shuffled/random controls,
4. perform coefficient stability,
5. perform held-out top-vs-random/bottom perturbation,
6. evaluate shared and task-specific biomarkers,
7. compare single-task against multi-task prediction.

That will be the final attempt to satisfy the advisor's requested story without
using outer-test results to manufacture a win.
