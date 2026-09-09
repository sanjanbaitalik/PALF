# PALF ICLR 2027 — Phase 2B: Joint Multi-Penalty Semantic-Group Ridge Pilot

## Executive decision from Phase 2A.1

Phase 2A.1 is a genuine **NO_GO** for the current adaptive anisotropy / line-graph
formulation.

The matched audit shows:

### Working Memory
```text
Matched robust no-prior FP r      ≈ 0.31510
Adaptive FP r                     ≈ 0.29917
FP delta                          ≈ -0.01593

Matched robust no-prior fused r   ≈ 0.30541
Adaptive fused r                  ≈ 0.29282
Fused delta                       ≈ -0.01258

Positive adaptive-vs-baseline seed deltas:
FP     0/4
Fused  0/4
```

### Fluid Intelligence
```text
Matched robust no-prior FP r      ≈ 0.23690
Adaptive FP r                     ≈ 0.23237
FP delta                          ≈ -0.00453

Matched robust no-prior fused r   ≈ 0.37613
Adaptive fused r                  ≈ 0.37382
Fused delta                       ≈ -0.00231

Positive adaptive-vs-baseline seed deltas:
FP     1/4
Fused  1/4
```

The semantic-shadow analysis is also negative.

Approximate mean seed-level semantic effects:

```text
Working Memory:
semantic FP delta                 ≈ -0.00204
semantic fixed-fusion delta       ≈ -0.00169

Fluid Intelligence:
semantic FP delta                 ≈ -0.00253
semantic fixed-fusion delta       ≈ -0.00161
```

Therefore:

> Do not spend additional compute tuning the current `D_rho + tau_L L_p`
> formulation.

Phase 2B tests a different, simpler hypothesis:

> The semantic prior may be useful as a **coarse group-level differential
> regularizer inside one joint FC+SC regression**, even though the previous
> continuous anisotropy and line-graph smoothness are too brittle.

This is the last major prediction-method pivot before deciding whether to keep
the current honest paper or stop the ICLR submission.

Do not edit the manuscript yet.

---

# 1. New model: PALF-MPR

Working name:

```text
PALF-MPR
Prior-Aware Multi-Penalty Regression
```

The model jointly fits:

1. generic FC edges,
2. semantic-priority FC edges,
3. SC edges,

with separate regularization strengths.

The semantic prior is used only to define a **disjoint FC feature group**.
There is no line-graph term and no continuous anisotropic `D(q)` in Phase 2B.

This is intentional: the previous fine-grained semantic geometry failed under
the matched pilot.

---

# 2. Define the semantic FC group

Use the frozen target-specific semantic prior already in the repository.

For each target:

1. sort the 116 AAL ROIs by frozen `prior_score`;
2. take exactly the top:

```text
K = 10
```

ROIs;

3. define semantic FC edge set `A` as every undirected FC edge incident to at
least one top-K ROI;

4. define generic FC edge set `G` as all remaining FC edges.

The two sets must be disjoint and exhaustive:

```text
A ∩ G = empty
A ∪ G = all 6670 FC edges
```

Do not tune `K` in Phase 2B.

Do not change the prior.

Do not use target labels to define `A`.

Save the edge indices and ROI names used for each target.

Expected active FC edge count for 10 ROIs in a 116-node undirected graph is:

```text
10 * (116 - 10) + C(10, 2) = 1105
```

Assert:

```text
len(A) == 1105
len(G) == 5565
```

unless the repository's edge convention proves otherwise.

---

# 3. Joint multi-penalty objective

Let:

```text
X_G = standardized generic FC edges
X_A = standardized semantic-priority FC edges
X_S = standardized SC edges
```

All standardization must be fit within the relevant training fold.

Solve:

\[
\hat\beta
=
\arg\min_{\beta_G,\beta_A,\beta_S}
\left\{
\|y -
X_G\beta_G -
X_A\beta_A -
X_S\beta_S\|_2^2
+
\lambda
\left(
\|\beta_G\|_2^2
+
r_A\|\beta_A\|_2^2
+
r_S\|\beta_S\|_2^2
\right)
\right\}.
\]

Where:

- `lambda` = overall regularization scale;
- `r_A` = semantic-FC penalty ratio;
- `r_S` = SC penalty ratio.

Interpretation:

```text
r_A = 1
```

means semantic and generic FC edges receive the same penalty.

```text
r_A < 1
```

means prior-priority FC edges are shrunk less.

Therefore the exact no-prior comparison is contained in the same solver.

Do not allow `r_A > 1` in this pilot. The semantic hypothesis being tested is:

> task-priority FC edges may deserve less shrinkage than the rest of FC.

---

# 4. Baseline and proposed model

Run exactly two joint regression models per outer split.

## Model A — Joint no-prior multi-modal Ridge

Force:

```text
r_A = 1.0
```

Tune:

```text
lambda
r_S
```

using nested CV.

Call it:

```text
Joint no-prior Ridge
```

This is the architecture-matched baseline.

---

## Model B — PALF-MPR

Tune:

```text
r_A
lambda
r_S
```

using the same nested-CV machinery.

Call it:

```text
PALF-MPR
```

The candidate family contains Model A because:

```text
r_A = 1
```

is part of the proposed search grid.

---

# 5. Hyperparameter grids

Use exactly:

```text
lambda_grid = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]

r_A_grid = [
    0.10,
    0.25,
    0.50,
    1.00
]

r_S_grid = [
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
    4.00,
    10.00
]
```

Do not modify these after observing results.

Model A searches:

```text
5 × 7 = 35 candidates
```

Model B searches:

```text
5 × 4 × 7 = 140 candidates
```

---

# 6. Fresh development seeds

Do NOT reuse:

```text
0–9
101
202
303
404
```

Those seeds have already influenced model-development decisions.

Use exactly:

```text
dev_seeds = [505, 606, 707, 808]
```

with five outer folds.

Total:

```text
2 targets × 4 seeds × 5 folds = 40 outer splits
```

Baseline and PALF-MPR must use identical train/test indices split-by-split.

Do not change these seeds after seeing results.

---

# 7. Inner validation

Because the new solver works in subject space and should be computationally
cheap, use:

```text
5-fold inner CV
```

inside every outer-training set.

Use deterministic:

```python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=20000 + 100 * seed + outer_fold
)
```

Feature means and standard deviations must be fit separately within each inner
training fold.

Target centering must also be fit using inner-training data only.

---

# 8. Selection criterion

Use the same deterministic criterion for baseline and PALF-MPR:

1. maximize pooled out-of-inner-fold Pearson correlation;
2. if tied within `1e-12`, minimize pooled RMSE;
3. if still tied, minimize pooled MAE.

For exact ties:

### Baseline
prefer:
```text
r_S closest to 1
then lambda closest to 1
then lower lambda
```

### PALF-MPR
prefer:
```text
r_A closest to 1
then r_S closest to 1
then lambda closest to 1
then lower lambda
```

This tie-break deliberately prefers the no-prior model when performance is
numerically indistinguishable.

Do NOT use the one-SE rule in Phase 2B.

The one-SE rule was already evaluated in the previous adaptive formulation.
This new pilot tests whether the simpler group-level parameterization is stable
under ordinary nested selection.

---

# 9. Efficient subject-space solver

Do not build or invert a dense `13340 × 13340` matrix.

For a given training fold, precompute:

```text
K_G = X_G X_G^T
K_A = X_A X_A^T
K_S = X_S X_S^T
```

For any `(r_A, r_S)` define:

\[
K_0 =
K_G +
\frac{1}{r_A}K_A +
\frac{1}{r_S}K_S.
\]

The fitted training predictions correspond to kernel Ridge with:

\[
K_0(K_0+\lambda I)^{-1}y.
\]

For validation/test samples use cross-kernels constructed with the same
training-fitted standardization.

For efficiency:

- precompute group Gram matrices once per inner fold;
- for each `(r_A,r_S)`, eigendecompose `K_0` once if useful;
- evaluate the five `lambda` values using the same decomposition;
- cache completed outer splits;
- make the run restartable.

Print runtime after the first two outer splits.

---

# 10. Intercept and standardization

For every inner/final training fit:

1. standardize each original feature using training-only mean/std;
2. replace zero std with 1;
3. center `y` by the training-only target mean;
4. solve the penalized model;
5. add the training target mean back to predictions.

No global scaling.

The three feature groups may be concatenated conceptually, but the solver must
preserve their separate penalties.

---

# 11. Recover and save true primal coefficients

This is mandatory because the next phase will be biomarker discovery if the
prediction pilot succeeds.

For each final outer fit recover:

```text
beta_G
beta_A
beta_S
```

in the standardized-feature coordinate system.

The dual-to-primal relation must be implemented exactly from the multi-penalty
objective.

Reconstruct a full 6670-dimensional FC vector:

```text
beta_FC_full
```

where generic and semantic-group coefficients are placed back at their original
edge indices.

Also save:

```text
beta_semantic_group
```

as the full 6670-vector with zeros outside group A.

Hard validation:

- prediction reconstructed from saved primal coefficients must match the
  subject-space prediction to tolerance `<= 1e-8`;
- coefficient arrays must have correct dimensions;
- all values finite.

Save coefficients as compressed NPZ per outer split.

Do NOT compute biomarker claims yet.

---

# 12. Also compute current late-fusion reference on the new dev seeds

We need to know whether the joint solver itself is competitive with the current
paper architecture.

For the **same new seeds 505/606/707/808**, compute one additional descriptive
reference using the current corrected same-solver late-fusion methodology:

```text
Generalized no-prior FC
+
SC Ridge
+
fully cross-fitted convex FP+SC fusion
```

Use the current final methodology and its current grids.

Call it:

```text
Current late-fusion reference
```

This reference is descriptive during development.

Do not use its outer-test performance to tune PALF-MPR.

The primary Phase-2B comparison remains:

```text
PALF-MPR vs Joint no-prior Ridge
```

---

# 13. Prediction metrics

For every split save:

```text
task
seed
outer_fold
model

pearson
rmse
mae

selected_lambda
selected_r_A
selected_r_S

n_semantic_edges
n_generic_edges
```

For the current late-fusion reference also save its:

```text
fused_pearson
fused_rmse
fused_mae
```

---

# 14. Seed-level summary

Average five outer folds within each seed.

For each target report:

### Current late-fusion reference
- mean r
- RMSE
- MAE

### Joint no-prior Ridge
- mean r
- RMSE
- MAE

### PALF-MPR
- mean r
- RMSE
- MAE

### Paired PALF-MPR minus joint baseline
- mean delta r
- median delta r
- positive seeds / 4

No final significance claim from four development seeds.

---

# 15. Semantic-use diagnostics

For PALF-MPR report:

```text
r_A distribution
fraction r_A = 1
fraction r_A < 1
```

by target and overall.

Also report:

```text
mean PALF-MPR - baseline delta
on r_A < 1 splits

mean PALF-MPR - baseline delta
on r_A = 1 splits
```

Important identity gate:

If PALF-MPR selects exactly the same:

```text
r_A = 1
lambda
r_S
```

as the joint baseline, predictions must be identical.

---

# 16. Predefined decision rule

## STRONG_GO

Return:

```text
PHASE2B_DECISION: STRONG_GO
```

if BOTH targets satisfy:

```text
PALF-MPR mean r > Joint no-prior mean r
PALF-MPR positive in >= 3/4 seeds
```

AND at least one target has:

```text
mean delta r >= +0.005
```

AND semantic shrinkage is actually used:

```text
r_A < 1 on >= 25% of outer splits overall
```

AND PALF-MPR is at least numerically competitive with the current late-fusion
reference on both targets:

```text
PALF-MPR mean r >= late-fusion mean r - 0.002
```

---

## WM_GO

Return:

```text
PHASE2B_DECISION: WM_GO
```

if Working Memory satisfies:

```text
mean delta r >= +0.005
positive seeds >= 3/4
```

and Fluid Intelligence is approximately neutral:

```text
PALF-MPR delta r >= -0.002
```

and semantic shrinkage is genuinely selected.

---

## ARCHITECTURE_ONLY_GO

Return:

```text
PHASE2B_DECISION: ARCHITECTURE_ONLY_GO
```

if:

- joint no-prior Ridge clearly improves over the current late-fusion reference,
- but PALF-MPR does not improve over joint no-prior Ridge.

This would mean joint feature-level multi-penalty integration helps prediction,
but the semantic prior does not.

Do not call this a prior success.

---

## NO_GO

Return:

```text
PHASE2B_DECISION: NO_GO
```

if:

- PALF-MPR is worse than the joint baseline on both targets;
- or semantic `r_A<1` settings systematically hurt;
- or the joint architecture itself is materially worse than current
  late-fusion prediction.

---

# 17. Required plots

Create:

```text
plots/fig_phase2b_seed_deltas.pdf/png
plots/fig_phase2b_three_model_comparison.pdf/png
plots/fig_phase2b_rA_selection.pdf/png
plots/fig_phase2b_rmse_mae.pdf/png
```

### Figure 1
Seed-level PALF-MPR minus joint-baseline Pearson delta.

### Figure 2
Three-model mean Pearson comparison:
- current late fusion
- joint no-prior Ridge
- PALF-MPR

### Figure 3
Selected `r_A` frequencies.

### Figure 4
RMSE and MAE comparison.

---

# 18. Output directory

Create only:

```text
outputs/iclr/palf_phase2b_joint_group_ridge_pilot/
```

with:

```text
README.md
RUN_REPORT.md
VALIDATION_REPORT.json
COMPLETE

split_metrics.csv
seed_metrics.csv
selection_summary.csv

coefficients/
  <task>_seed<seed>_fold<fold>_baseline.npz
  <task>_seed<seed>_fold<fold>_palf_mpr.npz

plots/
```

Zip to:

```text
outputs/iclr/palf_phase2b_joint_group_ridge_pilot.zip
```

Do not overwrite any prior output.

---

# 19. Tests

Add:

```text
tests/test_palf_multi_penalty_ridge.py
```

Minimum tests:

1. semantic/generic edge groups are disjoint.
2. union contains exactly 6670 FC edges.
3. semantic edge count is 1105 for top-10 ROI definition.
4. `r_A=1` gives uniform FC penalty across all FC edges.
5. baseline and PALF-MPR share the exact solver.
6. same `(lambda,r_A=1,r_S)` produces identical predictions.
7. train-only standardization is enforced.
8. target centering is train-only.
9. subject-space prediction equals explicit small synthetic primal solution.
10. recovered primal coefficients reproduce predictions.
11. coefficient vector dimensions are correct.
12. no final/test seeds 0–9 or previous dev seeds 101/202/303/404 are used.
13. baseline and PALF-MPR share outer splits.
14. current late-fusion reference is not used in PALF-MPR selection.
15. no Phase-1 stacking helper is used.
16. no old adaptive `D_rho/tau_L` helper is used in the new model.

Run targeted tests and the full suite.

---

# 20. Do NOT do yet

Do not:

- edit the manuscript;
- run final 10-seed inference;
- tune top-K;
- tune the semantic prior;
- add line-graph penalties;
- add continuous anisotropy;
- add stacking;
- add nonlinear neural models;
- run biomarker stability/faithfulness;
- choose new grids after seeing the new outer-test results.

The next phase depends entirely on this pilot.

---

# 21. Final OpenCode report

Print:

## A. Environment
- git HEAD
- new scripts/modules
- development seeds
- runtime
- test results

## B. Edge grouping
For each target:
- top-10 ROI names
- semantic edge count
- generic edge count
- disjoint/exhaustive assertions

## C. Three-model results
For each target:

```text
Current late-fusion reference r
Joint no-prior Ridge r
PALF-MPR r

PALF-MPR - joint baseline delta
PALF-MPR - late-fusion delta

RMSE
MAE
```

## D. Seed-level consistency
For each target:
- four PALF-MPR vs joint-baseline deltas
- positive seeds / 4
- mean delta
- median delta

## E. Semantic usage
For each target:
- r_A counts
- fraction r_A < 1
- mean delta on semantic-active splits
- mean delta on r_A=1 splits

## F. Coefficient validation
- number of coefficient files
- primal prediction reconstruction max error
- finite-value status

## G. Tests
- targeted
- full suite
- pre-existing failures

## H. Outputs
- output directory
- ZIP
- CSVs
- plots
- coefficient directory

Then print exactly one:

```text
PHASE2B_DECISION: STRONG_GO
```

```text
PHASE2B_DECISION: WM_GO
```

```text
PHASE2B_DECISION: ARCHITECTURE_ONLY_GO
```

or

```text
PHASE2B_DECISION: NO_GO
```

according to the predefined rules.

Finally:

```text
STATUS: PHASE2B_JOINT_GROUP_RIDGE_PILOT_COMPLETE
```

If any validity gate fails:

```text
STATUS: PHASE2B_NOT_VALID
```

and do not interpret performance.

---

# 22. Scientific interpretation boundary

The semantic group is defined from the frozen prior, so coefficient overlap
with that group is not an independent biomarker validation.

If Phase 2B succeeds, the next phase will evaluate biomarker quality using:

- cross-seed coefficient stability;
- top-k ROI stability;
- held-out top-vs-random/bottom perturbation;
- cross-task specificity;
- matched vs shuffled/random prior controls.

Do not claim biomarker superiority from Phase 2B alone.
