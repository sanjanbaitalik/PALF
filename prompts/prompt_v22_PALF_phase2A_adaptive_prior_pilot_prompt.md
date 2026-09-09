# PALF ICLR 2027 — Phase 2A: Adaptive Prior-Trust Regularization Pilot

## Objective

Phase 1 differential stacking failed:

- Working Memory differential stack: mean Δr ≈ -0.0090 vs corrected same-solver baseline.
- Fluid Intelligence differential stack: mean Δr ≈ -0.0053, Holm-adjusted p ≈ 0.0195 in the negative direction.
- Meta-CV improvement was anti-correlated with outer-test improvement for both targets.
- The nonnegative differential-Ridge candidate was never selected.

Therefore **do not continue stacking**.

The next hypothesis is that semantic information is useful, but the current
binary/full-strength regularizer is too rigid. Move adaptation *inside the FC
solver*, where it can continuously interpolate between:

- no semantic prior,
- anisotropic semantic shrinkage,
- semantic network regularization,
- and their combination.

This is a **development pilot**, not the final ICLR evaluation.

Do not edit the paper yet.
Do not run biomarker discovery yet.
Do not use the already-known outer seeds 0–9 for method development.

---

# 1. Scientific model: Adaptive PALF

The current generalized FC objective is

\[
\hat\beta
=
\arg\min_{\beta}
\left\{
\|\widetilde y-\widetilde X_F\beta\|_2^2
+
c\lambda_F \beta^\top D\beta
+
c\lambda_L \beta^\top L_p\beta
\right\}.
\]

Replace the binary prior mechanisms with two directly tunable strengths.

## 1A. Adaptive anisotropic trust

Let the existing matched-prior diagonal be

\[
D_p = D(q;\gamma), \qquad \gamma=0.5.
\]

Define

\[
D_{\rho}
=
(1-\rho)I+\rho D_p
\]

with

```text
rho_aniso_grid = [0.0, 0.25, 0.50, 0.75, 1.0]
```

Interpretation:

- `rho_aniso = 0` → isotropic same-solver Ridge geometry.
- `rho_aniso = 1` → current full anisotropic PALF geometry.
- intermediate values softly trust the semantic prior.

Because both `I` and `D_p` have average diagonal scale 1, the interpolation
retains the intended regularization scale.

---

## 1B. Direct network-strength parameter

Do not multiply another trust coefficient by the old `lambda_L`; that creates
an unnecessary identifiability/confounding problem.

Instead define a direct effective semantic network strength

\[
\tau_L \ge 0
\]

and use

\[
c\tau_L \beta^\top L_p\beta.
\]

Search

```text
tau_net_grid = [
    0.0,
    0.003,
    0.01,
    0.03,
    0.10,
    0.50,
    1.0,
    2.0,
    5.0
]
```

The adaptive objective is therefore

\[
\boxed{
\hat\beta_{\mathrm{A-PALF}}
=
\arg\min_{\beta}
\left\{
\|\widetilde y-\widetilde X_F\beta\|_2^2
+
c\lambda_F\beta^\top D_{\rho}\beta
+
c\tau_L\beta^\top L_p\beta
\right\}
}
\]

with fixed

```text
c = 13340
gamma = 0.5
epsilon = 1e-3
product edge lifting
top_k_prior_rois = 10
```

This candidate family contains all important current models as exact or near-exact
special cases:

```text
rho=0, tau=0      -> same-solver no prior
rho=1, tau=0      -> anisotropy only
rho=0, tau>0      -> network-only family
rho=1, tau>0      -> Full-PALF family
0<rho<1           -> new soft semantic anisotropy
```

That is the key methodology change.

---

# 2. FC and SC hyperparameter grids

Use the existing full generalized-FC grid:

```text
lambda_F_grid = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]
```

Do not narrow this grid based on old test results.

For SC Ridge, expand the upper end because the frozen final run frequently
selected the old maximum:

```text
alpha_SC_grid = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    300.0,
    1000.0,
    3000.0
]
```

The SC expansion must be applied identically to the development baseline and
Adaptive PALF.

Do not claim an Adaptive-PALF gain merely because its SC branch has a better
grid than the baseline.

---

# 3. Development seeds only

This phase must not reuse the already-inspected primary seeds 0–9 for tuning the
new method.

Use exactly these development seeds:

```text
dev_seeds = [101, 202, 303, 404]
```

Use five outer folds per seed.

Use the same outer split for all compared models inside a task.

Total pilot outer evaluations:

```text
2 targets × 4 seeds × 5 folds = 40 outer splits
```

Each split should produce both the development same-solver baseline and the
Adaptive-PALF result on identical subjects.

Do not change this seed set after seeing results.

---

# 4. Models to compare in the pilot

For every development outer split run exactly:

## Model A — Development same-solver baseline

Generalized FC:

```text
rho_aniso = 0
tau_net = 0
```

Select `lambda_F` by nested inner CV.

Select SC `alpha_SC` using the expanded SC grid.

Generate:
- FP-only prediction
- SC-only prediction
- fully cross-fitted convex FP+SC fusion

This is the development baseline.

---

## Model B — Adaptive PALF

Select jointly:

```text
rho_aniso
tau_net
lambda_F
```

from the predefined grids using inner training data only.

Use the **same expanded SC branch and SC-selection procedure** as Model A.

Generate:
- Adaptive FP-only prediction
- SC-only prediction
- fully cross-fitted convex Adaptive-FP + SC prediction
- equal-weight FP+SC prediction for diagnostics only

Do not add a differential stack or any Phase-1 meta-model.

---

# 5. Fully nested fitting rules

Preserve the existing fully cross-fitted PALF protocol.

For each outer split:

1. Outer-test subjects are completely inaccessible to:
   - preprocessing
   - FC hyperparameter selection
   - SC hyperparameter selection
   - fusion-weight selection

2. Fusion OOF predictions must be generated with fold-local preprocessing and
   branch selection.

3. Final branch hyperparameters are reselected on the complete outer-training
   set using only its internal folds.

4. Outer-test predictions are generated exactly once.

5. Adaptive parameter selection (`rho_aniso`, `tau_net`, `lambda_F`) occurs
   only within the corresponding inner training data.

---

# 6. Conservative inner-selection rule

Phase 1 showed that small apparent OOF improvements can fail to generalize.
Do not use an aggressive test-like selector.

For FC branch hyperparameter selection, implement a deterministic
**one-standard-error simplicity rule**.

Within an inner CV search:

1. For every candidate combination, compute the mean validation Pearson `r`
   across the inner folds.
2. Let the highest mean be `r_best`.
3. Let `SE_best` be the standard error of the foldwise Pearson values of the
   best candidate.
4. Define the eligible set:

```text
mean_r >= r_best - SE_best
```

5. Among eligible candidates, choose the simplest in this exact order:

```text
lowest rho_aniso
then lowest tau_net
then smallest absolute log10 distance of lambda_F from 0.1
then smaller lambda_F
```

This deliberately prefers the no-prior geometry unless semantic regularization
has a sufficiently stable inner-CV advantage.

### Important

Also save the ordinary maximum-mean candidate as a diagnostic, but the executed
pilot prediction must use the one-SE-selected candidate.

Do not switch between selectors after seeing outer-test results.

---

# 7. Efficient implementation

The adaptive grid is larger than the previous four-condition ablation. Optimize
the implementation rather than launching a naive dense calculation.

## Required efficiency points

- Build the matched prior vector once per target.
- Build `D_p` once per target.
- Build `L_p` once per target.
- Never form an unnecessary dense 6670×6670 matrix if the current solver can
  exploit diagonal + active Laplacian structure.
- Cache any prior-independent preprocessing per fold.
- Cache the active Laplacian eigendecomposition/factorization when possible.
- Reuse SC computations between baseline and Adaptive PALF within the same
  outer split.
- Checkpoint after every completed outer split.
- Make the run restartable.
- Print timing after the first 2 completed outer splits and estimate total ETA.

Do not change the mathematical search space merely to make the run faster.

---

# 8. Separate code and output paths

Do not modify the frozen v2 output directories.

Add a dedicated implementation path, preferably:

```text
src/metascfc/experiments/palf_adaptive_prior.py
scripts_paper/phase2a_adaptive_prior_pilot.py
configs/iclr/palf_adaptive_prior_pilot.yaml
```

Output only to:

```text
outputs/iclr/palf_phase2a_adaptive_prior_pilot/
```

Create a ZIP:

```text
outputs/iclr/palf_phase2a_adaptive_prior_pilot.zip
```

Never overwrite:

```text
outputs/iclr/palf_crossfit_ablation_v1/
outputs/iclr/palf_same_solver_fusion_corrected_v2/
outputs/iclr/palf_manuscript_freeze_v2_same_solver/
outputs/iclr/palf_phase1_differential_stacking/
```

---

# 9. Required split-level outputs

Create:

```text
split_metrics.csv
```

with one row per target / development seed / outer fold / model.

Required columns:

```text
task
seed
outer_fold
model
n_train
n_test

fp_pearson
fp_rmse
fp_mae

sc_pearson
sc_rmse
sc_mae

fused_pearson
fused_rmse
fused_mae

equal_weight_pearson
equal_weight_rmse
equal_weight_mae

selected_lambda_F
selected_rho_aniso
selected_tau_net
selected_alpha_SC

selected_w_FP
selected_w_SC

inner_best_mean_r
inner_best_se_r
selected_mean_r
one_se_threshold
selected_is_simplest_eligible
```

For the baseline rows:

```text
selected_rho_aniso = 0
selected_tau_net = 0
```

---

# 10. Seed-level outputs

Average the five outer folds within each development seed.

Create:

```text
seed_metrics.csv
```

For each target / seed report:

### Baseline
- FP r
- fused r
- RMSE
- MAE

### Adaptive PALF
- FP r
- fused r
- RMSE
- MAE

### Paired deltas
- Adaptive FP minus baseline FP
- Adaptive fused minus baseline fused

There must be exactly 4 development seed summaries per target.

Do not run inferential significance tests and present them as final evidence;
four development seeds are a pilot.

---

# 11. Development diagnostics

Create:

```text
adaptive_selection_summary.csv
```

For each task report:

- distribution of selected `rho_aniso`
- fraction with `rho_aniso = 0`
- fraction with `0 < rho_aniso < 1`
- fraction with `rho_aniso = 1`
- distribution of selected `tau_net`
- fraction with `tau_net = 0`
- fraction with `tau_net > 0`
- joint frequency table of `(rho_aniso, tau_net)`
- lambda_F frequency
- alpha_SC frequency
- number of SC selections at old boundary `100`
- number above old boundary `100`

The central scientific question is whether the adaptive selector actually
chooses semantic regularization rather than always collapsing to no prior.

---

# 12. Compare FP-only and fused behavior

Phase 1 indicates that fusion can be a bottleneck.

Therefore do **not** look only at fused prediction.

For each target report:

```text
baseline FP mean
adaptive FP mean
FP paired delta

baseline fused mean
adaptive fused mean
fused paired delta

adaptive equal-weight mean
```

Also report:

```text
fraction of outer splits where adaptive FP > baseline FP
fraction where adaptive fused > baseline fused
fraction where adaptive FP > adaptive fused
```

This will determine whether Phase 2B should focus on:
- further FC regularization,
- or fusion.

Do not use these outer-test diagnostics to change Phase 2A itself.

---

# 13. Predefined go / no-go interpretation

At the end of the pilot, classify the outcome without changing thresholds after
seeing the data.

## STRONG GO

Return:

```text
PHASE2A_DECISION: STRONG_GO
```

if BOTH targets satisfy:

```text
mean adaptive fused delta >= +0.002
AND
adaptive fused > baseline fused in at least 3/4 development seeds
```

OR if one target satisfies the above and the other satisfies:

```text
mean adaptive FP delta >= +0.005
AND
adaptive FP > baseline FP in at least 3/4 seeds
AND
mean adaptive fused delta >= 0
```

This second clause means the FC model improved but fusion is the remaining
bottleneck.

## WEAK GO

Return:

```text
PHASE2A_DECISION: WEAK_GO
```

if:
- at least one target has positive mean fused delta and >=3/4 positive seeds,
  but the strong threshold is not met;
- OR both targets show clear FP improvement but fusion remains mixed.

## NO GO

Return:

```text
PHASE2A_DECISION: NO_GO
```

if:
- both targets have nonpositive mean adaptive FP and fused deltas;
- OR semantic adaptation is selected frequently but worsens test performance;
- OR the selector collapses to no prior almost everywhere with no predictive
  improvement.

These are development criteria only.

---

# 14. Plots

Generate:

```text
plots/fig_adaptive_seed_deltas.pdf/png
plots/fig_fp_vs_fused_deltas.pdf/png
plots/fig_rho_tau_selection.pdf/png
plots/fig_sc_alpha_selection.pdf/png
```

### Figure 1
Development seed deltas:
- adaptive minus baseline FP
- adaptive minus baseline fused

### Figure 2
Per-split:
- x = FP delta
- y = fused delta
This reveals whether fusion preserves or destroys FC improvements.

### Figure 3
Heatmap/count plot of selected:
- `rho_aniso`
- `tau_net`

### Figure 4
SC alpha selection frequency under the expanded grid.

---

# 15. Validation gates

Phase 2A is valid only if all gates pass.

## Split gates
- exactly 4 development seeds per target
- exactly 5 outer folds per seed
- baseline and adaptive models use identical train/test indices

## Baseline gates
- baseline has rho=0 and tau=0
- baseline uses same generalized solver as Adaptive PALF
- SC grid is identical between baseline and adaptive

## Adaptive gates
- `D_rho = (1-rho)I + rho D_p`
- rho=0 reproduces isotropic diagonal exactly
- tau=0 removes network penalty exactly
- rho=1 reproduces existing anisotropic diagonal
- current four historical conditions are representable in the new parameterization

## Leakage gates
- outer-test target never enters hyperparameter/fusion selection
- fold-local preprocessing remains intact
- outer test evaluated once

## One-SE gates
- candidate fold scores saved
- best SE computed from inner folds
- selected candidate belongs to eligible one-SE set
- simplicity ordering deterministic

## Output gates
- finite predictions and metrics
- restartable checkpoints
- no frozen source output modified

---

# 16. Tests

Add:

```text
tests/test_palf_adaptive_prior.py
```

At minimum test:

1. `rho=0` gives exact identity diagonal.
2. `rho=1` gives exact existing prior diagonal.
3. interpolation preserves mean diagonal approximately 1.
4. `tau=0` removes the network term.
5. `(rho=0,tau=0)` matches current same-solver objective.
6. `(rho=1,tau=0)` matches anisotropy-only objective.
7. `(rho=0,tau=current_lambda)` matches network-only objective.
8. `(rho=1,tau=current_lambda)` matches Full-PALF objective.
9. one-SE selector returns an eligible candidate.
10. one-SE ties prefer lower rho.
11. then lower tau.
12. no outer-test target is passed to selection helpers.
13. baseline/adaptive split indices are identical.
14. SC selection grid is identical for both models.
15. no Phase-1 stacking helper is imported or invoked.
16. restart/checkpoint logic preserves completed splits.

Run targeted tests, then the full suite.

---

# 17. Do NOT do in Phase 2A

Do not yet:

- run final 10-seed ICLR evaluation
- reuse seeds 0–9 for adaptive-model development
- tune gamma
- tune top-K ROIs
- change product lifting
- regenerate LLM priors
- add differential stacking
- add a neural network
- choose task-specific grids
- reconstruct biomarkers
- edit the paper
- declare SOTA
- run significance claims from only four development seeds

This pilot has one job:

> determine whether soft prior trust inside the generalized FC solver is
> promising enough for a final fresh-seed run.

---

# 18. Final OpenCode report

Print exactly:

## A. Environment and code
- git HEAD
- script
- config
- development seeds
- number of completed splits
- runtime
- estimated equivalent 10-seed runtime

## B. Baseline reproduction on development seeds
For each target:
- baseline FP r
- baseline fused r
- baseline RMSE
- baseline MAE

## C. Adaptive PALF development results
For each target:
- adaptive FP r
- adaptive fused r
- equal-weight r
- RMSE
- MAE

## D. Paired development deltas
For each target:
- mean FP delta
- FP positive seeds / 4
- mean fused delta
- fused positive seeds / 4
- fraction positive outer splits

No final p-value claim.

## E. Current within-pilot ranking
For each target state:
- whether Adaptive FP beats baseline FP
- whether Adaptive fused beats baseline fused
- whether Adaptive FP beats Adaptive fused

## F. Adaptive parameter use
For each target:
- rho frequency
- tau frequency
- joint rho/tau frequency
- fraction with any semantic prior
- lambda_F frequency
- alpha_SC frequency
- fraction alpha_SC > 100

## G. Fusion preservation diagnostic
- Pearson correlation between split-level FP delta and fused delta
- number of splits where FP improves but fusion worsens
- plot path

## H. Tests
- targeted tests
- full suite
- pre-existing failures

## I. Output bundle
- directory
- ZIP
- split CSV
- seed CSV
- selection summary
- plots

Then print exactly one:

```text
PHASE2A_DECISION: STRONG_GO
```

```text
PHASE2A_DECISION: WEAK_GO
```

or

```text
PHASE2A_DECISION: NO_GO
```

according to the predefined rules.

Finally end with:

```text
STATUS: PHASE2A_ADAPTIVE_PRIOR_PILOT_COMPLETE
```

If any validity gate fails, instead end with:

```text
STATUS: PHASE2A_NOT_VALID
```

and do not interpret the prediction results.
