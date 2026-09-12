# PALF ICLR 2027 — Phase 3A-FIX
# Correct PG-MT-BCR Development on 412 Subjects + Keep 98-Subject Holdout Sealed

## Executive status

The existing Phase-3A result must be marked:

```text
INVALID_FOR_PHASE3A_DECISION
```

The 98-subject holdout remains correctly sealed and MUST remain untouched.

A direct audit of the Phase-3A code found that the run did not implement the
predeclared Phase-3A protocol. The major problems are listed below.

This is a corrective rerun of the SAME PG-MT-BCR hypothesis, not a new
architecture search.

If the corrected Phase-3A-FIX fails the original scientific gates, leave the
98-subject holdout sealed and stop model development on the 412-subject cohort.

---

# 0. Preserve and verify the holdout seal

The holdout manifest must remain exactly:

```text
count  = 98
unique = 98
canonical SHA256 =
89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425
```

Do not:
- load holdout FC;
- load holdout SC;
- load holdout labels;
- compute holdout distributions;
- run R0 on holdout;
- inspect any holdout metric.

Verify:

```text
set(development_412) ∩ set(holdout_98) = empty
```

Create the new output directory only:

```text
outputs/iclr/palf_phase3a_fix_pg_mt_bcr/
```

Do not modify the old Phase-3A output.

---

# 1. Mandatory forensic audit of the old Phase-3A run

Create:

```text
OLD_PHASE3A_FORENSIC_AUDIT.md
```

Confirm the following issues with exact code locations.

## F1 — baseline was not corrected R0

The old Phase-3A code created:

```python
X = np.hstack([FC_edges, SC_edges])
```

and fit ordinary Ridge.

That is not the paper baseline.

The required R0 is:

```text
generalized same-solver no-prior FP
+
SC Ridge
+
fully cross-fitted convex FP+SC fusion
```

from:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

The old audit reported approximately:

```text
WM r    = 0.22257
FI r    = 0.35239
WM RMSE = 11.7580
FI RMSE = 4.6885
```

while calling the audit `PASS`.

That is not acceptable.

---

## F2 — audit tolerances were weakened

The prompt required:

```text
Pearson tolerance <= 5e-4
RMSE tolerance    <= 0.05
```

but the old code used approximately:

```text
Pearson tolerance = 0.05
RMSE tolerance    = 0.5
```

Restore the strict tolerances.

---

## F3 — only one inner split was used

The old code repeatedly used:

```python
it, iv = next(iter(ikf.split(...)))
```

instead of generating predictions across all three inner folds.

The corrected model-selection score must use concatenated 3-fold inner OOF
predictions.

---

## F4 — inner preprocessing leakage

The old code standardized FC/SC edges once on the entire outer-training set and
then reused those matrices inside inner validation.

Correct behavior:
- every inner candidate fit gets FC scaler fit on inner-train only;
- SC scaler fit on inner-train only;
- inner validation transformed with those training statistics.

Residual target mean/std must also be inner-training-only.

---

## F5 — the wrong R0 residual generator was used

Old residuals came from ordinary concatenated FC+SC Ridge OOF predictions.

Correct residuals must come from the exact corrected R0 prediction procedure.

For any residual-model training scope S, every subject in S must receive a
baseline prediction from an R0 model that did not train on that subject.

---

## F6 — single-task WM result was overwritten

The old loop wrote:

```python
res[mid] = ...
```

once for WM and then overwrote the same model entry when FI was processed.

As a result, B0/B1 WM was often exactly A0.

Correct ST result storage must preserve both task predictions:

```text
B0:
  pW = ST-BCR-NP WM correction
  pF = ST-BCR-NP FI correction

B1:
  pW = ST-PG-BCR WM correction
  pF = ST-PG-BCR FI correction
```

---

## F7 — ST prior-strength grid was not searched

The old ST prior model effectively used only one prior strength.

Correct B1 must search:

```text
lambda_amp   = [0.1, 1.0, 10.0]
lambda_prior = [0.01, 0.1, 1.0]
alpha        = [0, 0.25, 0.50, 0.75, 1.0]
```

---

## F8 — multi-task shared amplitudes were accidentally shared across tasks

The intended model uses shared SPATIAL factors but task-specific amplitudes.

The old implementation used the same shared amplitudes for WM and FI:

```text
afc_sh
asc_sh
```

for both targets.

Correct model must use:

```text
amp_fc_shared_WM[2]
amp_fc_shared_FI[2]

amp_sc_shared_WM[2]
amp_sc_shared_FI[2]
```

Shared spatial factors remain common, but their contribution magnitude may
differ by task.

---

## F9 — optimizer did not match the frozen procedure

The Phase-3A prompt specified deterministic full-batch Adam, but the old runner
used L-BFGS with different iteration rules.

Use the frozen Adam procedure in this corrected run.

Do not choose between Adam and L-BFGS based on performance.

---

## F10 — requested biomarker/optimizer outputs were omitted

The old result ZIP did not contain the requested:
- optimizer diagnostics;
- coefficient exports;
- biomarker stability outputs;
- selection-details table.

The corrected run must produce these.

---

# 2. Correct R0 implementation — no approximation allowed

Import and reuse the validated code in:

```text
src/metascfc/experiments/palf_crossfit_ablation.py
```

At minimum reuse:

```python
CONDITIONS["R0"]
generate_crossfit_oof
search_fusion_weights
reselect_and_fit_final
make_fusion_folds
make_inner_selection_folds
```

For an outer split `(train_idx, test_idx)` and task y:

```python
condition = CONDITIONS["R0"]
uniform_prior = np.ones(116) / 116
```

Generate cross-fitted training predictions:

```python
oof = generate_crossfit_oof(
    X_fc,
    X_sc,
    y,
    train_idx,
    condition,
    uniform_prior,
    seed,
    outer_fold,
    ridge_grid=[0.001,0.01,0.1,1,10,100],
    n_fusion_folds=3,
    n_inner=3,
    n_rois=116,
)
```

Use:

```python
base_weights, _ = search_fusion_weights(
    y[train_idx],
    {"FP": oof.fp_oof, "SC": oof.sc_oof},
    ["FP","SC"],
)
```

Training-scope OOF baseline prediction:

```python
base_oof =
    base_weights["FP"] * oof.fp_oof
    + base_weights["SC"] * oof.sc_oof
```

Final outer-test baseline:

```python
_, final_sc, final_fp = reselect_and_fit_final(
    ...
    condition=CONDITIONS["R0"],
    n_final_cv=3,
)
```

```python
base_test =
    base_weights["FP"] * final_fp.test_pred
    + base_weights["SC"] * final_sc.test_pred
```

Do NOT use:
- concatenated FC+SC Ridge;
- ordinary FC Ridge as R0;
- `fc_oof` instead of `fp_oof`.

---

# 3. Strict frozen baseline audit

Using historical seeds 0–9 and five outer folds, reproduce:

```text
WM fused r    = 0.263515 ± tolerance
WM RMSE       = 11.292921 ± tolerance

FI fused r    = 0.370917 ± tolerance
FI RMSE       = 4.566689 ± tolerance
```

Hard tolerances:

```text
abs Pearson error <= 5e-4
abs RMSE error    <= 0.05
```

If any fail:

```text
STATUS: PHASE3A_FIX_BASELINE_AUDIT_FAILED
```

and stop.

Do not continue to BCR.

---

# 4. Development and holdout cohorts

Development cohort remains the exact 412 subjects already used historically.

Holdout remains the exact sealed 98-subject manifest.

Hard assertions:

```text
len(dev_ids) == 412
len(holdout_ids) == 98
dev ∩ holdout == empty
holdout SHA256 ==
89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425
```

Phase-3A-FIX may only call data loaders with development IDs.

Add an access logger. Every subject-ID batch requested by Phase-3A-FIX must be
written to:

```text
PHASE3A_DATA_ACCESS_LOG.jsonl
```

At completion assert no holdout ID occurred.

---

# 5. Keep the scientific PG-MT-BCR architecture fixed

Do not change ranks or priors.

For each modality:

```text
m ∈ {FC, SC}
```

Multi-task model has:

```text
2 shared spatial factors
1 WM-specific spatial factor
1 FI-specific spatial factor
```

Single-task model has:

```text
3 task-specific factors per task
```

so the per-task effective rank is matched.

The connectome is symmetric.

For factor `u`:

\[
g_u(M) = u^\top M u.
\]

---

# 6. Correct multi-task prediction equation

For task t:

\[
\hat r_t
=
b_t
+
\sum_{m\in\{FC,SC\}}
\left[
\sum_{k=1}^2
a^{shared}_{m,t,k}
u^{shared\top}_{m,k}
M_m
u^{shared}_{m,k}
+
a^{specific}_{m,t}
u^{specific\top}_{m,t}
M_m
u^{specific}_{m,t}
\right].
\]

Required parameter shapes:

```text
U_fc_shared : (116,2)
U_sc_shared : (116,2)

U_fc_WM : (116,1)
U_fc_FI : (116,1)
U_sc_WM : (116,1)
U_sc_FI : (116,1)

amp_fc_shared_WM : (2,)
amp_fc_shared_FI : (2,)
amp_sc_shared_WM : (2,)
amp_sc_shared_FI : (2,)

amp_fc_WM : scalar
amp_fc_FI : scalar
amp_sc_WM : scalar
amp_sc_FI : scalar
```

Never use the same shared amplitude vector for both tasks.

---

# 7. Correct single-task prediction equation

For each task separately:

```text
3 FC factors
3 SC factors
```

with independent amplitudes.

No shared factors exist in ST.

Do not let the FI fit overwrite the WM result.

---

# 8. Factor projection

After every optimizer step:

1. L2 normalize every factor column;
2. for the two shared FC factors, deterministic Gram-Schmidt;
3. renormalize;
4. repeat for the two shared SC factors.

Hard numerical validation:

```text
abs(norm(u)-1) <= 1e-8
abs(dot(shared_factor_1, shared_factor_2)) <= 1e-8
```

after projection.

The test must assert these actual tolerances, not broad ranges such as
`0.5 < norm < 2`.

---

# 9. Prior penalty

Keep exactly:

```text
epsilon = 1e-3
gamma   = 0.5
```

For ROI prior p:

\[
d_i(p)
=
\frac{(\epsilon+p_i)^{-\gamma}}
{\operatorname{mean}_j(\epsilon+p_j)^{-\gamma}}
\]

and:

\[
D(p)=diag(d_i).
\]

Shared prior:

\[
p_{shared}
=
normalize_{0,1}
\left(
\sqrt{(p_{WM}+10^{-6})(p_{FI}+10^{-6})}
\right).
\]

Matched MT prior penalty:

\[
R_{prior}
=
\sum_{m,k}
u_{m,k}^{shared\top}D(p_{shared})u_{m,k}^{shared}
+
\sum_m
u_{m,WM}^{\top}D(p_{WM})u_{m,WM}
+
\sum_m
u_{m,FI}^{\top}D(p_{FI})u_{m,FI}.
\]

ST uses only the matched task prior.

Architecture-matched no-prior models use:

```text
lambda_prior = 0
```

with the SAME architecture and optimizer.

---

# 10. Training objective

Residuals are standardized using statistics fit on the current BCR-training
scope only.

For multi-task:

\[
L =
MSE(zr_{WM}, \hat r_{WM})
+
MSE(zr_{FI}, \hat r_{FI})
+
\lambda_{amp}R_{amp}
+
\lambda_{prior}R_{prior}.
\]

`R_amp` contains every amplitude coefficient.

Intercepts are not penalized.

For ST, use the corresponding task only.

---

# 11. Frozen optimizer

Use full-batch PyTorch Adam exactly:

```text
dtype         = float64
learning_rate = 0.01
max_steps     = 1500
min_steps     = 200
patience      = 75
relative training-loss tolerance = 1e-7
gradient clip norm = 5.0
restarts      = 2
```

Stopping may use TRAINING penalized loss only.

After each `optimizer.step()` call the factor-projection function.

Choose restart by final TRAINING penalized objective only.

No validation metric may choose optimizer restart.

Record every restart in:

```text
optimizer_diagnostics.csv
```

with:
- model;
- seed/fold;
- CV scope;
- hyperparameters;
- restart;
- final loss;
- steps;
- converged;
- chosen flag.

---

# 12. Correct leakage-safe nested R0/BCR hierarchy

This section is mandatory.

Create reusable cached R0 helpers.

## 12A. `fit_r0_predict(train_idx, val_idx, scope_seed)`

This function:
- uses only `train_idx` for all R0 selection;
- predicts only `val_idx`;
- uses the validated R0 FP+SC procedure.

No `val_idx` labels enter model selection.

## 12B. `crossfit_r0_within(scope_idx, scope_seed, n_folds=3)`

Partition `scope_idx` into 3 folds.

For each held-out fold C:
- train/tune R0 only on B = scope minus C;
- predict C.

Return a prediction for every subject in `scope_idx`.

Thus every residual target is:

```text
y(subject) - R0 prediction from a model not trained on subject
```

Cache these predictions by:
- task;
- outer seed;
- outer fold;
- inner fold/scope hash.

Do not recompute them separately for every BCR candidate.

---

# 13. Correct nested BCR preprocessing

The raw development FC/SC matrices may be stored once.

But every BCR fit must do:

For training scope B:
1. vectorize upper-triangle FC;
2. fit FC edge mean/std on B only;
3. transform B and validation C;
4. reconstruct symmetric FC matrices;

and separately:
5. fit SC edge mean/std on B only;
6. transform B and C;
7. reconstruct symmetric SC matrices.

Never use outer-training scalers inside inner validation.

For final outer fit:
- fit scalers on all outer-training T;
- transform outer-test V.

Add index-spy tests proving this.

---

# 14. Correct 3-fold inner OOF candidate evaluation

For each outer split T/V:

```text
inner folds = 3
```

For every candidate hyperparameter tuple:

1. initialize empty OOF residual-prediction arrays over T;
2. for each inner fold B/C:
   - compute `crossfit_r0_within(B)` for leakage-safe B residual targets;
   - compute `fit_r0_predict(B,C)` for baseline prediction on C;
   - fit BCR scalers on B only;
   - standardize B residual targets using B-only statistics;
   - train BCR candidate on B;
   - predict residual on C;
   - convert predicted residual back to raw target units;
   - store prediction at C positions;
3. after all 3 folds, every T subject has one BCR residual prediction and one
   base R0 prediction;
4. ONLY NOW score candidate/alpha on the concatenated OOF predictions.

Do not:
- score just one inner fold;
- average hyperparameters chosen independently per inner fold;
- use outer-test V in any selection.

---

# 15. Hyperparameter grids remain frozen

No-prior ST/MT:

```text
lambda_amp = [0.1, 1.0, 10.0]
```

Prior-aware ST/MT:

```text
lambda_amp   = [0.1, 1.0, 10.0]
lambda_prior = [0.01, 0.1, 1.0]
```

Residual amplitude:

```text
alpha = [0.0, 0.25, 0.50, 0.75, 1.0]
```

For MT, alpha is task-specific, so evaluate the 25 predefined
`(alpha_WM, alpha_FI)` combinations after each fitted BCR candidate.

No rank tuning.

No gamma tuning.

No new hyperparameters.

---

# 16. Correct selection criterion

For MT candidate + alpha pair:

Compute concatenated 3-fold OOF final predictions:

```text
base_oof_WM + alpha_WM * residual_oof_WM
base_oof_FI + alpha_FI * residual_oof_FI
```

Candidate is eligible only if:

```text
delta_r_WM >= -0.002
delta_r_FI >= -0.002
```

against concatenated R0 inner OOF.

Among eligible candidates choose:

1. highest mean Fisher-z across WM/FI;
2. lower mean normalized RMSE;
3. lower mean normalized MAE;
4. smaller `alpha_WM + alpha_FI`;
5. lower `lambda_prior`;
6. larger `lambda_amp`;
7. deterministic tuple ordering.

Implement ALL tie-breaks.

For ST:
1. Pearson;
2. RMSE;
3. MAE;
4. smaller alpha;
5. lower lambda_prior;
6. larger lambda_amp.

If no prior-aware MT candidate is eligible:

```text
alpha_WM = 0
alpha_FI = 0
```

and final predictions equal R0 exactly.

---

# 17. Correct final outer refit

After inner selection on T:

1. obtain exact R0 outer-test prediction from T→V;
2. obtain `crossfit_r0_within(T)` for leakage-safe residual targets on T;
3. fit FC/SC scalers on all T;
4. standardize T residual targets using T-only residual mean/std;
5. train the selected BCR model on all T;
6. predict residuals on V;
7. de-standardize residual predictions using T residual statistics;
8. form:

```text
final_WM = R0_test_WM + alpha_WM * residual_test_WM
final_FI = R0_test_FI + alpha_FI * residual_test_FI
```

For alpha=0, hard assert bitwise/numerical equality to R0.

---

# 18. Development seeds

The old Phase-3A seeds:

```text
3535,3636,3737,3838
```

were inspected under an invalid implementation.

For the corrected decision use fresh development seeds:

```text
PHASE3A_FIX_DEV_SEEDS = [3939, 4040, 4141, 4242]
```

five outer folds.

Do not change them after results.

Historical 0–9 are baseline-audit only.

The 98-subject holdout remains untouched.

---

# 19. Prior controls

Freeze exactly:

```text
cross-task
shuffled
random
```

as in the original Phase-3A prompt.

Shuffled:
```text
WM seed = 8101
FI seed = 8102
```

Random:
```text
WM seed = 8103
FI seed = 8104
```

The controls use:
- identical architecture;
- identical nested CV;
- identical optimizer;
- identical grids.

Do not select the best control.

---

# 20. Models

Report exactly:

```text
A0  corrected R0

B0  ST-BCR-NP
B1  ST-PG-BCR

C0  MT-BCR-NP
C1  PG-MT-BCR matched

C2  PG-MT-BCR cross-task
C3  PG-MT-BCR shuffled
C4  PG-MT-BCR random
```

No other prediction architecture.

---

# 21. Required parameter/selection export

Create:

```text
selection_details.csv
```

One row per:
- outer seed;
- outer fold;
- model;
- candidate;
- inner fold / aggregate;
- lambda_amp;
- lambda_prior;
- alpha_WM;
- alpha_FI;
- WM OOF Pearson/RMSE/MAE;
- FI OOF Pearson/RMSE/MAE;
- joint Fisher score;
- eligible;
- selected.

This file is mandatory for independent audit.

---

# 22. Coefficient extraction and prediction reconstruction

For each selected final BCR fit export factor/amplitude arrays.

Construct:

\[
B_{m,t}^{shared}
=
\sum_{k=1}^2
a^{shared}_{m,t,k}
u^{shared}_{m,k}u^{shared\top}_{m,k}
\]

\[
B_{m,t}^{specific}
=
a^{specific}_{m,t}
u^{specific}_{m,t}u^{specific\top}_{m,t}
\]

and:

\[
B_{m,t}=B_{m,t}^{shared}+B_{m,t}^{specific}.
\]

Set diagonal to zero only for reporting.

Validate direct residual prediction from the factors against matrix inner
product using the saved coefficient matrix.

Because for symmetric zero-diagonal M:

```text
direct factor prediction
==
sum over full matrix B * M
```

within numerical tolerance.

Hard reconstruction error:

```text
max_abs_error <= 1e-8
```

Save:
- full 116x116 maps;
- 6670 upper-triangle maps;
- shared;
- specific;
- alpha-weighted final residual maps.

---

# 23. Biomarker diagnostics

Compute on development only; never use for model selection.

For C0 and C1:
- mean pairwise Spearman of abs 6670-edge map;
- top-100 Jaccard;
- top-300 Jaccard;
- top-10 ROI Jaccard;
- top-20 ROI Jaccard;
- sign consistency.

ROI score:

\[
s_i=\sum_{j\ne i}|B_{ij}|.
\]

Also separately evaluate:
- shared maps;
- WM-specific maps;
- FI-specific maps;
- FC;
- SC.

If alpha for a task is zero, mark the final residual biomarker as
`ABSTAINED` rather than treating an unused map as a final biomarker.

---

# 24. Faithfulness diagnostic

Development only.

Using outer-test development folds:
- top 5 ROIs;
- top 10 ROIs;
- bottom same-size;
- 50 deterministic random same-size sets.

Mask standardized FC/SC edges incident to the selected ROIs.

Do not retrain.

Report final prediction RMSE degradation.

Compare C1 versus C0.

Do not use faithfulness to choose hyperparameters.

---

# 25. Correct gates — unchanged scientific threshold

Use the original scientific gates.

## Gate 1 — final predictive improvement

For BOTH tasks:

```text
mean(C1-A0) >= +0.005
positive seeds >= 3/4
```

and at least one target:

```text
mean(C1-A0) >= +0.010
```

## Gate 2 — architecture-matched prior value

For BOTH tasks:

```text
mean(C1-C0) >= +0.003
positive seeds >= 3/4
```

## Gate 3 — prior specificity

For BOTH tasks:

```text
mean(C1-C3) > 0
mean(C1-C4) > 0
```

and matched must not be worse than cross-task on BOTH tasks.

## Gate 4 — stability

- finite predictions;
- >=95% selected final optimizer fits valid;
- no catastrophic RMSE >2× R0.

All four pass:

```text
PHASE3A_FIX_DECISION: FREEZE_AND_UNLOCK_PHASE3B
```

Anything else:

```text
PHASE3A_FIX_DECISION: DO_NOT_TOUCH_HOLDOUT
```

Do not change thresholds.

---

# 26. Automatic freeze only if corrected gates pass

If all gates pass, create:

```text
MODEL_FROZEN.json
MODEL_FROZEN.sha256
READY_FOR_PHASE3B
```

Record:
- git HEAD;
- diff hash;
- environment;
- development/holdout hashes;
- priors/control hashes;
- exact model equation/version;
- optimizer;
- grids;
- ranks;
- scaling;
- CV selection algorithm;
- alpha grid;
- gates;
- planned Phase-3B metrics.

If any gate fails:

```text
HOLDOUT_REMAINS_LOCKED
```

and do not create `MODEL_FROZEN.json`.

---

# 27. Required output bundle

Create:

```text
outputs/iclr/palf_phase3a_fix_pg_mt_bcr/
```

with:

```text
COMPLETE
OLD_PHASE3A_FORENSIC_AUDIT.md
BASELINE_AUDIT.json
HOLDOUT_SEAL_REPORT.json
VALIDATION_REPORT.json
RUN_REPORT.md
PHASE3A_DATA_ACCESS_LOG.jsonl

split_metrics.csv
seed_metrics.csv
model_summary.csv
selection_details.csv
optimizer_diagnostics.csv

coefficients/
predictions/
biomarkers/
prior_controls/
plots/
```

Zip:

```text
outputs/iclr/palf_phase3a_fix_pg_mt_bcr.zip
```

---

# 28. Tests — functional, not tautological

Add:

```text
tests/test_phase3a_fix_pg_mt_bcr.py
```

Minimum tests:

1. holdout count=98 and exact canonical SHA.
2. dev count=412.
3. dev/holdout intersection empty.
4. Phase-3A-FIX loader raises for every holdout ID.
5. access log contains no holdout ID.
6. baseline uses `CONDITIONS["R0"]`.
7. baseline fusion uses `fp_oof`, not `fc_oof`.
8. baseline uses FP+SC convex fusion.
9. audit WM r within 5e-4 of 0.263515.
10. audit FI r within 5e-4 of 0.370917.
11. audit WM RMSE within 0.05 of 11.292921.
12. audit FI RMSE within 0.05 of 4.566689.
13. no concatenated FC+SC ordinary Ridge is used as A0.
14. every inner candidate has predictions covering all three inner validation folds.
15. inner FC scaler sees inner-train indices only.
16. inner SC scaler sees inner-train indices only.
17. residual standardization uses B-only statistics.
18. residual target on B comes from cross-fitted R0.
19. R0 validation prediction C is produced from B-only data.
20. ST WM and ST FI outputs coexist and are not overwritten.
21. B1 searches all 3×3 lambda combinations.
22. MT shared spatial factors are shared.
23. MT shared amplitude arrays are task-specific.
24. factor norms within 1e-8 of 1 after projection.
25. shared factor dot products <=1e-8.
26. Adam settings match frozen configuration.
27. restart chosen by training objective only.
28. no validation metric chooses optimizer restart.
29. C0 lambda_prior exactly 0.
30. C1 matched priors match frozen files.
31. controls deterministic.
32. alpha=0 exactly equals corrected R0.
33. all 25 MT alpha pairs are evaluated.
34. concatenated three-fold OOF drives candidate selection.
35. eligibility safeguard uses concatenated OOF, not one fold.
36. final refit uses full T and T-cross-fitted residuals.
37. BCR coefficient reconstruction <=1e-8.
38. biomarker diagnostics are downstream only.
39. MODEL_FROZEN only if Gates 1–4 all pass.
40. no holdout evaluation function is called or imported.

Do not use tests such as:

```python
assert condition or True
```

or:
```python
assert "val" not in src or "val" in src
```

Every test must be capable of failing.

Run targeted and full suite.

---

# 29. Final OpenCode report

Print:

## A. Holdout seal
- count
- SHA
- dev intersection
- access-log audit
- confirmation no holdout features/labels touched

## B. Old Phase-3A forensic audit
Report F1–F10 as confirmed/not confirmed with locations.

## C. Correct baseline audit
- WM r/RMSE
- FI r/RMSE
- strict tolerance
- PASS/FAIL

## D. Correct PG-MT-BCR implementation
- shared factor ranks
- task-specific ranks
- task-specific shared amplitudes
- optimizer settings
- grids
- factor projection checks
- reconstruction max error

## E. Correct development results
For WM/FI:

```text
A0
B0
B1
C0
C1
C2
C3
C4
```

Pearson/RMSE/MAE.

## F. Seed-level decomposition
- C1-A0
- C1-C0
- C0-B0
- C1-B1
- B1-B0
- C1-C2
- C1-C3
- C1-C4

four seed deltas + mean + median + positive count.

## G. Selection diagnostics
- alpha frequencies
- lambda_amp frequencies
- lambda_prior frequencies
- fallback/abstention frequency
- percentage candidates passing eligibility

## H. Optimizer diagnostics
- total fits
- convergence fraction
- restarts
- failed fits
- runtime

## I. Biomarker readiness
- C0/C1 stability
- ROI Jaccards
- shared/specific stability
- faithfulness

## J. Gate evaluation
Gates 1–4 separately.

## K. Freeze/lock
Either:
```text
READY_FOR_PHASE3B
```
or:
```text
HOLDOUT_REMAINS_LOCKED
```

## L. Tests
- targeted
- full suite
- unrelated pre-existing failures

## M. Outputs
all paths.

Then exactly one:

```text
PHASE3A_FIX_DECISION: FREEZE_AND_UNLOCK_PHASE3B
```

or:

```text
PHASE3A_FIX_DECISION: DO_NOT_TOUCH_HOLDOUT
```

Finally:

```text
STATUS: PHASE3A_FIX_COMPLETE
```

If any validity gate fails:

```text
STATUS: PHASE3A_FIX_NOT_VALID
```

and holdout remains sealed.

---

# 30. Integrity stop condition

This corrective run is justified because the old Phase-3A implementation did
not execute the predeclared model/evaluation.

If the corrected Phase-3A-FIX is a valid `DO_NOT_TOUCH_HOLDOUT`, STOP.

Do not:
- introduce Phase 3C;
- alter ranks;
- alter priors;
- loosen gates;
- inspect the 98 holdout.

The 98-subject holdout is reserved only for a model that passes the original
development gates under a valid implementation.
