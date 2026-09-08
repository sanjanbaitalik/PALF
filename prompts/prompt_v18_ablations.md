# PALF — implement and run fully cross-fitted, equivalent-scale ablations

You are OpenCode working in the author's current local PALF checkout:
https://github.com/sanjanbaitalik/PALF

## 1. Deliver the experiment, not just a plan

Implement, test, and run the corrected evaluation described below. Continue from correctness tests through the real-data smoke run to the complete production experiment, then generate the analysis and review bundle. The author authorizes this complete local workflow. Do not stop after proposing edits, passing tests, or printing a launch command. Do not ask for another confirmation between successful stages. If a genuine missing-data, access, hardware, or numerical problem prevents completion, preserve resumable progress and report the exact blocker; never report an unfinished run as complete.

This is a NEW scientific evaluation. The earlier plotting-only instruction is superseded for this task: new training and nested model selection are required. Preserve all existing results and priors. The author will return the new outputs and a Gemini NanoBanana diagram for a later manuscript revision. Do not edit the main manuscript, supplement, bibliography, or method-overview image in this task.

The two required scientific changes are:

1. Every branch prediction used to train late fusion must exclude its held-out subjects from preprocessing, hyperparameter selection, and branch fitting. Final branch parameters must be reselected on the complete outer-training set.
2. Retuned component ablations must separate anisotropy and the line-graph penalty while keeping the generalized solver's effective penalty scale, input data, partitions, and fusion protocol equivalent.

The reviewed baseline is commit `67796e8ffd0f6026550532ec146d1d5ae1312b91`. Despite its commit message, it changes plotting and does not repair these training issues. Inspect the actual checkout before editing; a newer local implementation may already resolve some requirements. Do not reset or downgrade the checkout. Verify any claimed existing fix using the tests below.

Success means complete, reproducible, correctly isolated experiments. It does not require PALF to outperform a control or yield a significant p-value. Retain negative, null, and mixed findings.

## 2. Inspect and protect the repository

Record `git status --short`, `git rev-parse HEAD`, and the recent commit history. Read applicable `AGENTS.md` files. Preserve unrelated local changes. Capture the implementation diff and its hash, as well as the starting commit, in the new run's provenance. A commit hash alone is insufficient if the run uses uncommitted changes. Do not push, merge, or upload results to an external service.

Inspect these current files and their actual call paths:

- `src/metascfc/experiments/prior_aware_late_fusion.py`
- `src/metascfc/experiments/lf1_final_experiment.py`
- `src/metascfc/models/iclr_backbones/modality_selective_anisotropic_ncr.py`
- `src/metascfc/models/iclr_backbones/network_constrained_ridge.py`
- `scripts/114_run_lf1_final_10x5.py`
- `scripts/115_evidence_audit_biomarker.py`
- `configs/iclr/lf1_final_10x5.yaml`
- the late-fusion and MS-A-NCR tests under `tests/`
- existing split manifests and frozen result structures under `outputs/iclr/lf1_final_10x5/`
- `scripts_paper/` and any available `PALF_revision_guide.md` / `REVISION_CHECKS.md`.

Known problems to verify in this checkout:

- `evaluate_outer_split` / `compute_oof_branch` choose FP parameters using all outer-training labels and fit an FP scaler on all outer-training subjects before generating fusion-training predictions.
- Some Ridge helpers scale the OOF training subset before their deeper parameter-selection CV; those selection-validation observations must also be excluded from scaler fitting.
- The ordinary FC/SC final refits reuse the first OOF fold's alpha instead of reselecting it on all outer-training subjects.
- The final runner hard-codes product lifting and gamma 0.5 despite a broader YAML grid. For the new experiment, make the configuration truthful and executable.
- The old runner has a machine-specific `BASE` and creates its old output directory at import time. Do not reuse it as a new configurable CLI without fixing those behaviors.
- The generalized FC solver divides its kernel by `2 * n_edges`, including in FC-only mode. Matching its lambda numerically to sklearn Ridge alpha is not an equivalent-scale control.
- An old audit has incorrect Holm correction and correlations of `argsort` index sequences mislabeled as rank stability. Do not inherit these reporting errors.

Before any experiment, save SHA-256 inventories for existing files under:

- `outputs/iclr/lf1_final_10x5/`
- `outputs/iclr/lf1_final_evidence_audit/`
- `outputs/priors/`
- the exact input arrays, subject manifest, and configuration read by the new run.

Check the protected inventories again at completion. All existing experimental artifacts and prior files must remain byte-identical. Write new code/configuration separately wherever practical; a necessary shared-code repair is allowed with regression tests and a recorded diff. Never overwrite historical CSV, PKL, NPY, NPZ, figures, or evidence files.

Use a fresh root such as `outputs/iclr/palf_crossfit_ablation_v1/<run_id>/`. Put smoke outputs in a separate directory that cannot be mistaken for production. Do not mix any old predictions or pilot rows into the new results.

## 3. Fixed data, priors, and outer evaluation

Preserve the existing final cohort, subject order, FC/SC preprocessing inputs, atlas, and targets:

| Item | Required value/source |
|---|---|
| Cohort | The existing 412-subject HCP cohort; verify rather than silently filter |
| Atlas | AAL116 |
| Features per modality | 6,670 unique off-diagonal upper-triangle edges |
| FC | `inputs/dataset_FC/FC_all.npy` |
| SC | `inputs/dataset_SC/SC_all.npy` |
| Subject order | `inputs/dataset_SC/hcp_subjects_used.csv` |
| Working Memory | `ListSort_Unadj`, `inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy` |
| Fluid Intelligence | `PMAT24_A_CR`, `inputs/dataset_SC/label_all.npy` |
| Outer CV | Seeds 0–9, five folds per seed, subject-wise |
| Fusion OOF CV | Three folds within each outer-training set |
| Deeper branch-selection CV | Three folds inside each fusion-OOF training subset |
| Final branch-selection CV | Three folds over the complete outer-training set |

Load the saved outer train/test indices when available and validate them against subject ordering. If reconstructing the historical shuffled KFold partitions from seeds is necessary, compare every partition with the archived indices before production. Do not silently change partitions. Persist the complete nested split manifest, including global subject-row indices and the nesting hierarchy. Use identical split manifests for all four component conditions and both modalities. Both tasks should use the same outer partitions if the archived cohort and ordering agree.

Validate dimensions, finite data, target lengths, unique subject IDs, FC/SC alignment, upper-triangle ordering, and label-to-subject alignment. Do not infer that equal array lengths prove subject correspondence. Resolve any discrepancy from local manifests before fitting. Do not fabricate or drop subjects to reach 412.

The current primary protocol is subject-wise. Do not relabel it family-grouped or infer family independence. A family-aware or external-cohort evaluation is outside this task and must remain a stated limitation where relevant.

Use the frozen prior CSVs from the existing final configuration:

- WM matched: `outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv`
- Fluid matched: `outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv`
- Cross-task: the other target's matched prior, retaining internal key `unrelated` where compatibility requires it; display **Cross-task**.
- Shuffled: each target's existing `*_contrastive_qwen3_shuffled/roi_prior.csv`.
- Random: `outputs/priors/random_prior/aal116/roi_prior.csv`.

Verify ROI names/order against the feature atlas. Preserve the existing score vectors exactly; do not regenerate priors, re-prompt a model, choose a favorable shuffle, or add random draws after inspecting performance. Record that these controls use the existing fixed realizations. Keep the recorded model tag `qwen3.8:27b` and display name **Qwen3.8-27B**. The prior is external to the HCP feature/label fitting pipeline.

## 4. Exact FC penalty and equivalent scale

Let E = 116 × 115 / 2 = 6,670 and **c = 2E = 13,340**. Use the existing generalized solver's convention for all four component conditions. Do not change c to E for FC-only models or add a condition-dependent factor of n.

For standardized training features X and a centered/scaled training target z, the FC objective is:

`min_beta ||z - X beta||_2^2 + c * lambda_F * beta^T D beta + c * lambda_L * beta^T L_p beta`.

The intercept and target scaling are fitted on the applicable training subset only. Predictions must be returned in original target units before fusion and evaluation. If an algebraically equivalent formulation is used, demonstrate the equality and record its parameter mapping.

Preserve these fixed mechanisms:

- ROI prior p has 116 scores.
- Product lifting: `q_ij = p_i * p_j`, i < j, in the FC feature ordering. q has 6,670 scores.
- With anisotropy enabled, `gamma = 0.5`, `epsilon = 0.001`, and `D_e = (epsilon + abs(q_e))^(-gamma) / mean_f[(epsilon + abs(q_f))^(-gamma)]`.
- With anisotropy disabled, use **D = I exactly**. Gamma 0 is an implementation device for this condition, not another searched exponent.
- L_p is the existing **binary, symmetrically normalized line-graph Laplacian selected by the top 10 ROI scores p**. Two FC-edge vertices are connected when their FC edges share a selected ROI. Retain the current active-block construction and inactive zero block.
- L_p is not a continuously q-weighted graph. Use `L_p` in new documentation, tables, and figure metadata; internal compatibility names need not be globally renamed.
- Pass `top_k=10` explicitly; the cache builder's default 30 is not this experiment.
- Resolve and record the selected ROI set, active FC-edge indices, and tie behavior. Preserve the established top-10 set for each frozen prior. Do not inadvertently replace `argpartition` tie behavior with a different ranking and silently change the graph.
- No prior regularization enters the SC branch.

When the network penalty is disabled, set `lambda_L = 0.0` exactly. Removing a graph term must not remove the FC features, alter D, or modify c. When anisotropy is disabled in the network-only condition, recompute the appropriate cache/eigendecomposition for D = I while retaining the SAME L_p. Do not reuse a D-dependent whitened eigensystem from gamma 0.5 with gamma 0.

The same-solver isotropic case must agree with ordinary Ridge using **`alpha = c * lambda_F`**, after matching feature scaling, target centering/scaling, and intercept handling. Using `alpha = lambda_F` is the original mismatch. This equality requires a numerical test before production.

## 5. Mandatory retuned component matrix

Run every row for both targets and all 50 outer splits. Use descriptive condition names in exports in addition to IDs.

| ID | Name | D | Network term | SC branch | Fusion |
|---|---|---|---|---|---|
| R0 | Same-solver no prior | I | lambda_L = 0 | Ordinary SC Ridge | Corrected two-branch OOF fusion |
| R1 | Anisotropy only | D(q; 0.5) | lambda_L = 0 | Same SC Ridge | Same corrected fusion |
| R2 | Network only | I | positive lambda_L L_p | Same SC Ridge | Same corrected fusion |
| R3 | Full PALF | D(q; 0.5) | positive lambda_L L_p | Same SC Ridge | Same corrected fusion |

The primary baseline is **R0**, not the legacy unscaled-alpha FC Ridge reference. R0 must use the same FC loss, solver convention, scale, preprocessing, selection criterion, outer partitions, and fusion search as R1–R3. A verified algebraic shortcut for R0 is permitted only if equality to the generalized solver is tested.

Use these predeclared grids:

- `lambda_F`: `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]` in every R0–R3 condition.
- `lambda_L`: `[0.0]` for R0/R1; `[0.03, 0.1, 0.5, 1.0, 2.0, 5.0]` for R2/R3.
- SC sklearn Ridge `alpha`: `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]` in every condition.
- Gamma and lift are fixed by the condition; no gamma or lifting search.
- Fusion `w_FC`: `0.00, 0.05, ..., 1.00`; `w_S = 1 - w_FC`. In R3, use the paper notation `w_FP` for `w_FC`.

Retune each condition's remaining penalties inside the protocol below. Do not transfer R3's selected parameters or weights to R0–R2 and call it a retuned ablation. R2/R3 have a positive-only graph-strength grid by design; the actual no-network conditions are R0/R1.

Select branch parameters by the same deterministic rule in all conditions: maximize the mean selection-fold Pearson r, then minimize mean selection-fold RMSE, then MAE, then prefer larger lambda_F (or alpha), then larger lambda_L if still tied. Evaluate ties with a documented fixed numerical tolerance, not an outcome-dependent tolerance. Record the complete ordered grid and selection scores. These are explicit new protocol choices, not claims about the historical runner.

Select fusion weights by pooled outer-training OOF Pearson r, then RMSE, then MAE, then larger SC weight for an exact remaining tie. Declare the order and tolerance before production. Do not use outer-test scores to select branch parameters, weights, endpoints, grids, stopping conditions, or plot settings.

Report grid-boundary selections. Do not automatically expand grids after observing the outer-test results. Unequal numbers of candidates arising from a component's additional parameter must be disclosed; the controlled claim is equivalent solver scale and selection procedure, not identical search dimensionality.

Expected primary completeness: **2 targets × 10 seeds × 5 outer folds × 4 conditions = 400 unique primary condition/split rows**, and 80 primary seed/condition summary rows. Shared SC fits may be cached across conditions within the identical training context.

## 6. Fully cross-fitted training: required algorithm

Treat an entire branch fitting-and-selection procedure as the object being cross-fitted. Merely holding validation observations out of the final coefficient solve is insufficient.

For each outer split, let T be the outer-training set and E_test the outer-test set:

1. Create three fusion-training folds `(A_k, V_k)` partitioning T. Split generation uses deterministic seeds and subject indices, not target values. All modes use the same folds.
2. For each branch and each k, perform three-fold parameter-selection CV **inside A_k only**. In every selection split `(B, C)` inside A_k:
   - fit feature preprocessing on B;
   - fit target mean/std or any target transformation on B;
   - fit each candidate on B;
   - transform and predict C without refitting;
   - use only y_C to score candidates for that A_k selection.
3. Choose that branch's parameters from this A_k-only selection. Refit its scaler, target transform, and selected model on all A_k. Predict V_k in original target units. Store indices, parameters, transform statistics, and lineage for this exact fit.
4. After all three folds, each subject in T must have exactly one held-out prediction from each branch. Its own X and y must not have contributed to preprocessing, candidate selection, or coefficient fitting for that prediction. Every observation may contribute to other folds where it belongs to the training subset.
5. Train the convex fusion weights using only these branch OOF predictions and y_T. The OOF branch predictions are meta-training inputs; the fitted fusion's score on these same inputs is a selection diagnostic, not an unbiased outer evaluation.
6. **Independently reselect final parameters for each branch on all T using three-fold CV with fold-local preprocessing and target transforms.** Fit the selected branch and transforms on all T. Do not reuse the first OOF fold's parameters. Do not choose final parameters by averaging/voting the OOF-fold selections unless separately specified in a future protocol.
7. Predict E_test using the final fitted branches and the already fixed fusion weights. The fitting API must not receive y_E_test. Only the separate scoring stage may read it. Freeze the fitted state and selection records before scoring.

Perform steps 2–6 for SC as well as FC. An SC model fitted/selected on the identical training context can be shared across R0–R3, but a whole-T SC scaler or alpha cannot be reused for an A_k fit. Apply the same principle to any auxiliary ordinary FC reference.

It is valid to reuse cached inner fits for final selection only when the actual training/validation indices, transforms, target, candidates, and code/data hashes are identical. Do not assume that two operations both named "inner CV" have the same training scope.

All other data-dependent operations—imputation, feature filtering, confound regression, PCA, learned graph construction, or target transformations if present—must follow these boundaries. Do not introduce new such operations to the frozen feature representation in this task. External p/q/L_p computation may be cached globally because it uses only fixed atlas/prior information.

Use a stable deterministic seed derivation from explicit integer components or a stable cryptographic hash. Do not use Python's process-randomized `hash()` for folds. Persist the realized indices so later users need not guess seed derivations.

## 7. Additional outputs needed to refresh the existing paper consistently

Produce these alongside the four primary rows; keep their definitions distinct.

### 7.1 Branch and fusion diagnostics from the corrected fits

Using the already fitted models and held-out predictions, export:

- each condition's FC branch alone;
- SC Ridge alone;
- each condition's fixed 0.5 FC + 0.5 SC prediction;
- each condition's learned two-branch prediction.

No new outer-test-driven selection is allowed. Name fixed averaging accurately; it is not early feature fusion. Do not require fusion to improve every metric for every task.

Also rerun the legacy-grid ordinary FC Ridge + SC Ridge reference under the corrected nesting, with ordinary FC alpha `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`. Label it **Legacy-grid Ridge reference, retrained with corrected CV**. This is a secondary scale-comparison reference, not R0. It is relatively inexpensive and distinguishes effects of the prior from the old effective-alpha grid. Save its ordinary FC-only and equal-weight predictions too.

For continuity with the old augmentation diagnostic, use the newly cross-fitted ordinary FC, R3 FP, and SC predictions to select a three-branch nonnegative simplex on a 0.05 grid within T. Apply the weights to their new outer-test branch predictions. Use the same Pearson/RMSE/MAE rule, then prefer larger SC and smaller FP weights on a complete tie. Label this **Three-branch augmentation diagnostic**. It is secondary and uses no old fitted models.

### 7.2 Fixed prior-identity swaps

For each R3 outer split, refit the FC branch on T with each frozen cross-task, shuffled, and random prior. Rebuild both D and L_p for that control prior, retain gamma 0.5 and product lifting, and use R3's **final selected** lambda_F/lambda_L. Keep R3's final SC model and R3's fusion weights fixed. Evaluate each control on E_test.

This is a **matched-setting fixed prior-swap diagnostic**, not an independently retuned prior-control pipeline. State that distinction in tables, metadata, and the report. Do not generate fake strictly OOF control predictions by applying whole-T matched selections inside its OOF folds. No such OOF control predictions are needed for this fixed-swap diagnostic.

Save all three control coefficient vectors per outer split and their predictions. Expect 300 control FC refits over the 100 task/outer-split combinations. These are new fits; do not combine corrected R3 rows with historical control predictions. This design tests the effect of changing the prior under settings selected for the matched prior; it does not establish that a control is inferior after its own optimal retuning.

### 7.3 Scope exclusions

Do not add an SC-prior placement experiment, new LLM prior, atlas change, or family split to this run. The R0–R3 comparison isolates the specified FC mechanisms; it does not prove that FC-only prior placement is globally optimal. Historical AAAI E0–E10 and earlier PALF development versions remain historical, with their distinct targets/models/protocols. Do not place them in the primary paired component comparison.

## 8. Save exact fitted coefficients and lineage

Save each final FC coefficient vector during the new fit, rather than depending on a later reconstruction that may use different preprocessing. Required modes are R0–R3, the ordinary FC secondary reference, and all three fixed-swap controls.

Persist enough information to reproduce predictions: coefficient convention, feature ordering, feature means/scales, target mean/std, intercept, selected parameters, fit-subject indices, prior/hash, active graph indices, code/config hashes, and the final weights. Use NPZ/JSON or equivalently explicit schemas; never label a serialized object's contents only by a vague ID such as A4.

For comparison and biomarker analysis, use coefficients for **standardized FC features, expressed in original target units**. Also retain the solver's standardized-target coefficients or a precise conversion. If raw-feature coefficients are exported, label them separately and validate the intercept conversion. Verify both train and held-out predictions against the fitted predictor within a predeclared numerical tolerance. Do not demand exact bitwise zero from different algebraic evaluation orders.

Candidate FC biomarkers use the unfused FC coefficient vector beta_FP from R3. Do not average FC and SC coefficients, multiply beta_FP by a fusion weight without changing its stated definition, or derive FC coefficients from the fused prediction.

For every OOF fit, record the actual preprocessing-fit indices, parameter-selection universe, coefficient-fit indices, prediction indices, and stage. A boolean named `leakage_safe` is not evidence; the lineage must be generated from the executed fitting calls and validated against the split manifest.

## 9. Correctness gates before production

Add meaningful tests for the repaired statistical boundaries and solver semantics. A test that only checks source text, a function name, a flag, or equality to the same implementation is insufficient.

Required tests:

1. **Generalized-solver equivalence:** on a small synthetic connectome problem, R0 predictions and recovered coefficients match Ridge with alpha = 2E × lambda_F across multiple positive lambdas. Compare the generalized solver against an independently formed primal system for all four modes. Include a real-data R0 fold spot check at the actual E = 6,670. Suggested tolerances: float64 `rtol=1e-7`, `atol=1e-9` on scaled comparisons; diagnose conditioning instead of relaxing tolerances merely to pass.
2. **Disabled components:** R0 is invariant to all supplied prior values; R1 has zero network contribution; R2 has D exactly I but a nonzero L_p term; R3 contains both. Compare zero-network solves against the diagonal-penalty closed form. Confirm c is identical in every mode.
3. **Graph/cache identity:** product q ordering, gamma/epsilon/D normalization, top-10 ROI selection, active edge indices, symmetry/positive-semidefinite tolerance, and condition-appropriate whitened eigensystems are correct. R2/R3 use the same unwhitened L_p for the matched prior.
4. **OOF label perturbation:** with fixed splits, alter only y on V_k. The branch hyperparameters, scaler/target-transform state, and predictions for V_k must remain unchanged. Other folds' fits and the subsequently trained fusion weights are allowed to change. Test each branch and every component mode on a small fixture. This test must exercise the actual selection pipeline.
5. **OOF feature perturbation:** alter only X on V_k. State fitted on A_k, including its parameter selection, must not change. V_k predictions may change because its features changed. At a deeper selection-validation fold C, verify that changing C leaves B-fitted preprocessing and each fixed candidate's fitted state unchanged. C legitimately affects candidate validation scores and therefore may change the winning candidate; do not impose an incorrect selection-invariance test at that deeper level.
6. **Outer-test isolation:** perturb y_E_test and verify fitted state, selections, weights, and predictions are unchanged. Test-feature perturbation may change predictions but not fitted state. Assert that fitting cannot inspect outer-test labels.
7. **Final reselection:** a controlled fixture where different selection contexts prefer different candidates proves the final fit uses selection on all T, not the first OOF fold. Verify actual training indices and records for FC and SC.
8. **Complete nesting:** validate disjointness, subset membership, no duplicate held-out rows, exactly one branch OOF prediction per T subject, and identical split manifests across modes. Cache reuse with a changed training-index set must be rejected.
9. **Prediction reconstruction:** standardized-feature coefficients reproduce every saved final branch prediction, including controls and R0. Convex fusion reconstructs every saved fused prediction and obeys the selected grid.
10. **Statistics:** Holm adjustment uses a forward cumulative maximum in sorted-p order. Examples: four values 0.001953125 become four values 0.0078125; `[0.625, 0.625, 0.048828125]` becomes `[1, 1, 0.146484375]` in original order. Test rank metrics using actual feature-value ranks, tied values, and known identical/reversed rankings; do not correlate `argsort` index arrays.
11. **Resume integrity:** an interrupted-and-resumed small run matches an uninterrupted run in model selections, predictions, counts, and numerical summaries. A changed input/config/code/split fingerprint cannot append to the same production run.
12. **Protection and failures:** verify frozen artifacts are unchanged. Fail clearly on missing rows, invalid dimensions, failed fits, or undefined metrics; never catch a fit error and substitute zero predictions or p = 1 as though successful.

Document how constant predictions or tied/zero differences are handled mathematically. An undefined correlation must be recorded and treated consistently in candidate selection, not silently promoted to a winning score. If every candidate is invalid, stop that run with a diagnostic instead of dropping the split. Numerical failure is not a performance result.

Pass the new targeted tests and relevant existing solver/late-fusion regression tests. Do not rewrite an old failure's expected number to hide a change in semantics. Distinguish tests of the historical path from the new protocol when both remain available.

Run a real-data smoke evaluation containing both targets, all four modes, and at least one outer split per task through the FULL three-fold/deeper-three-fold nesting. A reduced candidate grid is acceptable only in the separately labeled smoke configuration. Exercise controls, export, resume, and reporting too. Do not reuse reduced-grid smoke outputs in production. Continue automatically to the full experiment after correctness passes.

## 10. Practical implementation and complete execution

Prefer a dedicated module, runner, configuration, and report script, for example:

- `src/metascfc/experiments/palf_crossfit_ablation.py`
- `scripts/116_run_palf_crossfit_ablation.py`
- `configs/iclr/palf_crossfit_ablation.yaml`
- `scripts_paper/summarize_palf_crossfit_ablation.py`
- `tests/test_palf_crossfit_ablation.py`
- `RUN_PALF_CROSSFIT_ABLATION.md`

Check for filename conflicts and adapt names if needed. The runner must have a real documented CLI with config, output directory/run ID, preflight/smoke/production modes, resume, and status support. Paths must resolve from the repository/configuration, not a developer's absolute home directory. Record exact executable commands that you actually ran; do not document nonexistent flags.

Persist settings in YAML and the run manifest rather than requiring undocumented session environment variables. Record Python, NumPy, SciPy, sklearn, BLAS, operating environment, CPU/GPU use, thread/worker settings, and wall-clock duration. This pipeline need not use a GPU to be correct.

Before production, write an immutable `protocol.json` and hash containing the cohort/split fingerprints, model definitions, grids, selection rules, analysis families, metrics, and planned outputs. This is a prospective specification for the corrected rerun, not a claim of external preregistration or an independent dataset. Record any later numerical bug repair/version change openly; never silently pool outputs from different implementations.

Benchmark a representative fold and estimate runtime/RAM. Use bounded workers and BLAS thread limits to avoid oversubscription. Cache expensive prior-only graph/eigen quantities per actual prior and D. Cache training-derived matrices only within the same exact fit context, with keys including training indices, transforms, task, mode, parameters as applicable, and hashes. Never optimize by pre-fitting a scaler on all T or the cohort. Avoid materializing a dense 6,670 × 6,670 system for every candidate when the existing dual/active-block solver suffices.

Write atomic checkpoints per completed task/outer split. Resume must validate the full fingerprint and skip only verified complete checkpoints; partial files must not count as completed rows. Keep progress counts, recent timing, logs, and a status command. If the environment supports durable sessions, use one for the long run and monitor it. Keep working through analysis after fitting completes. Merely starting a background job is not completion; if tool lifetime genuinely prevents monitoring to the end, report RUNNING with the job identity, log, status, and exact resume command.

Do not shorten the production run to three seeds, one task, fewer folds, or a reduced grid for convenience. Do not stop early because an effect is significant, negative, or small. A real resource blocker must be reported honestly with preserved progress.

## 11. Prespecified analysis and interpretation

### 11.1 Prediction aggregation

Compute Pearson r, RMSE, and MAE on each outer-test fold in original target units. For the main descriptive mean, average the five fold metrics within each seed, then average the ten seed summaries. Report the sample SD across seed summaries with ddof = 1. If pooled out-of-fold-per-seed correlation is also useful, export it under a separate name; do not replace the established mean-fold-r estimand without saying so.

All contrasts are paired using the same task, seed, and outer-fold IDs. Report mean/median seed difference, positive-seed count, and the metric's direction. Positive delta r favors the first model; negative delta RMSE/MAE favors it. Repeated partitions of the same cohort are not ten independent cohorts; seed SD and seed bootstrap intervals describe partition variation.

Primary inference: R3 minus R0 Pearson r for WM and Fluid, with Holm correction over these **two** tests.

Component inference: R1−R0, R2−R0, R3−R1, and R3−R2 for each task, giving **eight** Pearson-r tests in one separate exploratory Holm family. This directly tests incremental value without inferring it from selected nonzero penalties. Report all eight regardless of direction. An optional factorial difference-of-differences is descriptive unless separately specified before the run.

Fixed-swap predictive specificity: matched R3 minus each of the three swaps, a separate **three-test Holm family within each task**. Do not confuse this with alignment specificity or independently tuned controls. Branch, fixed-average, legacy-grid, and three-branch diagnostic comparisons are descriptive; do not add unplanned significance stars to those tables.

Use two-sided Wilcoxon signed-rank tests on the ten paired seed differences, recording the numerical library version and exact method. With no zeros/tied absolute differences, use the exact signed-rank distribution. If ties or zero differences occur, use an explicitly implemented/validated exact sign-flip distribution of the rank statistic (at most 2^10 patterns), removing exact zero differences and using average ranks for ties. Document the choice; do not call an asymptotic fallback exact. If every difference is zero, define p = 1. These tests remain sensitive to dependence between repeated partitions and are not participant-level evidence.

For descriptive mean-difference intervals, use 10,000 paired-seed bootstrap resamples with a fixed documented RNG seed, e.g. 20260906, and percentile 2.5/97.5 bounds. Export unrounded values. Report paired dz where defined; handle zero variance explicitly instead of silently replacing an undefined effect size with zero.

For the primary and component contrasts, also give a corrected repeated-CV sensitivity calculation from the 50 paired fold differences: `SE = sqrt((1/m + mean(n_test/n_train)) * s_d^2)`, m = 50 and sample variance s_d², with the actual fold-size ratios and Student-t df = m−1 documented. Report the raw sensitivity p-values and, if adjusted, use the same declared families separately. This is an approximate dependence sensitivity check, not proof that HCP family/cohort dependence is solved.

Implement correct Holm: stable sort raw p-values, multiply by the remaining family sizes, take the forward cumulative maximum, cap at 1, then restore original order. Always export raw p, adjusted p, contrast direction, family ID/size, and test method.

### 11.2 Alignment and genuine resampling stability

For all four modes, the secondary ordinary FC model, and controls, use the coefficient convention in Section 8 and the same FC feature ordering.

- ROI saliency for an outer fit is the mean absolute coefficient over edges incident to each ROI.
- Alignment is Spearman correlation of this ROI saliency vector with the target's **matched** p, for EVERY condition. Do not switch the reference to each control's own prior when comparing semantic alignment.
- Compute alignment per fit, average within seed, then summarize across seeds. Preserve fit-level and seed-level records. For R3 versus R0/cross-task/shuffled/random alignment, use a separate four-test Holm family within each task if reporting inferential p-values.
- Edge-rank stability is Spearman correlation between two fits' absolute coefficient vectors for the SAME edges.
- ROI-rank stability is Spearman correlation between two fits' ROI saliency vectors for the SAME ROIs.
- Within each seed, compare all ten pairs of its five outer fits. Average those ten pair values, then summarize the ten seed means. Do not treat the 100 pair observations as independent samples.
- Report edge top-k Jaccard overlap across fits for **k = 10, 20, 50, 100**, all four values. Also report top-10 ROI Jaccard across fits. Use a fixed documented tie rule for set selection.
- Keep top-10 ROI overlap WITH THE MATCHED PRIOR as a separate alignment/overlap statistic, never as resampling stability or top-10 edge overlap.
- Report actual resampling metrics for R0 as well as R3, so the new report can assess a stability difference instead of merely showing an absolute R3 value.

Alignment to a prior used in regularization is an expected consequence of that inductive bias, not independent biological validation. Stability also reflects overlapping training samples. Coefficient maps are candidate FC patterns, not causal or externally validated biomarkers. Keep these specific interpretive limits in the report without manufacturing a positive narrative.

## 12. Required run artifacts and schemas

Write a complete, machine-readable new-run directory containing at least:

| Artifact | Required contents |
|---|---|
| `protocol.json` | Frozen model/selection/analysis specification and hash |
| `run_manifest.json` | Starting HEAD, implementation diff/hash, configuration, inputs, priors, environment, timestamps, run status |
| `split_manifest.json` or equivalent | Outer/fusion/selection global indices, IDs, seeds, and subset lineage |
| `fit_lineage.jsonl` | Executed fit/selection/preprocessing scopes and associated artifacts |
| `outer_metrics.csv` | Task, seed, fold, condition, n_train/n_test, r, RMSE, MAE, protocol ID |
| `seed_metrics.csv` | Task, seed, condition, five-fold means, and separately named pooled metrics if exported |
| `component_summary.csv` | R0–R3 means/SDs and penalty definitions |
| `paired_comparisons.csv` | All declared contrasts, raw effects, intervals, tests, family IDs and adjusted p-values |
| `branch_fusion_summary.csv` | FC-only, SC-only, fixed-average, learned fusion, legacy-grid, three-branch diagnostics |
| `prior_control_metrics.csv` | All new fixed-swap outer results, with transfer-versus-retuning flags |
| `selected_hyperparameters.csv` | OOF and final selections explicitly distinguished; full selected values and training-context IDs |
| `selection_scores/` | Candidate-level inner-CV scores and final deterministic selection keys |
| `fusion_weights.csv` | Each outer split's actual weights, condition, branch names, and weight-selection scope |
| `oof_predictions/` | Local complete branch OOF predictions, subject-row mapping, and fitting lineage |
| `outer_predictions/` | Local complete new branch/fusion/control predictions and scoring targets, with row mapping |
| `coefficients/` | Every required final FC coefficient vector, transforms, intercepts, and graph/prior metadata |
| `biomarker_fit_metrics.csv` / `biomarker_seed_metrics.csv` | Alignment and separately defined prior-overlap values |
| `stability_pair_metrics.csv` / `stability_seed_metrics.csv` | True within-seed pairwise-fit rank/Jaccard metrics with k and pair IDs |
| `validation_report.json` | Test results, lineage checks, solver equivalence/reconstruction errors, completeness, hash checks |
| `diagram_readiness.json` | Evidence-based figure-label readiness defined below |
| `RUN_REPORT.md` | Methods, numerical results, limitations, timing, exact commands, comparison definitions, and status |
| `TABLE_AND_FIGURE_MANIFEST.md` | New artifact → source data → script → statistical definition mapping |

Use unique keys and verify all expected rows. A resumed run must neither duplicate nor average duplicate splits. Every table and figure must identify this new run and be reproducible from its saved records without retraining. Keep full-precision source data; round only display tables.

Create compact table fragments for later manuscript integration, without changing the manuscript itself:

- `tables/table_primary_prediction.tex`
- `tables/table_component_ablation.tex`
- `tables/table_branch_and_fusion.tex`
- `tables/table_prior_controls.tex`
- `tables/table_resampling_stability.tex`

The first two tables are candidates for the main paper; fuller diagnostics belong in the supplement. The author will later maintain the nine-page main-text limit. Do not alter LaTeX style, margins, or manuscript spacing here.

## 13. Plots from the NEW run only

Write plots beneath the new run directory, not over `figures_iclr/`. Use Matplotlib or the existing scientific plotting stack, white backgrounds, vector PDF plus PNG previews, a consistent restrained style, and labels legible at the actual approximately 5.5-inch ICLR single-column text width.

Required outputs:

- Four-condition ablation plot/table, showing R0–R3 for each task with the statistic and variability clearly labeled.
- A four-panel main-results candidate: WM and Fluid seed-level R3−R0 delta r; WM and Fluid ROI-prior alignment for R0, R3, cross-task, shuffled, random. Delta points must not be connected as a temporal trajectory. Alignment error bars should use seed SD, explicitly labeled, rather than imply independent-subject uncertainty.
- R3 fusion-weight distributions from the 50 actual split weights per target; preserve the 0.05 selection grid.
- A stability comparison showing R0–R3, with all predefined k values available in supplementary outputs.
- R3 top-20 FC-edge maps and source TSVs for both tasks, selected by mean absolute coefficient across 50 new final fits. Use signed mean coefficients for edge color and disclose task-specific scales. Never reuse old maps as if they came from the new fits.

Use display label `Cross-task` for the internal `unrelated` key. Keep coefficient units, aggregation rules, and source run IDs in the manifest/caption suggestions. Do not use significance stars or favorable colors to hide weak results. No expected old means or old fusion weights are acceptance targets for these new figures.

Do not generate or alter the method overview. The author will generate it with Gemini NanoBanana. Your responsibility is to provide correct diagram facts and readiness evidence.

## 14. Figure readiness and final review bundle

Populate `diagram_readiness.json` from executed validation results, not constants. Include:

- `production_complete`
- `all_400_primary_rows_present`
- `strict_oof_preprocessing_and_selection_verified`
- `final_branch_reselection_verified`
- `outer_test_exclusion_verified`
- `same_solver_scale_verified`
- `all_four_component_conditions_evaluated`
- `coefficients_and_stability_verified`
- `frozen_artifacts_unchanged`
- `ready_for_corrected_protocol_figure`
- `evidence_paths` and concise reasons for any false value.

The final readiness flag is true only when the required experiment and relevant gates actually pass. It is independent of performance direction or significance. Supply these immutable figure facts: AAL116; E = 6,670; c = 13,340; model display Qwen3.8-27B; ROI prior p; product edge prior q; fixed gamma 0.5; top-10-ROI binary normalized L_p; FC-only prior; ordinary SC Ridge; nested branch cross-fitting; convex two-branch fusion; separate final reselection/refit; R0 same-solver D = I/lambda_L = 0 control; biomarkers from unfused beta_FP.

Create `PALF_crossfit_ablation_review_<run_id>.zip` with the run report, manifests, validation/readiness reports, aggregate and seed-level result tables, hyperparameter/weight summaries, table fragments, plot PDFs/PNGs/TSVs, exact commands, tests, and implementation patch. Keep the full fitting checkpoints, subject-level labels/predictions, and large coefficient archives locally in the complete run directory; identify their paths and hashes in the bundle. Do not publicly publish HCP data.

The report must explicitly answer:

1. Were the two scientific defects repaired, and what tests prove it?
2. How does R3 compare with the equivalent-scale R0 on each target?
3. What do R1/R2 reveal about anisotropy and the network penalty, after retuning?
4. Does learned fusion improve each task/metric over its branches and equal averaging?
5. How do fixed prior swaps affect prediction and matched-prior alignment, separately?
6. Does R3 improve resampling stability over R0 under the defined measures?
7. Which earlier paper claims are supported, weakened, or still untested?
8. Which corrected diagram labels are now justified?

Record possible disadvantages honestly: reused cohort and development history, subject-wise/family dependence, grid-boundary selection, control-transfer design, fixed random/shuffled realizations, prior-alignment circularity, and lack of external validation. Do not claim that correcting cross-fitting retroactively validates historical results.

## 15. Final completion report

At the end provide the exact run path, review ZIP path, code commit/diff hash, executed commands, duration, passed/failed tests, maximum reconstruction/equivalence error, and these counts:

- Both targets complete: yes/no.
- Ten seeds and five outer folds complete per target: yes/no.
- Primary R0–R3 rows: actual/400.
- Primary seed summaries: actual/80.
- Secondary reference and fixed-swap rows: actual/expected.
- All required fit lineage and final parameter selections saved: yes/no.
- New coefficient and stability exports complete: yes/no.
- Historical result/prior hashes unchanged: yes/no.
- Corrected protocol figure ready: yes/no, with evidence path.

Print `STATUS: COMPLETE_FOR_AUTHOR_REVIEW` only after fitting, analysis, and artifact checks finish successfully. Use `RUNNING`, `BLOCKED`, or `FAILED_VALIDATION` otherwise and explain the precise remaining action. Do not use a performance or significance threshold in this status.

Begin by inspecting the current checkout, then implement and execute this workflow to completion.
