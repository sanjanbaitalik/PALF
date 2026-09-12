# OLD PHASE-3A FORENSIC AUDIT

Status: `INVALID_FOR_PHASE3A_DECISION`

The old Phase-3A run (`scripts_paper/phase3a_pilot.py`, output
`outputs/iclr/palf_phase3a_pg_mt_bcr/`) did not implement the predeclared
protocol. Findings F1-F10 below cite the exact locations in the old code.

---

## F1 — Baseline was not corrected R0

**CONFIRMED.** `scripts_paper/phase3a_pilot.py` (old) built the baseline as

```python
fc_upper = symmetric_to_upper(FC)
sc_upper = symmetric_to_upper(SC)
X_upper = np.hstack([fc_upper, sc_upper])
oof = crossfit_ridge_oof(X_upper, y, seed, n_folds=5)
```

i.e. an ordinary concatenated FC+SC Ridge. That is not the paper baseline. The
required R0 is generalized same-solver no-prior FP + SC Ridge + cross-fitted
convex FP+SC fusion from `src/metascfc/experiments/palf_crossfit_ablation.py`.

The old audit reported WM r=0.22257, FI r=0.35239, WM RMSE=11.7580,
FI RMSE=4.6885 while labelling the audit `PASS`. These do not match the frozen
corrected values (WM r=0.2635147736, FI r=0.3709173350, WM RMSE=11.2929210027,
FI RMSE=4.5666891937).

---

## F2 — Audit tolerances were weakened

**CONFIRMED.** The old code declared

```python
TOL_PEARSON, TOL_RMSE = 0.05, 0.5
```

instead of the predeclared `5e-4` / `0.05`. The wide tolerances let the wrong
baseline pass.

---

## F3 — Only one inner split was used

**CONFIRMED.** The old inner candidate loop used

```python
it, iv = next(iter(ikf.split(Xfc_tr)))
```

i.e. a single inner fold, and scored candidates on it. The protocol requires
concatenated predictions across all three inner folds.

---

## F4 — Inner preprocessing leakage

**CONFIRMED.** The old code standardized FC/SC edges once on the entire outer
training set

```python
fc_tr_s, fc_te_s, fc_mu, fc_std = standardize_edges_train_test(fc_tr_upper, fc_te_upper)
```

and reused those matrices inside inner validation. Inner candidates must fit
FC/SC scalers on inner-train only and transform inner-validation with those
statistics; residual target mean/std must also be inner-train only.

---

## F5 — Wrong R0 residual generator

**CONFIRMED.** Old residual targets were

```python
r0_oof_WM = crossfit_ridge_oof(X_tr_upper, y_wm_tr, seed, n_folds=5)
rW_tr = yW[tr] - r0Wtr
```

derived from the approximate concatenated-Ridge OOF. Correct residuals must
come from the validated R0 procedure, and for a scope `S` every subject must
receive a baseline prediction from an R0 model that did not train on it
(`crossfit_r0_within`).

---

## F6 — Single-task WM result was overwritten

**CONFIRMED.** The old ST loop wrote

```python
results[model_id] = {"pred_WM": pred_WM, "pred_FI": pred_FI, ...}
```

once for WM and then overwrote the same entry when FI was processed. As a
result B0/B1 WM was often exactly A0. Correct storage must preserve both task
predictions.

---

## F7 — ST prior-strength grid was not searched

**CONFIRMED.** The old ST path used `lp = LAMBDA_PRIOR_GRID[1]` (a single fixed
value) and did not iterate over `LAMBDA_PRIOR_GRID`. B1 must search
`lambda_amp ∈ {0.1,1,10}`, `lambda_prior ∈ {0.01,0.1,1}`, and all five alphas.

---

## F8 — Multi-task shared amplitudes were shared across tasks

**CONFIRMED.** The old MT network used a single pair of shared amplitude
vectors

```python
self.a_fc_sh = nn.Parameter(torch.zeros(SHARED_RANK))
self.a_sc_sh = nn.Parameter(torch.zeros(SHARED_RANK))
```

for both WM and FI. The intended model has shared spatial factors but
task-specific shared amplitudes:
`amp_fc_shared_WM`, `amp_fc_shared_FI`, `amp_sc_shared_WM`,
`amp_sc_shared_FI`.

---

## F9 — Optimizer did not match the frozen procedure

**CONFIRMED.** The old runner replaced the frozen full-batch Adam with
`torch.optim.LBFGS` and different iteration rules:

```python
opt = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=20, ...)
```

The corrected run uses the frozen Adam configuration (`lr=0.01`,
`max_steps=1500`, `min_steps=200`, `patience=75`, `rel_tol=1e-7`,
`grad_clip=5.0`, `restarts=2`).

---

## F10 — Requested biomarker/optimizer outputs were omitted

**CONFIRMED.** The old ZIP contained no `optimizer_diagnostics.csv`,
`selection_details.csv`, coefficient exports, or biomarker stability outputs.
The corrected run produces all of these.
