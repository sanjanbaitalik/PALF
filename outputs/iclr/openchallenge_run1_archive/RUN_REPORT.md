# OpenChallenge CKE — RUN_REPORT

Decision: `OPENCHALLENGE_DECISION: NO_GO`
Status: `STATUS: OPENCHALLENGE_COMPLETE`

Proposal lock: `MODEL_PROPOSAL_LOCK.md`
SHA256 = `daff28b377bb8141718f726d9fa13b685f2627bb572a3c8fedaf7d5d64b2f0dd`
(frozen before any outer-CV result of the locked method; only feasibility
screens and the strict baseline audit were run before the lock)

---

## A. Method chosen

**CKE — Calibrated Kernel Ensemble.** Hypothesis: the unexploited signal on the
412-subject cohort is a new *function class*, not a new edge representation. A
bandwidth-controlled RBF kernel ridge predictor over standardized FC+SC edges
captures smooth nonlinear (prototype/distance) structure that no linear model
on any edge transform can represent, and a per-task, inner-CV-calibrated
convex weight adds it on top of the unmodified corrected R0 baseline:

```
f_prop(x) = (1 - w) * f_R0(x) + w * f_KRR(x),  w ∈ {0, 0.1, ..., 1.0}
f_KRR(x)  = b + Σ_i α_i exp(-||x - x_i||² / (2σ²)),
σ = m · median pairwise distance (train scope), λ ∈ {0.3, 1, 3, 10},
m ∈ {0.5, 1, 2}.  w = 0 nests R0 exactly.
```

No semantic LLM prior is used (`PRIOR_CLAIM_NOT_APPLICABLE`). Evidence base
(pre-lock screens, all leakage-safe 5-fold OOF on development data): node
topology, graph-spectral FC-on-SC-eigenbasis, reduced-rank ridge, and
multi-view strength profiles all showed ZERO or negative complementarity with
R0 residuals and optimal ensemble weight ≈ 0. Only the nonlinear-function-class
screen showed positive complementarity (FI KRR own r 0.3668 > R0 0.3534,
residual corr +0.063, pooled ensemble +0.032), which motivated the lock.

## B. Holdout seal
- 98 / 98 unique; SHA256 `89c56360...c425`; dev∩holdout = empty
- Access log: 0 non-empty holdout overlaps; **holdout untouched**

## C. Baseline audit (strict, seeds 0-9)
WM r=0.2635147736 (err 2.3e-07), RMSE=11.2929210027 (err 2.7e-09) — PASS
FI r=0.3709173350 (err 3.4e-07), RMSE=4.5666891937 (err 1.9e-07) — PASS

## D. Implementation validity
- R0 fused linear map reconstruction: max err 1.7e-13 (20/20 outer fits)
- KRR dual solve: finite in all 1,560 recorded fits (solve_ok = True)
- Attribution: analytic kernel gradients == finite differences (≤1e-4 rel);
  linear-kernel control gradient == exact ridge primal map
- Ensemble/units: `f = (1−w)·R0 + w·KRR` in raw target units; w=0 ⇒ exactly R0
- OOF label alignment, train-scope-only scalers, deterministic partitions —
  all covered by 18/18 functional tests (`tests/test_openchallenge.py`)

## E. Development prediction results (20 outer fits)

| Model | WM r | WM RMSE | FI r | FI RMSE |
|-------|------|---------|------|---------|
| R0 | **0.2989** | **11.044** | **0.3782** | **4.557** |
| P1_rbf (proposed) | 0.2549 | 11.112 | 0.3754 | 4.563 |
| P0_linear (control) | 0.2602 | 11.356 | 0.3796 | 4.569 |

## F. Seed-level deltas
- Proposed − R0: WM mean −0.0440, median −0.0453, 0/4 positive;
  FI mean −0.0029, median −0.0037, 1/4 positive
- Proposed − control: WM mean −0.0053, 3/4 positive; FI mean −0.0042, 1/4

## G. Biomarker results
- Stability (proposed vs control): edge Spearman WM 0.563 vs 0.495,
  FI 0.537 vs 0.479; ROI top-10 Jaccard WM 0.385 vs 0.359;
  sign consistency 0.865/0.867 (proposed WM/FI)
- Faithfulness (top10 − random10 ΔRMSE): proposed WM +0.043, FI +0.054
  (means positive) but seed-level positive counts 0/4 and 0/4
- Task-specific contrast (own − cross): proposed WM +0.020, FI +0.067
  (both positive); control WM −0.092, FI +0.011

## H. Mechanistic interpretation
The pooled pre-lock ensemble gain was real in-sample-complementarity but did
not survive honest per-split inner-CV weight selection: the per-split optimal
w is unstable (0.0–1.0 across folds) and the KRR component is on average
worse than R0 out-of-sample (own OOF r below R0 on most splits), so the
calibrated fusion inherits selection noise. What IS real: the kernel
component's gradient attribution is more stable and more faithful than its
linear twin (B2 PASS on 3/4 metrics per task), and the WM/FI attribution
rankings are genuinely task-specific (positive contrast on both tasks). But
neither property rescues prediction: Gate P1 fails on both tasks, and B1's
seed-level consistency fails. Honest conclusion: at n=412 the calibrated
kernel ensemble does not beat corrected R0 on BOTH tasks; the earlier screen
gain was selection noise, and the biomarker advantage alone does not justify
a freeze.

## I. Gate table
| Gate | Result |
|------|--------|
| P1 (beats R0) | FAIL (WM −0.044 0/4; FI −0.003 1/4) |
| P2 (prior value) | NOT APPLICABLE |
| P3 (specificity) | NOT APPLICABLE |
| B1 (faithfulness seeds) | FAIL (means positive but 0/4 positive seeds) |
| B2 (matched vs control biomarkers) | PASS |
| B3 (no circularity) | PASS (by construction) |
| validity | PASS |

## J. Final decision

```text
OPENCHALLENGE_DECISION: NO_GO
STATUS: OPENCHALLENGE_COMPLETE
```

The 98-subject holdout remains sealed. Model development on the 412-subject
cohort stops here.
