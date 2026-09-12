# OpenChallenge — RUN REPORT

```text
OPENCHALLENGE_DECISION: NO_METHOD_JUSTIFIED
STATUS: OPENCHALLENGE_COMPLETE
```

Per prompt §9, the challenge was rejected **before coding a final method**: after
inspecting the repository evidence and running an independent, honest screening
programme, no genuinely new method has a scientifically plausible chance to
improve BOTH WM and FI prediction by the predeclared margin while passing the
biomarker specificity gates. Running any screened candidate under a lock would
be a cosmetic variant prohibited by §20. See `MODEL_PROPOSAL_REJECTED.md`.

---

## A. Holdout status

- count = 98; unique = 98
- SHA256 = `89c563602778a0a2618e4b94245c3cd1b8058f0ef4e6248b6a294bfd4176c425`
- development cohort = 412; intersection = 0
- **The 98-subject holdout was never loaded, inspected, or used in any way.**
  Only development files were accessed. `HOLDOUT_SEAL_REPORT.json` records this.

## B. Baseline audit (strict, seeds 0-9)

| Task | Pearson r | RMSE | expected r | expected RMSE | result |
|------|-----------|------|------------|---------------|--------|
| WM | 0.2635147736 | 11.2929210027 | 0.263515 | 11.292921 | PASS |
| FI | 0.3709173350 | 4.5666891937 | 0.370917 | 4.566689 | PASS |

Errors ≤ 3.4e-07 (Pearson) and ≤ 1.9e-07 (RMSE). R0 was used unmodified; no
search-space weakening.

## C. Screening programme (the scientific work)

Protocol: every candidate predictor was late-fused with R0 by a convex weight.
**All selection happened on a 3-fold inner OOF inside the outer-training
scope**; the outer-test fold was untouched. Screening seeds 11–20 are fresh
development seeds, not the locked seeds.

Candidate families screened (60 predictors, 9 screens):

1. Cross-view PLS/CCA latent scores; supervised PLS (K = 2, 5, 10, 20).
2. Reduced-rank regression (rank 2–16).
3. Structural-diffusion (heat-kernel) features of SC at τ = 0.5, 1, 2.
4. Population graph-Laplacian eigenbasis projections of FC and SC.
5. Per-subject rank normalization; per-subject z-scoring (subject-relative).
6. Cross-modal FC-from-SC residual features (with/without SC).
7. kNN regression in supervised latent space (k = 15, 20, 40).
8. RBF kernel ridge and SVR.
9. Gradient-boosted trees (L2 and Poisson losses); random forests were not run.
10. Random-subspace ridge (100 subspaces, 10% features); bagged ridge.
11. Elastic net; rank-target ridge; separable two-alpha FC/SC ridge.
12. Data-driven edge-cluster aggregates (K = 50, 200).
13. Shrinkage precision (partial correlation); SC effective resistance.
14. Node graph-topology descriptors (strength, clustering, centrality).
15. SC log/sqrt/binary transforms; signed/sqrt FC; per-edge FC×SC products.
16. Multi-alpha ridge averaging.

Full results: `screening/screening_evidence.json`, `screening/` logs.

### What was found

- **WM has local headroom; FI does not.** Strong-ridge FC features (+0.02..+0.05),
  PCR-100 (+0.02..+0.04), kNN (+0.01), per-subject z-scored FC (+0.014, 4/4
  seeds on 11–14), and count-GLM (+0.01) improved WM on individual screens.
  No candidate improved FI stably; the largest FI effects were ≤ +0.006 and
  sign-flipped on the next seed.
- **Apparent gains failed replication and control matching.** The one
  apparently robust FI signal (subject-relative normalization, +0.0108 over
  seeds 11–14, 4/4) failed on fresh seeds 16–20: FI proposed −0.0012 vs its
  architecture-matched control −0.0018; WM proposed +0.0066 vs control +0.0062.
  The control isolates the only novel ingredient, and it contributes nothing.
- **Joint selection collapses the gains.** Selecting representation, ridge
  strength, and blend weight jointly on inner OOF (screen #7) gives
  WM −0.0003 and FI +0.0043 — the apparent gains live in selection noise.

## D. Why this is the correct decision

The predeclared Gate P1 requires, on BOTH tasks: mean(Proposed − R0) ≥ +0.005,
≥3/4 seeds positive, and ≥ +0.010 on at least one task. Across the entire
screened space these thresholds are not met on FI under any candidate; the WM
effects that exist are matched by architecture-matched controls and/or are
"different lambda" regularization variants explicitly excluded by §7. The
biomarker gates require a locked model whose biomarker object can be validated;
without a prediction-passing model there is no justified object to lock.

This mirrors and extends the repository's own falsification of families A–G
(the lead finding — R0 is at/near the reliable ceiling for this 412-subject
cohort — is confirmed independently).

## E. Prediction results

Not applicable: no method was locked or run, so there are no Proposed/P0/P1
development results to report. R0 itself was audited and reproduced exactly.

## F. Biomarker results

Not applicable (no locked method). The rejection was made before any
biomarker claim was generated.

## G. Gate table

| Gate | Result |
|------|--------|
| P1 (beats R0, both tasks) | NOT RUN — NO LOCKED METHOD |
| P2 (semantic prior value) | NOT APPLICABLE (no semantic prior) |
| P3 (specificity) | NOT APPLICABLE |
| B1 (held-out faithfulness) | NOT RUN — NO LOCKED METHOD |
| B2 (architecture-matched biomarker advantage) | NOT RUN — NO LOCKED METHOD |
| B3 (no circularity) | PASS (by construction; no biomarker claim made) |
| validity / leakage | PASS (holdout untouched; audit strict) |

## H. Mechanistic interpretation

Two robust facts from the screens:

1. R0's FI branch (SC ridge plus fusion) is at the reliable ceiling of the
   tested representation space. FI prediction barely responds to any change of
   representation, kernel, or estimator at n=412, and its small residuals do
   not carry extractable signal (candidate-vs-residual correlations at or
   below zero).
2. WM has modest, real headroom over R0, but it is shared across many
   regularization variants of the same raw-edge features. It does not
   constitute a new source of generalizable signal and it does not transfer
   to FI, so it cannot satisfy the joint gate.

## I. Outputs

- `HOLDOUT_SEAL_REPORT.json`, `BASELINE_AUDIT.json`, `VALIDATION_REPORT.json`
- `MODEL_PROPOSAL_REJECTED.md` (pre-coding rejection, per §9)
- `RUN_REPORT.md`, `COMPLETE`
- `screening/screening_evidence.json` + screen logs
- `tests/` functional tests (seal, audit, honest-nesting invariants)
- `plots/` screening summary
- `predictions/`, `biomarkers/`, `controls/`, `coefficients_or_attributions/`
  are NOT APPLICABLE (no locked method); each contains a README note.
- `outputs/iclr/openchallenge.zip`
