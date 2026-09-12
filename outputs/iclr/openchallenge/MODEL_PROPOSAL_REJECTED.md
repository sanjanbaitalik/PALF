# OpenChallenge — Pre-coding decision: NO_METHOD_JUSTIFIED

Per prompt §9, a method may be rejected before coding when no genuinely new
method has a scientifically plausible chance. This document records the
evidence and the reasoning for the rejection. No `MODEL_PROPOSAL_LOCK.md` is
created because no method was locked.

## Diagnosis of the evidence

1. **R0 is a strong, well-calibrated baseline.** The strict audit reproduces
   WM r=0.2635147736, RMSE=11.2929210027 and FI r=0.3709173350,
   RMSE=4.5666891937 on seeds 0-9 (errors ~1e-7). R0 late-fuses a no-prior
   FC kernel-ridge branch with an SC ridge branch; its FI performance is
   especially hard to beat.
2. **Repository history already falsified seven mechanism families** (A-G in
   the prompt): global prior penalties, differential stacking, grouped ridge,
   multi-task residual NCR, prior-selected experts, SFC coupling, and PG-MT-BCR.
3. **Independent pre-lock screens (this run) tested ~60 genuinely distinct
   candidate predictors** with an honest nested protocol (blend weights and
   candidate hyperparameters selected only on inner OOF); see
   `screening/screening_evidence.json` for the full list.

## Why every candidate failed

Two reproducible regimes were observed:

* **WM has local headroom; FI does not.** Several distributed representations
  improved WM out-of-sample on individual seeds (strong-ridge FC features
  +0.02..+0.05, PCR +0.02..+0.04, kNN +0.01, per-subject z-scored FC +0.014
  over 4 seeds, count-GLM +0.01). But the predeclared Gate P1 requires
  improvement on **both** tasks. FI improved by at most +0.006 on a single
  seed and was negative on the next; no candidate showed a stable FI gain.
* **Apparent gains did not replicate on fresh seeds and were matched by
  architecture-matched controls.** The one signal that looked robust on
  screening seeds 11-14 (per-subject z-scored edge ridge: FI +0.0108, 4/4
  seeds) failed to replicate on fresh seeds 16-20: FI proposed -0.0012 vs
  control -0.0018, WM proposed +0.0066 vs control +0.0062. The paired control
  isolates the only novel ingredient (subject-relative normalization), and it
  contributes nothing measurable.
* **Joint hyperparameter selection is the binding constraint.** When the
  representation, ridge strength, and blend weight were selected jointly on
  the 3-fold inner OOF (screen #7, seeds 11-15), the deltas collapsed to
  WM -0.0003 and FI +0.0043. The apparent gains live in selection noise, not
  in the representation.

## Novelty rejection

The residual WM effects trace to a *different regularization strength* on the
same raw-edge representation (heavy ridge, alpha=100..1000). Section 7 of the
challenge explicitly rejects "different lambda" as sufficient novelty. The
subject-relative normalization is mathematically distinct, but it failed both
the replication test and the architecture-matched control test; running it
under a lock would be a cosmetic variant prohibited by §20.

## Conclusion

No genuinely new method in the explored space has a scientifically plausible
chance to clear Gate P1 (+0.005 on WM and FI, ≥3/4 seeds positive, ≥+0.010 on
one task) *and* the biomarker specificity gates, without weakening R0 or
exploiting the holdout. The scientifically correct action is to stop before
coding.

```text
OPENCHALLENGE_DECISION: NO_METHOD_JUSTIFIED
STATUS: OPENCHALLENGE_COMPLETE
```
