# MODEL PROPOSAL LOCK — OpenCode Go Open Challenge

This document is the pre-registered proposal for the Open Research Challenge.
It was written AFTER inspecting the repository, the prior negative results, the
baseline errors, and AFTER running four pre-lock feasibility screens
(scripts_paper/oc_screen_multiview.py, oc_screen_nonlinear.py plus inline
screens of topology, spectral, and reduced-rank families). No outer-CV result
of the method below has been observed. After the SHA256 of this file is
recorded, the scientific method and hyperparameter grid are frozen.

---

## A. Diagnosis

1. R0 (generalized same-solver no-prior FP + SC Ridge + cross-fitted convex
   FP+SC fusion) is a strong, well-calibrated baseline: WM r=0.2635,
   FI r=0.3709 (strict audit, seeds 0-9).
2. Every failed family (A-G in the prompt) either (a) reweighted/selected the
   same raw edges with a semantic prior, or (b) trained a residual learner on
   the same raw-edge representation. All residual learners correlated with
   noise, and prior reweightings never beat uniform controls.
3. Screening evidence gathered for this challenge (development-cohort OOF,
   leakage-safe 5-fold, candidate models trained on training folds only):
   - node-topology descriptors (strength/clustering): own OOF r = 0.085/0.233,
     correlation with R0 OOF residuals = -0.105/-0.149, optimal ensemble
     weight = 0.00/0.00 (no complementarity);
   - graph-spectral FC energies on each subject's own SC Laplacian eigenbasis:
     own r = 0.062/0.036, residual corr = -0.039/-0.019, weight 0.10/0.00;
   - reduced-rank ridge (rank 2-16, joint WM/FI output): r = 0.240/0.353 and
     0.221/0.335 — below R0 on both tasks;
   - multi-view node-strength profiles: own r = -0.021/0.253, residual corr
     -0.282/-0.153, weight 0.00/0.05.
   All four candidate representations are re-descriptions of the raw-edge
   signal: they do not extract what R0 misses.
4. The nonlinear-function-class screen (oc_screen_nonlinear.py), evaluated
   against the REAL R0 pooled OOF (r=0.2583 WM, 0.3534 FI), is the only screen
   with positive complementarity:
   - RBF kernel ridge regression (RBF-KRR): own OOF r = 0.2212 (WM) and
     0.3668 (FI); correlation with R0 OOF residuals = -0.104 (WM) and
     +0.063 (FI); calibrated ensemble with R0 gives pooled r = 0.2677 (WM,
     +0.009) and 0.3849 (FI, +0.032).
   - Gradient-boosted trees: no complementarity (weight 0.00/0.30, gains
     0.000/+0.009), and unstable at n=412 with 13,340 features.
5. Conclusion of the diagnosis: the remaining unexploited signal is not a new
   edge representation but a new FUNCTION CLASS. Smooth nonlinear structure in
   the standardized connectome features — which no prior family tested —
   carries FI-relevant signal that is nearly orthogonal to R0's linear
   prediction (residual corr +0.063 is the only positive residual correlation
   observed in any screen).

## B. Proposed method

**Name: CKE — Calibrated Kernel Ensemble (two-predictor late fusion).**

Per task t ∈ {WM, FI}, per fitting scope, with all preprocessing fit on the
fitting scope's training subjects only:

1. **Base predictor (frozen, unmodified):** corrected R0 fused prediction
   f_R0 (validated FP+SC cross-fitted convex fusion from
   src/metascfc/experiments/palf_crossfit_ablation.py).
2. **New predictor:** radial-basis-function kernel ridge regression on the
   standardized 13,340-dim edge vector x = [FC_z; SC_z]:

   f_K(x) = b + Σ_{i=1}^{n_train} α_i k(x, x_i),
   k(x, x') = exp( -||x - x'||² / (2σ²) ),
   α = (K + λ I)^{-1} (y - ȳ_train),  b = ȳ_train,
   K = k(x_i, x_j) on training subjects (diagonal = 1),
   σ = m · median( ||x_i - x_j||₂ : i<j, i,j ∈ train )  (median heuristic).

   KRR is fit on the raw target y (NOT on R0 residuals; residual learners on
   the same representation are a rejected family).
3. **Calibrated late fusion:**

   f_prop(x) = (1 - w) · f_R0(x) + w · f_K(x),

   with the ensemble weight w selected, per task, on concatenated 3-fold
   inner OOF predictions (never on outer-test data). w = 0 is in the grid, so
   the proposed model nests R0 exactly.

4. **No semantic prior.** The LLM prior is not used anywhere in CKE.

## C. Novelty map

| Family (§4) | Mechanism | CKE difference |
|---|---|---|
| R0 | linear ridge per modality, convex fusion of two linear branches | CKE adds a nonlinear RKHS predictor and a third calibrated fusion weight; new function class, not new weights on the same representation |
| A. global prior penalties (PALF, adaptive) | reweighted raw-edge ridge | CKE uses no prior and no raw-edge reweighting |
| B. differential stacking | residual/difference stack of linear models | CKE fits KRR on the raw target, not residuals; function class differs (RBF vs linear) |
| C. joint prior-aware ridge | grouped penalties on raw edges | absent in CKE |
| D. multi-task residual NCR | residual learner on raw edges | rejected family; CKE has no residual stage |
| E. prior-selected raw-edge experts | feature selection by prior | absent in CKE |
| F. SFC coupling | SC×FC system-pair product features | absent in CKE |
| G. PG-MT-BCR | 3 bilinear factors per modality on residuals | CKE is full-dimensional RBF (all pairwise interactions, bandwidth-smoothed), on the raw target, with no factor optimization |

New mathematical/statistical object: a bandwidth-controlled RBF kernel over
standardized connectome features, i.e. an RKHS predictor (infinite-dimensional
smooth function class), late-fused with the linear R0 baseline by an
inner-CV-calibrated convex weight.

What CKE can represent that R0 cannot: nonlinear (smooth, radial) dependence
of the target on the joint FC+SC feature vector — e.g., distances to
prototypical connectivity patterns — which no linear model on any
edge-transform can represent. What it can represent that §4 families cannot:
distributed nonlinear interactions among arbitrary edge combinations (BCR's
bilinear factors cover only three rank-1 quadratic forms per modality; SFC
covers only system-pair products; all others are linear).

Why generalization rather than mere capacity: the KRR has exactly two
effective complexity knobs (σ via median heuristic, λ), both selected on
inner OOF; the median-heuristic bandwidth adapts to the feature scale; the
dual has n_train ≈ 330 unknowns with λ-controlled effective degrees of
freedom; and the nested R0 fallback (w=0) bounds the downside. This is the
standard small-sample regime in which kernel ridge with tuned bandwidth is
appropriate.

Why plausible at n=412: the screen above already shows positive out-of-fold
complementarity (+0.063 residual correlation for FI, +0.032 pooled ensemble
gain) from exactly this predictor, and the two hyperparameters are selected
by inner CV, not hand-tuned.

## D. Predictive mechanism

R0 is linear in standardized edges. CKE adds a smooth radial-basis predictor
over the same features. If subjects with similar multivariate connectivity
signatures share cognitive outcomes in a way that is not a linear function of
individual edges (prototype/distance effects), the KRR captures it, and the
calibrated fusion adds it on top of R0. The FI screen result (own KRR r
0.3668 > R0 0.3534, ensemble 0.3849) is direct OOF evidence for this
mechanism. For WM the KRR alone is weaker (0.2212) but the ensemble still
gained +0.009 pooled; the locked protocol will test whether this survives
honest inner-fold weight selection.

## E. Biomarker mechanism

The KRR attribution object is the analytic gradient of the kernel predictor
w.r.t. the standardized edge vector, aggregated over the training subjects:

g(x)_e = ∂f_K/∂x_e = -(1/σ²) [ x_e · Σ_i α_i k(x, x_i) - Σ_i α_i k(x, x_i) x_{i,e} ]

Importance map per outer fit: ā_e = mean_{s ∈ train} |g(x_s)_e| (deterministic
for fixed model + data). Signed map: sgn_e = sign(mean_s g(x_s)_e).
ROI score: S_i = Σ_{j≠i} ā_{edge(i,j)}.

Why this should be more stable/faithful than a raw linear map: the RBF
gradient localizes credit on edges that separate a subject from its kernel
neighbors — distributed, bandwidth-smoothed, and regularized by λ — rather
than on single noisy edge weights. The ensemble is a genuine
held-out-perturbation test (masking top ROIs changes both the KRR and R0
components through their actual prediction paths).

## F. LLM prior role

**Not used.** The semantic LLM prior is absent from CKE by design. Rationale:
seven prior mechanism families have failed to show matched-prior predictive
value (prompt §4, verified in outputs/iclr/*), and every matched-vs-control
specificity test in this repository has been negative or noise-level. Adding
a prior to CKE would add degrees of freedom with a track record of zero
positive specificity, against a method whose screened gain comes from a
nonlinear function class, not from semantic weighting. Therefore:
`PRIOR_CLAIM_NOT_APPLICABLE` will be reported, gates P2/P3 are omitted, and no
"LLM-prior improved" claim will be made. Paper-direction consequence: the
manuscript's method section shifts from prior-informed linear fusion to
calibrated nonlinear ensembling; the LLM-prior analyses remain as the
negative-result backbone.

## G. Architecture-matched no-prior control

P0 (comparator) = identical CKE with the RBF kernel replaced by a **linear
kernel** (k(x,x') = x·x'), same grids (λ ∈ {0.3, 1, 3, 10} on the linear
kernel; same weight grid w; same selection rule). The linear-kernel ensemble
is the same architecture with the novel (nonlinear) component ablated; it is
also mathematically ≈ ridge on the same features late-fused with R0. The ONLY
difference between proposed and control is the kernel function (nonlinear vs
linear); there is no semantic prior in either.

## H. Frozen hyperparameter grid

Per task, selected on concatenated 3-fold inner OOF Pearson (max), with
tie-breaks in order: lower OOF RMSE, lower OOF MAE, smaller ensemble weight w,
larger λ, larger bandwidth multiplier m, then deterministic tuple order:

- ensemble weight w ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
- ridge λ ∈ {0.3, 1.0, 3.0, 10.0}
- bandwidth multiplier m ∈ {0.5, 1.0, 2.0} × median heuristic
- kernel: RBF (proposed) / linear (control)
- ridge grid for nothing else; no feature selection; no rank/top-k; no other
  hyperparameters. Kernel input: per-scope standardized FC+SC edges
  (StandardScaler-equivalent, train-scope only). Targets are de-meaned per
  training scope; predictions re-centered with the training mean.

## I. Frozen development seeds

OPENCHALLENGE_DEV_SEEDS = [4747, 4848, 4949, 5050]
outer_folds = 5, inner_folds = 3.
Outer splits: np.random.RandomState(seed).permutation(412) contiguous folds
(412 = 5×82 + 2; first two folds get 83).
Inner splits within each outer-training scope: the same deterministic
partition helper (hash-seeded) used in Phase 3A-FIX2. Historical seeds 0-9
are used ONLY for the strict R0 audit.

## J. Predeclared prediction and biomarker gates

Prediction:
- Gate P1 (beats R0): mean(Proposed − R0) ≥ +0.005 Pearson for BOTH WM and FI;
  seed-level positive deltas ≥ 3/4 for both; and ≥ +0.010 mean on at least one
  task.
- Gate P2 (semantic prior value): NOT APPLICABLE (no prior used).
- Gate P3 (specificity): NOT APPLICABLE (no prior used).

Biomarker:
- Gate B1 (held-out faithfulness): for BOTH tasks,
  mean(top10 ΔRMSE − random10 ΔRMSE) > 0 with seed-level positive differences
  ≥ 3/4, where ΔRMSE = RMSE_masked − RMSE_unmasked (positive = masking top
  biomarkers hurts prediction = faithful). Masking: incident edges of the ROI
  set zeroed in BOTH standardized modalities of outer-test subjects; the FULL
  proposed prediction (both components) recomputed without retraining.
- Gate B2 (architecture-matched advantage): proposed (RBF) vs control
  (linear-kernel twin), per task, positive mean improvement on ≥ 2 of:
  (i) absolute-rank stability (mean pairwise Spearman of ā over the 20 outer
  fits), (ii) top-10 ROI Jaccard stability, (iii) held-out top-vs-random
  faithfulness (top10−random10), (iv) task-specific perturbation contrast
  (WM-model top10 masked: ΔWM − ΔFI > 0 contrast strength).
- Gate B3 (no circularity): attribution maps are computed from outer-training
  model + data only; validation uses outer-test perturbation; no prior
  alignment or training-set-only deletion is used as validation.

Biomarker mapping (predeclared): attribution lives on the 6,670 FC upper-tri
edges and 6,670 SC edges jointly (13,340); the 6670-edge Spearman/Jaccard use
the FC block; ROI score uses Σ over incident edges of both blocks. If a
selected fit has w = 0 (kernel abstains), that fit's kernel attribution is
marked ABSTAINED and excluded from stability aggregates; the count is
reported.

## K. Compute estimate

- R0 evaluations: 20 outer splits × (3 inner + 1 outer) fit_r0_predict calls
  ≈ 80 × 6.5 s ≈ 9 min (plus strict audit ≈ 6 min, cached).
- KRR: per outer split, 12 kernel configs × 3 inner folds × 2 tasks dual
  solves (≤330³ flops each) + kernel construction (≤330² × 13,340 ≈ 1.5e9
  flops per fold) — seconds. Attribution maps: 330 × 6670 per fit — trivial.
- Total pipeline < 30 min single-core-ish; memory < 2 GB.

---

FREEZE: after MODEL_PROPOSAL_LOCK.sha256 is written, no change to the method,
grid, seeds, selection rule, attribution definition, or gates is permitted.
Bug fixes must restore exactly this locked computation.
