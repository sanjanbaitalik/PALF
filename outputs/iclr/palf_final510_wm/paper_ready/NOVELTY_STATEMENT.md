# Methodological contributions

## Contribution 1 - LLM-generated task semantics as a frozen atlas prior
A task description is converted into an AAL116 ROI prior by a frozen LLM call (temperature 0.2, seed 42). The prior is generated independently of HCP labels, prediction residuals, model coefficients, and test folds, and is versioned and SHA256-hashed.

## Contribution 2 - Prior-guided subspace selection, not global prior regularization
The prior defines compact FC/SC expert subspaces via direct top-K edge masks (q_ij = p_i p_j) and ROI-incident masks (top-M ROIs), rather than penalizing all 6670 edges.

## Contribution 3 - Hierarchical fusion with a strong no-prior backbone
The model preserves the validated corrected R0 backbone and adds a restricted prior-guided expert: yhat = (1-alpha) yhat_R0 + alpha (v yhat_FC + (1-v) yhat_SC), with v and alpha selected on training OOF only.

## Contribution 4 - Architecture-matched semantic controls
Matched, cross-task, shuffled, and random priors are evaluated under identical architecture, grids, preprocessing, folds, and tie-breaks; only the prior array changes. This separates semantic prior value from generic regularization or feature restriction. Claims are conditional on these controls (Tier P3).

## Contribution 5 - Prediction-linked biomarker discovery
The same expert that contributes to prediction yields primal coefficient maps c_m = alpha * w_m * beta_m, ROI importance I_i = sum_j |c_FC_ij| + sum_j |c_SC_ij|, evaluated by cross-fold stability and outer-fold perturbation faithfulness (masking edges to training-fold means without retraining).

Novelty is not claimed merely because an LLM is used: the prior enters as a frozen external score vector that controls which edges are estimable, and its value is tested against architecture-matched controls.
