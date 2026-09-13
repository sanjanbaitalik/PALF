# Illustrative examples (all values traced to saved artifacts)

## 1. High-prior WM ROI and selected incident edges

- Highest-prior ROI: index 3 (Frontal_Sup_L), prior score 1.0000.
- In fold seed=7171/fold=0, the R-MATCHED FC mask contains 115 edges incident to this ROI (mask family roi_incident, size 10, 1105 edges total).

## 2. Example direct top-K edge

- Candidate edge: ROI 3 (Frontal_Sup_L) - ROI 4 (Frontal_Sup_R).
- p_i = 1.0000, p_j = 0.9785, q_ij = p_i*p_j = 0.9785.
- Selected in the fold's FC mask: True.

## 3. Fold example (seed=7171, fold=0, R-MATCHED)

- FC mask: roi_incident size 10 (1105 edges); SC mask: direct_topk size 1200 (1200 edges).
- v = 0.55, alpha = 0.35, lambda_fc = 1000.0, lambda_sc = 1000.0.
- Fold Pearson r: 0.2924.
- Coefficient maps valid (alpha>0): True.

## 4. Biomarker perturbation example (seed=7171, fold=0, R-MATCHED)

- base RMSE = 11.2041.
- top10 masked RMSE = 11.4728 (delta = +0.2687).
- random10 mean delta = +0.0604; top10 - random10 = +0.2083.
- bottom10 delta = -0.0170.

Masking replaces affected raw FC/SC edges with training-fold feature means; no retraining is performed.
