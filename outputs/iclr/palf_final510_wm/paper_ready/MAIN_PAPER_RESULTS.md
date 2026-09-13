# Final 510-subject Working-Memory results

> All reported performance estimates are obtained by repeated nested cross-validation within the final 510-subject cohort; no separate external validation cohort is claimed.

The primary estimand is the mean of seed-wise outer-CV Pearson values across 5 seeds x 5 folds (repeated nested CV). Ensemble sensitivity is reported separately and is not the primary metric.

## 31.1 Working-memory prediction

- R0 backbone: r = 0.2430 (seed SD 0.0248), RMSE = 11.403, MAE = 9.195.
- SPSEF-Ridge (R-MATCHED): r = 0.2439 (seed SD 0.0129), RMSE = 11.293, MAE = 9.130.
- Paired delta r (SPSEF-Ridge - R0): +0.0010 (positive seeds 3/5).
- Ensemble sensitivity delta r vs R0: -0.0033 (95% CI [-0.0174, +0.0109]; fraction <= 0 = 0.679).
- RIDGE_PREDICTION_LEVEL: **P1**.

## 31.2 Does the LLM semantic prior matter?

Architecture-matched Ridge controls isolate prior identity:

- R-MATCHED - R-CROSS: +0.0000 (positive seeds 2/5).
- R-MATCHED - R-SHUFFLED: +0.0127 (positive seeds 4/5).
- R-MATCHED - R-RANDOM: +0.0148 (positive seeds 4/5).

Tier P3 requires positive mean deltas against all controls and >=4/5 positive seeds against shuffled and random; achieved: True.

## 31.3 Structured NCR regularization

- N-MATCHED - R-MATCHED: -0.0053 (positive seeds 1/5).
- NCR does not beat Ridge; NCR is reported as a structured-regularization ablation and not as the strongest predictor.

## 31.4 Biomarker reproducibility

Coefficient stability across valid outer folds under architecture-matched priors is summarised below (see Table 5):

- R-MATCHED: FC Spearman 0.723, SC Spearman 0.554, multimodal top10 Jaccard 0.480, sign consistency 0.942.
- R-CROSS: FC Spearman 0.652, SC Spearman 0.529, multimodal top10 Jaccard 0.514, sign consistency 0.915.
- R-SHUFFLED: FC Spearman 0.522, SC Spearman 0.503, multimodal top10 Jaccard 0.549, sign consistency 0.939.
- R-RANDOM: FC Spearman 0.601, SC Spearman 0.671, multimodal top10 Jaccard 0.502, sign consistency 0.940.
- N-MATCHED: FC Spearman 0.698, SC Spearman 0.547, multimodal top10 Jaccard 0.387, sign consistency 0.941.
- N-CROSS: FC Spearman 0.629, SC Spearman 0.518, multimodal top10 Jaccard 0.442, sign consistency 0.921.
- N-SHUFFLED: FC Spearman 0.519, SC Spearman 0.501, multimodal top10 Jaccard 0.549, sign consistency 0.933.
- N-RANDOM: FC Spearman 0.593, SC Spearman 0.662, multimodal top10 Jaccard 0.439, sign consistency 0.943.

## 31.5 Biomarker faithfulness

Outer-fold perturbation faithfulness (positive = faithful):

- R-MATCHED: top10 delta_RMSE +0.040, random10 mean +0.042, top10-random10 -0.002, percentile 0.550, bottom10 +0.044.
- R-CROSS: top10 delta_RMSE +0.005, random10 mean +0.030, top10-random10 -0.024, percentile 0.507, bottom10 +0.024.
- R-SHUFFLED: top10 delta_RMSE -0.125, random10 mean +0.025, top10-random10 -0.150, percentile 0.331, bottom10 +0.032.
- R-RANDOM: top10 delta_RMSE -0.182, random10 mean +0.032, top10-random10 -0.214, percentile 0.356, bottom10 +0.032.
- N-MATCHED: top10 delta_RMSE -0.141, random10 mean +0.030, top10-random10 -0.171, percentile 0.336, bottom10 +0.022.
- N-CROSS: top10 delta_RMSE +0.085, random10 mean +0.038, top10-random10 +0.047, percentile 0.534, bottom10 +0.029.
- N-SHUFFLED: top10 delta_RMSE -0.096, random10 mean +0.027, top10-random10 -0.123, percentile 0.322, bottom10 +0.025.
- N-RANDOM: top10 delta_RMSE -0.187, random10 mean +0.037, top10-random10 -0.224, percentile 0.360, bottom10 +0.002.

## Decision

- RIDGE_PREDICTION_LEVEL: **P1**
- BIOMARKER_LEVEL (Ridge): **B0**
- BIOMARKER_LEVEL (NCR): **B0**
- FINAL510_PAPER_STATUS: **EXPLORATORY_WM_METHOD_PAPER**

> All reported performance estimates are obtained by repeated nested cross-validation within the final 510-subject cohort; no separate external validation cohort is claimed.

