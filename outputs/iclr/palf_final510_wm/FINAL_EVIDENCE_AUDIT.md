# FINAL 510 WM evidence audit

## A. Runtime
- total report+analysis time: 19s

## B. Cohort
- n = 510 unique subjects; see COHORT_AUDIT.json

## C. R0 validation
- historical implementation audit: PASS

## D. Final WM prediction (mean seed-wise outer-CV)

- R0: r=0.2430 (SD 0.0248), RMSE=11.403, MAE=9.195, delta vs R0=+0.0000, positive seeds 0/5
- R-MATCHED: r=0.2439 (SD 0.0129), RMSE=11.293, MAE=9.130, delta vs R0=+0.0010, positive seeds 3/5
- R-CROSS: r=0.2439 (SD 0.0214), RMSE=11.292, MAE=9.132, delta vs R0=+0.0009, positive seeds 2/5
- R-SHUFFLED: r=0.2313 (SD 0.0259), RMSE=11.365, MAE=9.189, delta vs R0=-0.0117, positive seeds 1/5
- R-RANDOM: r=0.2292 (SD 0.0312), RMSE=11.390, MAE=9.191, delta vs R0=-0.0138, positive seeds 2/5
- N-MATCHED: r=0.2386 (SD 0.0180), RMSE=11.338, MAE=9.155, delta vs R0=-0.0043, positive seeds 2/5
- N-CROSS: r=0.2439 (SD 0.0191), RMSE=11.288, MAE=9.121, delta vs R0=+0.0010, positive seeds 2/5
- N-SHUFFLED: r=0.2320 (SD 0.0210), RMSE=11.330, MAE=9.165, delta vs R0=-0.0110, positive seeds 0/5
- N-RANDOM: r=0.2318 (SD 0.0360), RMSE=11.375, MAE=9.175, delta vs R0=-0.0111, positive seeds 3/5

## E. Semantic prior evidence (architecture-matched)

- C2_R-MATCHED - R-CROSS: +0.0000 (positive seeds 2/5)
- C2_R-MATCHED - R-SHUFFLED: +0.0127 (positive seeds 4/5)
- C2_R-MATCHED - R-RANDOM: +0.0148 (positive seeds 4/5)
- C4_N-MATCHED - N-CROSS: -0.0053 (positive seeds 2/5)
- C4_N-MATCHED - N-SHUFFLED: +0.0067 (positive seeds 5/5)
- C4_N-MATCHED - N-RANDOM: +0.0068 (positive seeds 3/5)

## F. Ablations
- See prediction_ablations.csv and ABLATION_PREDICTION.md.

## G. Biomarker stability

- R-MATCHED: FC Spearman 0.723, SC Spearman nan, top10 Jaccard 0.480, sign 0.847
- R-CROSS: FC Spearman 0.652, SC Spearman nan, top10 Jaccard 0.514, sign 0.906
- R-SHUFFLED: FC Spearman 0.522, SC Spearman 0.503, top10 Jaccard 0.549, sign 0.939
- R-RANDOM: FC Spearman 0.601, SC Spearman nan, top10 Jaccard 0.502, sign 0.863
- N-MATCHED: FC Spearman 0.698, SC Spearman nan, top10 Jaccard 0.387, sign 0.901
- N-CROSS: FC Spearman 0.629, SC Spearman nan, top10 Jaccard 0.442, sign 0.904
- N-SHUFFLED: FC Spearman 0.519, SC Spearman 0.501, top10 Jaccard 0.549, sign 0.933
- N-RANDOM: FC Spearman 0.593, SC Spearman nan, top10 Jaccard 0.439, sign 0.887

## H. Faithfulness

- R-MATCHED: top10 +0.040, random10 +0.042, contrast -0.002, bottom10 +0.044
- R-CROSS: top10 +0.005, random10 +0.030, contrast -0.024, bottom10 +0.024
- R-SHUFFLED: top10 -0.125, random10 +0.025, contrast -0.150, bottom10 +0.032
- R-RANDOM: top10 -0.182, random10 +0.032, contrast -0.214, bottom10 +0.032
- N-MATCHED: top10 -0.141, random10 +0.030, contrast -0.171, bottom10 +0.022
- N-CROSS: top10 +0.085, random10 +0.038, contrast +0.047, bottom10 +0.029
- N-SHUFFLED: top10 -0.096, random10 +0.027, contrast -0.123, bottom10 +0.025
- N-RANDOM: top10 -0.187, random10 +0.037, contrast -0.224, bottom10 +0.002

## I. Top WM biomarkers
- See paper_ready/table_final_wm_roi_ranking.csv and Figure 7.

## J. Novelty
- See paper_ready/NOVELTY_STATEMENT.md.

## K. Paper figures
- paper_ready/fig_method_overview.*; plots/fig2..fig7; supplementary/fig_S1..fig_S8.

## L. Claim levels
- RIDGE_PREDICTION_LEVEL: P1
- BIOMARKER_LEVEL Ridge: B0
- BIOMARKER_LEVEL NCR: B0

## M. Paper status
- FINAL510_PAPER_STATUS: EXPLORATORY_WM_METHOD_PAPER

STATUS: FINAL510_WM_EVIDENCE_FREEZE_COMPLETE
