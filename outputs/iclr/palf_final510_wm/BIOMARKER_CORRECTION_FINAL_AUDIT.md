# Biomarker stability correction - final audit

1. Final cohort remains 510 subjects (unchanged).
2. The 98-subject holdout remained sealed (sha256 89c563602778a0a2..., no access).
3. No prediction models were retrained; no fitting code executed.
4. No hyperparameters changed.
5. No prior changed.
6. No CV split changed.
7. No seed changed.
8. Only biomarker stability reporting was corrected.
9. FC/SC modality-specific ROI Jaccard is now computed from the modality-specific coefficient maps.
10. The multimodal ranking (I_MULTI = I_FC + I_SC) is preserved and still defines the top-10 biomarker list.
11. All corrected values trace to saved outer-fold coefficients in `_state/folds/`.
12. All correction tests pass (see tests/test_biomarker_stability_correction_results.txt).

## Before/after (top-10 ROI Jaccard)

| model | old FC t10 J | new FC t10 J | old SC t10 J | new SC t10 J | old MULTI t10 J | new MULTI t10 J |
|---|---|---|---|---|---|---|
| R-MATCHED | 0.4796 | 0.6294 | 0.4796 | 0.4098 | 0.4796 | 0.4796 |
| R-CROSS | 0.5138 | 0.6081 | 0.5138 | 0.4552 | 0.5138 | 0.5138 |
| R-SHUFFLED | 0.5492 | 0.6417 | 0.5492 | 0.5556 | 0.5492 | 0.5492 |
| R-RANDOM | 0.5018 | 0.4778 | 0.5018 | 0.4054 | 0.5018 | 0.5018 |
| N-MATCHED | 0.3875 | 0.5062 | 0.3875 | 0.3683 | 0.3875 | 0.3875 |
| N-CROSS | 0.4419 | 0.4766 | 0.4419 | 0.3746 | 0.4419 | 0.4419 |
| N-SHUFFLED | 0.5492 | 0.5846 | 0.5492 | 0.5195 | 0.5492 | 0.5492 |
| N-RANDOM | 0.4392 | 0.4296 | 0.4392 | 0.3661 | 0.4392 | 0.4392 |

## Corrected stability (mean over valid pairs)

```
     model  arch    prior  n_valid  n_abstained  fc_edge_spearman  sc_edge_spearman  edge_rank_stability  multimodal_top10_roi_jaccard  fc_top10_roi_jaccard  sc_top10_roi_jaccard  top10_roi_jaccard  sign_consistency  top10_delta_rmse  random10_mean  top10_minus_random10  percentile  bottom10_delta_rmse  top5_minus_random5
 R-MATCHED ridge  matched       22            3          0.722875          0.553618             0.638246                      0.479590              0.629385              0.409765           0.506247          0.941910          0.040385       0.042109             -0.001724    0.549545             0.044160            0.020572
   R-CROSS ridge    cross       22            3          0.652057          0.528924             0.590491                      0.513810              0.608115              0.455225           0.525717          0.915220          0.005341       0.029630             -0.024289    0.506818             0.023918            0.061360
R-SHUFFLED ridge shuffled       18            7          0.521849          0.502835             0.512342                      0.549204              0.641670              0.555635           0.582170          0.939070         -0.124600       0.025221             -0.149822    0.330556             0.032263           -0.030979
  R-RANDOM ridge   random       22            3          0.601248          0.670608             0.635928                      0.501766              0.477793              0.405366           0.461641          0.940060         -0.181672       0.032448             -0.214121    0.355909             0.032073           -0.178304
 N-MATCHED   ncr  matched       19            6          0.697746          0.546824             0.622285                      0.387497              0.506198              0.368287           0.420661          0.941404         -0.141417       0.029766             -0.171183    0.336316             0.021674           -0.138862
   N-CROSS   ncr    cross       22            3          0.628998          0.518415             0.573706                      0.441906              0.476586              0.374604           0.431032          0.921193          0.084791       0.037566              0.047225    0.534091             0.029268            0.024602
N-SHUFFLED   ncr shuffled       18            7          0.518807          0.501220             0.510014                      0.549245              0.584620              0.519498           0.551121          0.933248         -0.095874       0.026933             -0.122807    0.321667             0.025130           -0.051210
  N-RANDOM   ncr   random       21            4          0.592949          0.662472             0.627711                      0.439250              0.429628              0.366101           0.411660          0.943007         -0.187330       0.036726             -0.224056    0.360476             0.001575           -0.165038
```
