# Supplementary ablations - final 510 WM

## Per-seed metrics

seed          7171    7272    7373    7474    7575
model                                             
N-CROSS     0.2648  0.2428  0.2593  0.2354  0.2174
N-MATCHED   0.2557  0.2217  0.2605  0.2272  0.2281
N-RANDOM    0.2529  0.2410  0.2710  0.2163  0.1781
N-SHUFFLED  0.2496  0.2215  0.2592  0.2125  0.2172
R-CROSS     0.2665  0.2389  0.2586  0.2447  0.2109
R-MATCHED   0.2575  0.2389  0.2577  0.2354  0.2301
R-RANDOM    0.2331  0.2336  0.2727  0.2209  0.1855
R-SHUFFLED  0.2539  0.2118  0.2644  0.2177  0.2085
R0          0.2720  0.2355  0.2654  0.2144  0.2276

## Per-fold metrics (all 25 folds)

See `fold_metrics.csv`.

## Selected masks / lambdas / ratios / v / alpha

See `SELECTED_CONFIGS.csv`. Summary:

```
                                          fc_family                                                 fc_size                               sc_family                                                sc_size  lambda_fc_mean  lambda_sc_mean  ratio_fc_mean  ratio_sc_mean  v_mean  alpha_mean  abstained
model                                                                                                                                                                                                                                                                                                  
N-CROSS      {'roi_incident': 17, 'direct_topk': 8}                    {15: 8, 10: 8, 600: 7, 5: 1, 100: 1}  {'direct_topk': 22, 'roi_incident': 3}  {100: 9, 300: 6, 1200: 4, 600: 3, 5: 1, 15: 1, 10: 1}       400.24000        54.08812          0.500          0.416   0.756       0.286          3
N-MATCHED    {'roi_incident': 18, 'direct_topk': 7}                                 {10: 9, 15: 9, 1200: 7}  {'roi_incident': 18, 'direct_topk': 7}         {5: 12, 1200: 5, 15: 4, 10: 2, 300: 1, 100: 1}       360.31600       130.44880          0.448          0.464   0.670       0.234          6
N-RANDOM    {'roi_incident': 14, 'direct_topk': 11}                 {1200: 8, 15: 7, 10: 7, 300: 2, 600: 1}  {'roi_incident': 18, 'direct_topk': 7}                 {5: 13, 1200: 6, 10: 3, 15: 2, 600: 1}        56.06920        82.16812          0.520          0.512   0.576       0.322          4
N-SHUFFLED  {'roi_incident': 15, 'direct_topk': 10}  {15: 11, 100: 7, 10: 3, 5: 1, 600: 1, 300: 1, 1200: 1}  {'direct_topk': 16, 'roi_incident': 9}  {600: 5, 100: 5, 300: 4, 15: 3, 5: 3, 10: 3, 1200: 2}        93.64204       203.28400          0.356          0.320   0.508       0.270          7
R-CROSS      {'roi_incident': 17, 'direct_topk': 8}                    {15: 8, 10: 8, 600: 7, 5: 1, 100: 1}  {'direct_topk': 22, 'roi_incident': 3}  {100: 9, 300: 6, 1200: 4, 600: 3, 5: 1, 15: 1, 10: 1}       780.40000       620.40016          0.000          0.000   0.730       0.268          3
R-MATCHED    {'roi_incident': 18, 'direct_topk': 7}                                 {10: 9, 15: 9, 1200: 7}  {'roi_incident': 18, 'direct_topk': 7}         {5: 12, 1200: 5, 15: 4, 10: 2, 300: 1, 100: 1}       964.00000       784.00000          0.000          0.000   0.714       0.246          3
R-RANDOM    {'roi_incident': 14, 'direct_topk': 11}                 {1200: 8, 15: 7, 10: 7, 300: 2, 600: 1}  {'roi_incident': 18, 'direct_topk': 7}                 {5: 13, 1200: 6, 10: 3, 15: 2, 600: 1}       416.80000       804.00016          0.000          0.000   0.614       0.304          3
R-SHUFFLED  {'roi_incident': 15, 'direct_topk': 10}  {15: 11, 100: 7, 10: 3, 5: 1, 600: 1, 300: 1, 1200: 1}  {'direct_topk': 16, 'roi_incident': 9}  {600: 5, 100: 5, 300: 4, 15: 3, 5: 3, 10: 3, 1200: 2}       250.44004       928.00000          0.000          0.000   0.572       0.240          7
```

## FC-only / SC-only / expert / final (R-MATCHED)

```
                     group        variant target seed   regime   pearson      rmse       mae  runtime_s model  v_fc_mean  alpha_mean  n_selected
B_matched_ridge_components        FC_only     WM 7171 outer_cv  0.247298 11.261816  9.047815        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components        FC_only     WM 7272 outer_cv  0.246548 11.435282  9.205839        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components        FC_only     WM 7373 outer_cv  0.259458 11.224716  9.039121        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components        FC_only     WM 7474 outer_cv  0.238419 11.321373  9.002153        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components        FC_only     WM 7575 outer_cv  0.232119 11.319554  9.061240        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components        SC_only     WM 7171 outer_cv  0.017755 13.319567 10.415443        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components        SC_only     WM 7272 outer_cv -0.009337 13.306113 10.509404        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components        SC_only     WM 7373 outer_cv  0.062291 12.621446 10.166979        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components        SC_only     WM 7474 outer_cv  0.075780 11.828624  9.381861        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components        SC_only     WM 7575 outer_cv  0.059540 11.998118  9.712795        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components   FC_SC_expert     WM 7171 outer_cv  0.212391 11.359437  9.195383        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components   FC_SC_expert     WM 7272 outer_cv  0.150995 11.958679  9.598942        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components   FC_SC_expert     WM 7373 outer_cv  0.232909 11.261789  9.143031        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components   FC_SC_expert     WM 7474 outer_cv  0.227710 11.311083  9.000078        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components   FC_SC_expert     WM 7575 outer_cv  0.177438 11.437191  9.240246        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components R0_plus_expert     WM 7171 outer_cv  0.257496 11.260055  9.081689        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components R0_plus_expert     WM 7272 outer_cv  0.238933 11.271224  9.101115        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components R0_plus_expert     WM 7373 outer_cv  0.257748 11.288920  9.153421        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components R0_plus_expert     WM 7474 outer_cv  0.235434 11.308994  9.125439        NaN   NaN        NaN         NaN         NaN
B_matched_ridge_components R0_plus_expert     WM 7575 outer_cv  0.230086 11.334906  9.187638        NaN   NaN        NaN         NaN         NaN
```

## Direct-edge vs ROI-incident (candidate inner-CV)

```
               group          variant target seed             regime  pearson      rmse       mae  runtime_s model  v_fc_mean  alpha_mean  n_selected
F_G_mask_family_size  direct_topk_100     WM  all inner_cv_selection 0.099543 11.733842  9.465459   0.044548   NaN        NaN         NaN         1.0
F_G_mask_family_size  direct_topk_300     WM  all inner_cv_selection 0.119768 13.316285 10.605568   0.067997   NaN        NaN         NaN         1.0
F_G_mask_family_size  direct_topk_600     WM  all inner_cv_selection 0.123409 11.787691  9.513076   0.089011   NaN        NaN         NaN         0.0
F_G_mask_family_size direct_topk_1200     WM  all inner_cv_selection 0.151124 11.596833  9.352640   0.121691   NaN        NaN         NaN        12.0
F_G_mask_family_size   roi_incident_5     WM  all inner_cv_selection 0.145557 12.287102  9.875186   0.088999   NaN        NaN         NaN        12.0
F_G_mask_family_size  roi_incident_10     WM  all inner_cv_selection 0.151811 11.613456  9.361770   0.132070   NaN        NaN         NaN        11.0
F_G_mask_family_size  roi_incident_15     WM  all inner_cv_selection 0.151064 11.735156  9.425165   0.148105   NaN        NaN         NaN        13.0
```

## Ridge vs NCR and prior identity

```
                 group variant target seed regime  pearson      rmse      mae  runtime_s      model  v_fc_mean  alpha_mean  n_selected
C_ridge_prior_identity     NaN    NaN 7171    NaN 0.266495 11.256077 9.062105   5.301604    R-CROSS        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7272    NaN 0.238892 11.309062 9.117496   5.301604    R-CROSS        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7373    NaN 0.258582 11.221357 9.110655   5.301604    R-CROSS        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7474    NaN 0.244699 11.281521 9.118627   5.301604    R-CROSS        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7575    NaN 0.210912 11.393149 9.252460   5.301604    R-CROSS        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7171    NaN 0.253853 11.258984 9.091560   4.984797 R-SHUFFLED        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7272    NaN 0.211831 11.386975 9.181626   4.984797 R-SHUFFLED        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7373    NaN 0.264386 11.345443 9.193206   4.984797 R-SHUFFLED        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7474    NaN 0.217733 11.439163 9.227817   4.984797 R-SHUFFLED        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7575    NaN 0.208485 11.396851 9.251630   4.984797 R-SHUFFLED        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7171    NaN 0.233145 11.305223 9.154686   5.270870   R-RANDOM        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7272    NaN 0.233591 11.293927 9.079504   5.270870   R-RANDOM        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7373    NaN 0.272684 11.276395 9.113274   5.270870   R-RANDOM        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7474    NaN 0.220879 11.336825 9.151234   5.270870   R-RANDOM        NaN         NaN         NaN
C_ridge_prior_identity     NaN    NaN 7575    NaN 0.185504 11.736723 9.453996   5.270870   R-RANDOM        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7171    NaN 0.255674 11.306037 9.097713  30.898599  N-MATCHED        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7272    NaN 0.221715 11.360061 9.175495  30.898599  N-MATCHED        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7373    NaN 0.260534 11.334635 9.193862  30.898599  N-MATCHED        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7474    NaN 0.227229 11.345283 9.120011  30.898599  N-MATCHED        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7575    NaN 0.228068 11.343045 9.188734  30.898599  N-MATCHED        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7171    NaN 0.264828 11.257222 9.062100  27.239692    N-CROSS        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7272    NaN 0.242836 11.297704 9.096709  27.239692    N-CROSS        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7373    NaN 0.259264 11.264998 9.111838  27.239692    N-CROSS        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7474    NaN 0.235390 11.280757 9.119005  27.239692    N-CROSS        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7575    NaN 0.217353 11.338406 9.213128  27.239692    N-CROSS        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7171    NaN 0.249574 11.249934 9.096259  29.234510 N-SHUFFLED        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7272    NaN 0.221547 11.317044 9.126886  29.234510 N-SHUFFLED        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7373    NaN 0.259161 11.329298 9.203175  29.234510 N-SHUFFLED        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7474    NaN 0.212454 11.399849 9.170837  29.234510 N-SHUFFLED        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7575    NaN 0.217221 11.351933 9.228535  29.234510 N-SHUFFLED        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7171    NaN 0.252854 11.270109 9.099307  30.990239   N-RANDOM        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7272    NaN 0.241006 11.203307 9.023143  30.990239   N-RANDOM        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7373    NaN 0.270978 11.243054 9.089132  30.990239   N-RANDOM        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7474    NaN 0.216262 11.350482 9.155443  30.990239   N-RANDOM        NaN         NaN         NaN
  E_ncr_prior_identity     NaN    NaN 7575    NaN 0.178080 11.808236 9.507446  30.990239   N-RANDOM        NaN         NaN         NaN
```

## NCR ratio ablation (inner-CV diagnostics)

```
Empty DataFrame
Columns: [group, variant, target, seed, regime, pearson, rmse, mae, runtime_s, model, v_fc_mean, alpha_mean, n_selected]
Index: []
```

## Top5/top10 faithfulness, random distributions, abstention

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

> All reported performance estimates are obtained by repeated nested cross-validation within the final 510-subject cohort; no separate external validation cohort is claimed.
