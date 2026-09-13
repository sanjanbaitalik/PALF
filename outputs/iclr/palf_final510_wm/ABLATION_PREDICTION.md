# Prediction ablations (final 510 WM)

## A_backbone

```
     group variant target seed   regime  pearson      rmse      mae  runtime_s model  v_fc_mean  alpha_mean  n_selected
A_backbone      R0     WM 7171 outer_cv 0.272001 11.341001 9.076649        NaN   NaN        NaN         NaN         NaN
A_backbone      R0     WM 7272 outer_cv 0.235510 11.336634 9.135454        NaN   NaN        NaN         NaN         NaN
A_backbone      R0     WM 7373 outer_cv 0.265439 11.425981 9.215850        NaN   NaN        NaN         NaN         NaN
A_backbone      R0     WM 7474 outer_cv 0.214365 11.542854 9.332867        NaN   NaN        NaN         NaN         NaN
A_backbone      R0     WM 7575 outer_cv 0.227559 11.368980 9.214279        NaN   NaN        NaN         NaN         NaN
```

## B_matched_ridge

```
          group variant target seed regime  pearson      rmse      mae  runtime_s     model  v_fc_mean  alpha_mean  n_selected
B_matched_ridge     NaN    NaN 7171    NaN 0.257496 11.260055 9.081689   3.562641 R-MATCHED        NaN         NaN         NaN
B_matched_ridge     NaN    NaN 7272    NaN 0.238933 11.271224 9.101115   3.562641 R-MATCHED        NaN         NaN         NaN
B_matched_ridge     NaN    NaN 7373    NaN 0.257748 11.288920 9.153421   3.562641 R-MATCHED        NaN         NaN         NaN
B_matched_ridge     NaN    NaN 7474    NaN 0.235434 11.308994 9.125439   3.562641 R-MATCHED        NaN         NaN         NaN
B_matched_ridge     NaN    NaN 7575    NaN 0.230086 11.334906 9.187638   3.562641 R-MATCHED        NaN         NaN         NaN
```

## B_matched_ridge_components

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

## C_ridge_prior_identity

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
```

## D_ncr_ratio_inner_cv

```
               group      variant target seed   regime  pearson  rmse  mae  runtime_s model  v_fc_mean  alpha_mean  n_selected
D_ncr_ratio_inner_cv ratio=0.0_fc     WM  all inner_cv 0.235506   NaN  NaN        NaN   NaN        NaN         NaN         NaN
D_ncr_ratio_inner_cv ratio=0.0_sc     WM  all inner_cv 0.121452   NaN  NaN        NaN   NaN        NaN         NaN         NaN
D_ncr_ratio_inner_cv ratio=0.1_fc     WM  all inner_cv 0.233385   NaN  NaN        NaN   NaN        NaN         NaN         NaN
D_ncr_ratio_inner_cv ratio=0.1_sc     WM  all inner_cv 0.128808   NaN  NaN        NaN   NaN        NaN         NaN         NaN
D_ncr_ratio_inner_cv ratio=0.3_fc     WM  all inner_cv 0.234285   NaN  NaN        NaN   NaN        NaN         NaN         NaN
D_ncr_ratio_inner_cv ratio=0.3_sc     WM  all inner_cv 0.129051   NaN  NaN        NaN   NaN        NaN         NaN         NaN
D_ncr_ratio_inner_cv ratio=1.0_fc     WM  all inner_cv 0.235937   NaN  NaN        NaN   NaN        NaN         NaN         NaN
D_ncr_ratio_inner_cv ratio=1.0_sc     WM  all inner_cv 0.129342   NaN  NaN        NaN   NaN        NaN         NaN         NaN
```

## D_ncr_ratio_outer

```
            group   variant target seed                  regime  pearson      rmse      mae  runtime_s model  v_fc_mean  alpha_mean  n_selected
D_ncr_ratio_outer ratio=0.0     WM 7171 outer_cv_reduced_budget 0.262642 11.245874 9.081689   0.000466   NaN       0.78        0.34         NaN
D_ncr_ratio_outer ratio=0.1     WM 7171 outer_cv_reduced_budget 0.261183 11.283981 9.117865  12.739911   NaN       0.76        0.26         NaN
D_ncr_ratio_outer ratio=0.3     WM 7171 outer_cv_reduced_budget 0.258872 11.322243 9.123088  12.250948   NaN       0.75        0.27         NaN
D_ncr_ratio_outer ratio=1.0     WM 7171 outer_cv_reduced_budget 0.259188 11.290730 9.102430  12.785377   NaN       0.73        0.31         NaN
```

## E_ncr_prior_identity

```
               group variant target seed regime  pearson      rmse      mae  runtime_s      model  v_fc_mean  alpha_mean  n_selected
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

## F_G_mask_family_size

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

## F/G mask candidate inner-CV detail

```
                    inner_r  selected  n_edges
family       size                             
direct_topk  100   0.099543         1    100.0
             300   0.119768         1    300.0
             600   0.123409         0    600.0
             1200  0.151124        12   1200.0
roi_incident 5     0.145557        12    565.0
             10    0.151811        11   1105.0
             15    0.151064        13   1620.0
```
