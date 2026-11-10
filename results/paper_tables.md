# v3 paper tables

Registered 2026-10-25T13:57:01+00:00 (hash ab54c50ceb8e); deviations: none.

## Pools

| pool | domain | n | encoders | grouped | anchor | base_member | matched_pairs | primary_pair | primary_cka_gap | primary_comp_gap_pp | primary_D_pp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ag_news | text | 20000 | 9 | False | bge_small | mpnet_base | 0 |  | — | — | — |
| aircraft | vision | 3334 | 10 | False | dinov2_l | convnext_l | 2 | dinov2_b vs openclip_h14 | 0.45 | 0.72 | -2.16 |
| banking77 | text | 9993 | 9 | False | bge_base | minilm_l6 | 0 |  | — | — | — |
| bracs | pathology | 3969 | 12 | True | virchow2 | ctranspath | 2 | hibou_b vs prov_gigapath | 0.18 | 0.50 | -0.65 |
| cifar100 | vision | 20000 | 10 | False | eva02_clip | swin_l | 2 | openclip_h14 vs convnext_l | 0.21 | 1.03 | 0.68 |
| country211 | vision | 20000 | 10 | False | openclip_h14 | convnext_l | 2 | clip_b32 vs dinov2_l | 0.26 | 0.46 | 0.44 |
| crc_val_7k | pathology | 7180 | 12 | False | prov_gigapath | phikon | 13 | h_optimus_0 vs virchow2 | 0.25 | 0.06 | -0.01 |
| dtd | vision | 1880 | 10 | False | openclip_h14 | dinov2_b | 2 | siglip_b16 vs convnext_l | 0.17 | 0.90 | 0.00 |
| emotion | text | 16000 | 9 | False | bge_base | mpnet_base | 4 | gte_base vs glove_avg | 0.61 | 1.34 | -3.05 |
| eurosat | vision | 20000 | 10 | False | openclip_h14 | siglip_b16 | 7 | eva02_clip vs vit_mae | 0.61 | 1.06 | 0.24 |
| food101 | vision | 20000 | 10 | False | eva02_clip | dinov2_b | 1 | openclip_h14 vs dinov2_l | 0.16 | 1.02 | 0.60 |
| gtsrb | vision | 20000 | 10 | False | openclip_h14 | swin_l | 4 | siglip_b16 vs vit_mae | 0.55 | 1.38 | -0.37 |
| mhist | pathology | 3152 | 12 | False | virchow2 | ctranspath | 6 | uni2_h vs phikon | 0.25 | 0.76 | 0.60 |
| nct_crc_100k | pathology | 20000 | 12 | False | phikon_v2 | uni2_h | 10 | phikon vs virchow2 | 0.47 | 0.01 | 0.03 |
| newsgroups20 | text | 11014 | 9 | False | gte_base | minilm_l6 | 1 | bge_base vs mpnet_base | 0.23 | 0.25 | -1.62 |
| pcam | pathology | 19611 | 12 | True | prov_gigapath | kaiko_vitb16 | 3 | uni vs virchow2 | 0.39 | 0.68 | 0.14 |
| pets | vision | 3680 | 10 | False | dinov2_l | eva02_clip | 3 | dinov2_b vs openclip_h14 | 0.28 | 0.52 | -0.05 |
| sicapv2 | pathology | 12081 | 12 | True | virchow2 | ctranspath | 1 | kaiko_vitb16 vs h_optimus_0 | 0.17 | 0.57 | -0.65 |
| sst5 | text | 8544 | 9 | False | bge_small | minilm_l6 | 1 | minilm_l3 vs glove_avg | 0.20 | 0.11 | 0.09 |
| trec_coarse | text | 5452 | 9 | False | e5_base | distilroberta | 1 | gte_base vs minilm_l6 | 0.15 | 0.95 | -0.33 |

## A. Matched-competence null (REGISTERED; D = utility(higher-CKA) − utility(lower-CKA), pp)

| subset | analysis | D_pp | ci_low | ci_high | p | pairs | G | reference | mixed_effects_D | mixed_effects_p |
|---|---|---|---|---|---|---|---|---|---|---|
| all | primary (1 pair/pool) | -0.34 | -0.84 | 0.16 | 0.17 | 18 | 18 | t(G-1) = t(17) | -0.34 | 1.18e-130 |
| all | all matched pairs | -0.10 | -0.31 | 0.12 | 0.36 | 65 | 18 | t(G-1) = t(17) | -0.19 | 0.18 |
| pathology | primary (1 pair/pool) | -0.09 | -0.60 | 0.42 | 0.66 | 6 | 6 | t(G-1) = t(5) | -0.09 | 6.38e-38 |
| pathology | all matched pairs | 0.02 | -0.17 | 0.22 | 0.79 | 35 | 6 | t(G-1) = t(5) | -0.09 | 0.56 |
| vision | primary (1 pair/pool) | -0.08 | -0.84 | 0.68 | 0.81 | 8 | 8 | t(G-1) = t(7) | -0.08 | — |
| vision | all matched pairs | -0.02 | -0.35 | 0.30 | 0.86 | 23 | 8 | t(G-1) = t(7) | -0.03 | 0.89 |
| text | primary (1 pair/pool) | -1.23 | -3.48 | 1.03 | 0.18 | 4 | 4 | t(G-1) = t(3) | -1.23 | 7.62e-25 |
| text | all matched pairs | -0.92 | -1.79 | -0.06 | 0.04 | 7 | 4 | t(G-1) = t(3) | -0.92 | — |
| image | primary (1 pair/pool) | -0.08 | -0.51 | 0.34 | 0.67 | 14 | 14 | t(G-1) = t(13) | -0.08 | 2.13e-93 |
| image | all matched pairs | 0.00 | -0.15 | 0.15 | 0.96 | 58 | 14 | t(G-1) = t(13) | -0.05 | 0.68 |

## C. Equivalence, TOST (REGISTERED; all matched pairs, cluster-robust)

`tightest_margin_cleared` is the claim to make. `summary_md_printed` shows what summary.md reported (the largest margin cleared, i.e. the weakest claim).


| subset | G | estimate_pp | ci90_low | ci90_high | tightest_margin_cleared | summary_md_printed | p_tost_1.0 | p_tost_0.5 | p_tost_0.25 |
|---|---|---|---|---|---|---|---|---|---|
| all | 18 | -0.10 | -0.27 | 0.08 | 0.50 | 1.00 | 4.43e-08 | 0.000509 | 0.08 |
| pathology | 6 | 0.02 | -0.13 | 0.18 | 0.25 | 1.00 | 2.53e-05 | 0.000748 | 0.02 |
| vision | 8 | -0.02 | -0.29 | 0.24 | 0.50 | 1.00 | 0.000105 | 0.01 | 0.07 |
| text | 4 | -0.92 | -1.56 | -0.28 | — | — | 0.40 | 0.89 | 0.96 |
| image | 14 | 0.00 | -0.12 | 0.13 | 0.25 | 1.00 | 1.16e-09 | 3.65e-06 | 0.00 |

## B and E. Dose-response and mechanism (REGISTERED)

| subset | B_slope_pp_per_cka | B_p | B_mde80 | B_interpretable(MDE<=0.5) | E_cka_coef | E_cka_p | E_disagreement_coef | E_disagreement_p | E_acc_coef | E_acc_p | E_G |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | -1.43 | 0.27 | 3.51 | False | 1.94 | 0.05 | 13.40 | 0.03 | 0.13 | 0.03 | 20 |
| pathology | 0.11 | 0.86 | 1.55 | False | 0.89 | 0.16 | 1.42 | 0.83 | -0.01 | 0.81 | 6 |
| vision | -0.71 | 0.51 | 2.88 | False | 3.44 | 0.07 | 46.83 | 0.08 | 0.46 | 0.07 | 8 |
| text | -3.92 | 0.04 | 3.08 | False | 2.78 | 0.28 | 13.47 | 0.06 | 0.11 | 0.13 | 6 |
| image | -0.43 | 0.54 | 1.90 | False | 1.42 | 0.31 | 10.43 | 0.28 | 0.11 | 0.22 | 14 |

## D. Selection regret (REGISTERED)

| subset | criterion | mean_regret_pp | median_regret_pp | p90_regret_pp | top1_agreement_with_oracle | n_pools |
|---|---|---|---|---|---|---|
| all | cka_diversity | 1.76 | 0.68 | 4.59 | 0.10 | 20 |
| all | disagreement_with_base | 1.92 | 1.00 | 4.59 | 0.05 | 20 |
| all | competence | 0.27 | 0.01 | 0.70 | 0.50 | 20 |
| all | oracle_headroom | 0.28 | 0.07 | 0.88 | 0.25 | 20 |
| pathology | cka_diversity | 0.88 | 0.21 | 2.43 | 0.17 | 6 |
| pathology | disagreement_with_base | 1.21 | 1.10 | 2.51 | 0.17 | 6 |
| pathology | competence | 0.07 | 0.01 | 0.20 | 0.50 | 6 |
| pathology | oracle_headroom | 0.41 | 0.14 | 1.08 | 0.17 | 6 |
| vision | cka_diversity | 2.66 | 1.00 | 7.96 | 0.12 | 8 |
| vision | disagreement_with_base | 2.72 | 1.00 | 8.01 | 0.00 | 8 |
| vision | competence | 0.32 | 0.00 | 0.82 | 0.62 | 8 |
| vision | oracle_headroom | 0.22 | 0.10 | 0.63 | 0.25 | 8 |
| text | cka_diversity | 1.42 | 0.84 | 3.29 | 0.00 | 6 |
| text | disagreement_with_base | 1.58 | 1.33 | 3.29 | 0.00 | 6 |
| text | competence | 0.41 | 0.11 | 1.10 | 0.33 | 6 |
| text | oracle_headroom | 0.25 | 0.03 | 0.71 | 0.33 | 6 |
| image | cka_diversity | 1.90 | 0.68 | 6.13 | 0.14 | 14 |
| image | disagreement_with_base | 2.07 | 1.00 | 6.13 | 0.07 | 14 |
| image | competence | 0.21 | 0.00 | 0.34 | 0.57 | 14 |
| image | oracle_headroom | 0.30 | 0.10 | 0.76 | 0.21 | 14 |

## D′. Paired regret, CKA diversity minus competence (EXPLORATORY — not a registered test)

| subset | pools | mean_extra_regret_cka_vs_competence_pp | ci_low | ci_high | t_p | wilcoxon_p | cka_worse | tie | cka_better |
|---|---|---|---|---|---|---|---|---|---|
| all | 20 | 1.49 | 0.41 | 2.56 | 0.01 | 0.00 | 15 | 0 | 5 |
| pathology | 6 | 0.81 | -0.55 | 2.17 | 0.19 | 0.31 | 4 | 0 | 2 |
| vision | 8 | 2.35 | -0.19 | 4.89 | 0.07 | 0.02 | 7 | 0 | 1 |
| text | 6 | 1.02 | -0.93 | 2.96 | 0.24 | 0.44 | 4 | 0 | 2 |
| image | 14 | 1.69 | 0.25 | 3.13 | 0.03 | 0.01 | 11 | 0 | 3 |

## Leave-one-pool-out, text subset (EXPLORATORY)

| subset | analysis | dropped | D_pp | ci_low | ci_high | p | G | pairs |
|---|---|---|---|---|---|---|---|---|
| text | all pairs | (none) | -0.92 | -1.79 | -0.06 | 0.04 | 4 | 7 |
| text | all pairs | emotion | -0.62 | -2.83 | 1.59 | 0.35 | 3 | 3 |
| text | all pairs | ag_news | -0.92 | -1.79 | -0.06 | 0.04 | 4 | 7 |
| text | all pairs | banking77 | -0.92 | -1.79 | -0.06 | 0.04 | 4 | 7 |
| text | all pairs | trec_coarse | -1.02 | -2.22 | 0.18 | 0.07 | 3 | 6 |
| text | all pairs | newsgroups20 | -0.81 | -2.32 | 0.70 | 0.15 | 3 | 6 |
| text | all pairs | sst5 | -1.09 | -1.93 | -0.25 | 0.03 | 3 | 6 |
| text | primary | (none) | -1.23 | -3.48 | 1.03 | 0.18 | 4 | 4 |
| text | primary | emotion | -0.62 | -2.83 | 1.59 | 0.35 | 3 | 3 |
| text | primary | ag_news | -1.23 | -3.48 | 1.03 | 0.18 | 4 | 4 |
| text | primary | banking77 | -1.23 | -3.48 | 1.03 | 0.18 | 4 | 4 |
| text | primary | trec_coarse | -1.52 | -5.43 | 2.39 | 0.24 | 3 | 3 |
| text | primary | newsgroups20 | -1.10 | -5.33 | 3.14 | 0.38 | 3 | 3 |
| text | primary | sst5 | -1.67 | -5.05 | 1.71 | 0.17 | 3 | 3 |

## Leave-one-pool-out, all pools (EXPLORATORY)

| subset | analysis | dropped | D_pp | ci_low | ci_high | p | G | pairs |
|---|---|---|---|---|---|---|---|---|
| all | all pairs | (none) | -0.10 | -0.31 | 0.12 | 0.36 | 18 | 65 |
| all | all pairs | bracs | -0.08 | -0.30 | 0.14 | 0.43 | 17 | 63 |
| all | all pairs | sicapv2 | -0.09 | -0.31 | 0.13 | 0.41 | 17 | 64 |
| all | all pairs | mhist | -0.14 | -0.37 | 0.08 | 0.19 | 17 | 59 |
| all | all pairs | crc_val_7k | -0.12 | -0.39 | 0.15 | 0.36 | 17 | 52 |
| all | all pairs | nct_crc_100k | -0.12 | -0.37 | 0.14 | 0.35 | 17 | 55 |
| all | all pairs | pcam | -0.10 | -0.33 | 0.12 | 0.34 | 17 | 62 |
| all | all pairs | dtd | -0.10 | -0.32 | 0.13 | 0.36 | 17 | 63 |
| all | all pairs | aircraft | -0.06 | -0.27 | 0.14 | 0.53 | 17 | 63 |
| all | all pairs | eurosat | -0.13 | -0.36 | 0.11 | 0.28 | 17 | 58 |
| all | all pairs | pets | -0.10 | -0.32 | 0.13 | 0.39 | 17 | 62 |
| all | all pairs | food101 | -0.11 | -0.33 | 0.11 | 0.32 | 17 | 64 |
| all | all pairs | cifar100 | -0.11 | -0.34 | 0.11 | 0.29 | 17 | 63 |
| all | all pairs | gtsrb | -0.09 | -0.32 | 0.14 | 0.42 | 17 | 61 |
| all | all pairs | country211 | -0.11 | -0.33 | 0.12 | 0.32 | 17 | 63 |
| all | all pairs | emotion | -0.03 | -0.18 | 0.13 | 0.71 | 17 | 61 |
| all | all pairs | ag_news | -0.10 | -0.31 | 0.12 | 0.36 | 18 | 65 |
| all | all pairs | banking77 | -0.10 | -0.31 | 0.12 | 0.36 | 18 | 65 |
| all | all pairs | trec_coarse | -0.09 | -0.31 | 0.13 | 0.38 | 17 | 64 |
| all | all pairs | newsgroups20 | -0.07 | -0.28 | 0.14 | 0.47 | 17 | 64 |
| all | all pairs | sst5 | -0.10 | -0.32 | 0.12 | 0.35 | 17 | 64 |
| all | primary | (none) | -0.34 | -0.84 | 0.16 | 0.17 | 18 | 18 |
| all | primary | bracs | -0.32 | -0.85 | 0.21 | 0.22 | 17 | 17 |
| all | primary | sicapv2 | -0.32 | -0.85 | 0.21 | 0.22 | 17 | 17 |
| all | primary | mhist | -0.39 | -0.91 | 0.12 | 0.13 | 17 | 17 |
| all | primary | crc_val_7k | -0.36 | -0.89 | 0.17 | 0.17 | 17 | 17 |
| all | primary | nct_crc_100k | -0.36 | -0.89 | 0.17 | 0.17 | 17 | 17 |
| all | primary | pcam | -0.37 | -0.89 | 0.16 | 0.16 | 17 | 17 |
| all | primary | dtd | -0.36 | -0.89 | 0.17 | 0.17 | 17 | 17 |
| all | primary | aircraft | -0.23 | -0.70 | 0.24 | 0.32 | 17 | 17 |
| all | primary | eurosat | -0.37 | -0.90 | 0.15 | 0.15 | 17 | 17 |
| all | primary | pets | -0.35 | -0.88 | 0.18 | 0.18 | 17 | 17 |
| all | primary | food101 | -0.39 | -0.91 | 0.12 | 0.13 | 17 | 17 |
| all | primary | cifar100 | -0.40 | -0.91 | 0.12 | 0.12 | 17 | 17 |
| all | primary | gtsrb | -0.34 | -0.87 | 0.19 | 0.20 | 17 | 17 |
| all | primary | country211 | -0.38 | -0.90 | 0.14 | 0.14 | 17 | 17 |
| all | primary | emotion | -0.18 | -0.57 | 0.21 | 0.35 | 17 | 17 |
| all | primary | ag_news | -0.34 | -0.84 | 0.16 | 0.17 | 18 | 18 |
| all | primary | banking77 | -0.34 | -0.84 | 0.16 | 0.17 | 18 | 18 |
| all | primary | trec_coarse | -0.34 | -0.87 | 0.19 | 0.20 | 17 | 17 |
| all | primary | newsgroups20 | -0.26 | -0.77 | 0.24 | 0.29 | 17 | 17 |
| all | primary | sst5 | -0.36 | -0.89 | 0.16 | 0.16 | 17 | 17 |

## Secondary fusion rule: fits reaching the registered 500-iteration cap

Pools analysed before the fit log existed (the pathology pools and DTD, Aircraft, EuroSAT, Pets) are not listed.


| pool | classes | fits | capped | capped_pct |
|---|---|---|---|---|
| food101 | 101 | 45 | 14 | 31.10 |
| cifar100 | 100 | 45 | 7 | 15.60 |
| gtsrb | 43 | 45 | 0 | 0.00 |
| country211 | 211 | 45 | 0 | 0.00 |
| emotion | 6 | 40 | 0 | 0.00 |
| ag_news | 4 | 40 | 0 | 0.00 |
| banking77 | 77 | 40 | 33 | 82.50 |
| trec_coarse | 6 | 40 | 0 | 0.00 |
| newsgroups20 | 20 | 40 | 0 | 0.00 |
| sst5 | 5 | 40 | 0 | 0.00 |
