# v3 results summary

Pre-registration: 2026-10-25T13:57:01+00:00 (hash ab54c50ceb8e); deviations: none

## Pools

| pool | domain | n | encoders | grouped | anchor | base | matched pairs | primary D (pp) |
|---|---|---|---|---|---|---|---|---|
| bracs | pathology | 3969 | 12 | True | virchow2 | ctranspath | 2 | -0.65 |
| sicapv2 | pathology | 12081 | 12 | True | virchow2 | ctranspath | 1 | -0.65 |
| mhist | pathology | 3152 | 12 | False | virchow2 | ctranspath | 6 | +0.60 |
| crc_val_7k | pathology | 7180 | 12 | False | prov_gigapath | phikon | 13 | -0.01 |
| nct_crc_100k | pathology | 20000 | 12 | False | phikon_v2 | uni2_h | 10 | +0.03 |
| pcam | pathology | 19611 | 12 | True | prov_gigapath | kaiko_vitb16 | 3 | +0.14 |
| dtd | vision | 1880 | 10 | False | openclip_h14 | dinov2_b | 2 | +0.00 |
| aircraft | vision | 3334 | 10 | False | dinov2_l | convnext_l | 2 | -2.16 |
| eurosat | vision | 20000 | 10 | False | openclip_h14 | siglip_b16 | 7 | +0.24 |
| pets | vision | 3680 | 10 | False | dinov2_l | eva02_clip | 3 | -0.05 |
| food101 | vision | 20000 | 10 | False | eva02_clip | dinov2_b | 1 | +0.60 |
| cifar100 | vision | 20000 | 10 | False | eva02_clip | swin_l | 2 | +0.68 |
| gtsrb | vision | 20000 | 10 | False | openclip_h14 | swin_l | 4 | -0.37 |
| country211 | vision | 20000 | 10 | False | openclip_h14 | convnext_l | 2 | +0.44 |
| emotion | text | 16000 | 9 | False | bge_base | mpnet_base | 4 | -3.05 |
| ag_news | text | 20000 | 9 | False | bge_small | mpnet_base | 0 | - |
| banking77 | text | 9993 | 9 | False | bge_base | minilm_l6 | 0 | - |
| trec_coarse | text | 5452 | 9 | False | e5_base | distilroberta | 1 | -0.33 |
| newsgroups20 | text | 11014 | 9 | False | gte_base | minilm_l6 | 1 | -1.62 |
| sst5 | text | 8544 | 9 | False | bge_small | minilm_l6 | 1 | +0.09 |

## Subset: all

- **A (primary, one pair per pool)**: D = -0.34 pp, 95% CI [-0.84, +0.16], p = 0.171, G = 18
- A (all matched pairs, clustered): D = -0.10 pp, 95% CI [-0.31, +0.12], pairs = 65, G = 18
- B (dose-response): slope -1.43 pp/unit CKA, SE +1.26, p = 0.270, MDE80 = +3.51; interpretable: False
- C (TOST, all matched pairs): largest margin cleared = 1.0 pp (G = 18)

| criterion | mean regret (pp) | median | p90 | top-1 agreement | pools |
|---|---|---|---|---|---|
| cka_diversity | 1.76 | 0.68 | 4.59 | 10% | 20 |
| disagreement_with_base | 1.92 | 1.00 | 4.59 | 5% | 20 |
| competence | 0.27 | 0.01 | 0.70 | 50% | 20 |
| oracle_headroom | 0.28 | 0.07 | 0.88 | 25% | 20 |

- E (joint model, cluster-robust): CKA +1.94 (p = 0.049), disagreement +13.40 (p = 0.030), accuracy +0.133 (p = 0.028); candidates = 166, G = 20

## Subset: pathology

- **A (primary, one pair per pool)**: D = -0.09 pp, 95% CI [-0.60, +0.42], p = 0.664, G = 6
- A (all matched pairs, clustered): D = +0.02 pp, 95% CI [-0.17, +0.22], pairs = 35, G = 6
- B (dose-response): slope +0.11 pp/unit CKA, SE +0.55, p = 0.857, MDE80 = +1.55; interpretable: False
- C (TOST, all matched pairs): largest margin cleared = 1.0 pp (G = 6)

| criterion | mean regret (pp) | median | p90 | top-1 agreement | pools |
|---|---|---|---|---|---|
| cka_diversity | 0.88 | 0.21 | 2.43 | 17% | 6 |
| disagreement_with_base | 1.21 | 1.10 | 2.51 | 17% | 6 |
| competence | 0.07 | 0.01 | 0.20 | 50% | 6 |
| oracle_headroom | 0.41 | 0.14 | 1.08 | 17% | 6 |

- E (joint model, cluster-robust): CKA +0.89 (p = 0.160), disagreement +1.42 (p = 0.826), accuracy -0.014 (p = 0.812); candidates = 60, G = 6

## Subset: vision

- **A (primary, one pair per pool)**: D = -0.08 pp, 95% CI [-0.84, +0.68], p = 0.814, G = 8
- A (all matched pairs, clustered): D = -0.02 pp, 95% CI [-0.35, +0.30], pairs = 23, G = 8
- B (dose-response): slope -0.71 pp/unit CKA, SE +1.03, p = 0.513, MDE80 = +2.88; interpretable: False
- C (TOST, all matched pairs): largest margin cleared = 1.0 pp (G = 8)

| criterion | mean regret (pp) | median | p90 | top-1 agreement | pools |
|---|---|---|---|---|---|
| cka_diversity | 2.66 | 1.00 | 7.96 | 12% | 8 |
| disagreement_with_base | 2.72 | 1.00 | 8.01 | 0% | 8 |
| competence | 0.32 | 0.00 | 0.82 | 62% | 8 |
| oracle_headroom | 0.22 | 0.10 | 0.63 | 25% | 8 |

- E (joint model, cluster-robust): CKA +3.44 (p = 0.066), disagreement +46.83 (p = 0.079), accuracy +0.456 (p = 0.071); candidates = 64, G = 8

## Subset: text

- **A (primary, one pair per pool)**: D = -1.23 pp, 95% CI [-3.48, +1.03], p = 0.182, G = 4
- A (all matched pairs, clustered): D = -0.92 pp, 95% CI [-1.79, -0.06], pairs = 7, G = 4
- B (dose-response): slope -3.92 pp/unit CKA, SE +1.10, p = 0.038, MDE80 = +3.08; interpretable: False
- C (TOST, all matched pairs): largest margin cleared = None pp (G = 4)

| criterion | mean regret (pp) | median | p90 | top-1 agreement | pools |
|---|---|---|---|---|---|
| cka_diversity | 1.42 | 0.84 | 3.29 | 0% | 6 |
| disagreement_with_base | 1.58 | 1.33 | 3.29 | 0% | 6 |
| competence | 0.41 | 0.11 | 1.10 | 33% | 6 |
| oracle_headroom | 0.25 | 0.03 | 0.71 | 33% | 6 |

- E (joint model, cluster-robust): CKA +2.78 (p = 0.283), disagreement +13.47 (p = 0.060), accuracy +0.112 (p = 0.135); candidates = 42, G = 6

## Subset: image

- **A (primary, one pair per pool)**: D = -0.08 pp, 95% CI [-0.51, +0.34], p = 0.675, G = 14
- A (all matched pairs, clustered): D = +0.00 pp, 95% CI [-0.15, +0.15], pairs = 58, G = 14
- B (dose-response): slope -0.43 pp/unit CKA, SE +0.68, p = 0.537, MDE80 = +1.90; interpretable: False
- C (TOST, all matched pairs): largest margin cleared = 1.0 pp (G = 14)

| criterion | mean regret (pp) | median | p90 | top-1 agreement | pools |
|---|---|---|---|---|---|
| cka_diversity | 1.90 | 0.68 | 6.13 | 14% | 14 |
| disagreement_with_base | 2.07 | 1.00 | 6.13 | 7% | 14 |
| competence | 0.21 | 0.00 | 0.34 | 57% | 14 |
| oracle_headroom | 0.30 | 0.10 | 0.76 | 21% | 14 |

- E (joint model, cluster-robust): CKA +1.42 (p = 0.313), disagreement +10.43 (p = 0.282), accuracy +0.111 (p = 0.222); candidates = 124, G = 14
