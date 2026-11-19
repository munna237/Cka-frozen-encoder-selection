# Unmatched pairs across the 20 pools (EXPLORATORY)

pairs of category 'unmatched' in results/pools/*.json: CKA gap >= 0.15 and beyond 3 median subsampling SDs, competence gap > 1.5 points.

| subset | pairs | pools | higher-CKA more accurate | mean signed gap | D [95% CI] | D at zero gap [95% CI] | p |
|---|---|---|---|---|---|---|---|
| all | 198 | 19 | 95% | +10.74 | +1.23 [+0.40, +2.05] | -0.06 [-0.82, +0.70] | 0.88 |
| pathology | 49 | 5 | 90% | +5.71 | +0.72 [+0.07, +1.37] | -0.28 [-0.59, +0.03] | 0.07 |
| vision | 97 | 8 | 98% | +14.59 | +1.74 [+0.04, +3.43] | +0.15 [-1.22, +1.51] | 0.81 |
| text | 52 | 6 | 96% | +8.29 | +0.75 [-0.53, +2.03] | -0.40 [-1.74, +0.94] | 0.47 |
| image | 146 | 13 | 95% | +11.61 | +1.40 [+0.31, +2.49] | +0.07 [-0.80, +0.94] | 0.86 |

| pool | domain | pairs | higher-CKA more accurate | mean signed gap | mean D |
|---|---|---|---|---|---|
| bracs | pathology | 13 | 100% | +7.46 | +1.01 |
| sicapv2 | pathology | 8 | 100% | +9.89 | +1.58 |
| mhist | pathology | 4 | 100% | +3.62 | -0.12 |
| crc_val_7k | pathology | 0 | — | — | — |
| nct_crc_100k | pathology | 8 | 100% | +3.15 | +0.06 |
| pcam | pathology | 16 | 69% | +3.99 | +0.59 |
| dtd | vision | 13 | 100% | +11.42 | +0.74 |
| aircraft | vision | 15 | 100% | +22.09 | +4.95 |
| eurosat | vision | 2 | 50% | +0.53 | +0.12 |
| pets | vision | 12 | 100% | +7.09 | +0.33 |
| food101 | vision | 17 | 100% | +25.87 | +1.29 |
| cifar100 | vision | 11 | 91% | +17.39 | +0.38 |
| gtsrb | vision | 11 | 100% | +2.42 | -0.09 |
| country211 | vision | 16 | 100% | +11.97 | +3.46 |
| emotion | text | 10 | 80% | +1.97 | -0.82 |
| ag_news | text | 1 | 100% | +1.70 | +0.10 |
| banking77 | text | 8 | 100% | +8.27 | -0.02 |
| trec_coarse | text | 10 | 100% | +17.96 | +2.10 |
| newsgroups20 | text | 11 | 100% | +8.78 | +0.70 |
| sst5 | text | 12 | 100% | +5.62 | +1.54 |
