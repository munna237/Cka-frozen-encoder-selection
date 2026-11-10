# Reviewer analyses (EXPLORATORY)

None of these analyses was part of the registration. They read only the saved per-pool records, held-out logits and frozen features; no probe was refitted. Before anything else the code asserts that the saved logits reproduce every registered per-fold utility (max difference 0.0e+00), that every matched pair's accuracy-based D is reproduced, and that the registered selection rules give the registered regrets when applied in-sample.

## R1. Competence imbalance within matched pairs (review point 2, Q1)

Signed gap = accuracy of the higher-CKA candidate minus the lower-CKA one. Gap-adjusted D is the intercept of D regressed on the signed gap (cluster-robust by pool): the difference expected at equal competence.

| subset | pairs | mean signed gap (pts) | 95% CI | higher-CKA more accurate | gap-adjusted D | 95% CI | p | D per gap point |
|---|---|---|---|---|---|---|---|---|
| all | 65 | +0.33 | [+0.10, +0.56] | 65% | -0.09 | [-0.23, +0.06] | 0.226 | -0.03 |
| pathology | 35 | +0.32 | [-0.15, +0.78] | 63% | -0.01 | [-0.20, +0.18] | 0.852 | +0.12 |
| vision | 23 | +0.36 | [+0.03, +0.69] | 70% | -0.06 | [-0.30, +0.18] | 0.575 | +0.10 |
| text | 7 | +0.31 | [-0.84, +1.47] | 57% | -0.71 | [-1.43, +0.02] | 0.053 | -0.70 |
| image | 58 | +0.33 | [+0.08, +0.58] | 66% | -0.03 | [-0.16, +0.09] | 0.588 | +0.11 |

## R2. Caliper sensitivity (review point 2)

Pairs with CKA gap ≥ 0.15 (and beyond the subsampling tolerance) whose competence gap is within the caliper; 1.5 is the registered caliper.

| subset | caliper (pts) | pairs | G | D all pairs | 95% CI | D primary [95% CI] | TOST (tightest margin) |
|---|---|---|---|---|---|---|---|
| all | 0.5 | 27 | 12 | -0.07 | [-0.23, +0.09] | -0.17 [-0.48, +0.15] | 0.25 |
| all | 1.0 | 50 | 16 | -0.08 | [-0.27, +0.11] | -0.33 [-0.72, +0.07] | 0.25 |
| all | 1.5 | 65 | 18 | -0.10 | [-0.31, +0.12] | -0.34 [-0.84, +0.16] | 0.5 |
| all | 3.0 | 107 | 20 | -0.12 | [-0.40, +0.16] | -0.15 [-0.61, +0.31] | 0.5 |
| pathology | 0.5 | 15 | 4 | -0.04 | [-0.13, +0.06] | -0.11 [-0.35, +0.13] | 0.25 |
| pathology | 1.0 | 28 | 6 | +0.02 | [-0.22, +0.26] | -0.09 [-0.60, +0.42] | 0.25 |
| pathology | 1.5 | 35 | 6 | +0.02 | [-0.17, +0.22] | -0.09 [-0.60, +0.42] | 0.25 |
| pathology | 3.0 | 46 | 6 | +0.07 | [-0.09, +0.24] | +0.26 [-0.01, +0.53] | 0.25 |
| vision | 0.5 | 10 | 6 | +0.02 | [-0.19, +0.23] | -0.01 [-0.28, +0.27] | 0.25 |
| vision | 1.0 | 17 | 6 | -0.10 | [-0.54, +0.34] | -0.30 [-1.28, +0.69] | 0.5 |
| vision | 1.5 | 23 | 8 | -0.02 | [-0.35, +0.30] | -0.08 [-0.84, +0.68] | 0.5 |
| vision | 3.0 | 43 | 8 | -0.03 | [-0.33, +0.26] | -0.12 [-1.01, +0.77] | 0.5 |
| text | 0.5 | 2 | 2 | -0.76 | [-11.62, +10.10] | -0.76 [-11.62, +10.10] | none |
| text | 1.0 | 5 | 4 | -0.57 | [-1.51, +0.36] | -0.73 [-1.94, +0.48] | none |
| text | 1.5 | 7 | 4 | -0.92 | [-1.79, -0.06] | -1.23 [-3.48, +1.03] | none |
| text | 3.0 | 18 | 6 | -0.84 | [-1.85, +0.17] | -0.60 [-1.93, +0.73] | none |
| image | 0.5 | 25 | 10 | -0.01 | [-0.10, +0.07] | -0.05 [-0.20, +0.11] | 0.25 |
| image | 1.0 | 45 | 12 | -0.03 | [-0.21, +0.16] | -0.19 [-0.65, +0.26] | 0.25 |
| image | 1.5 | 58 | 14 | +0.00 | [-0.15, +0.15] | -0.08 [-0.51, +0.34] | 0.25 |
| image | 3.0 | 89 | 14 | +0.02 | [-0.12, +0.17] | +0.04 [-0.43, +0.52] | 0.25 |

## R3. Strong candidates only (review point 3, Q2)

Rule: both candidates within 3.0 points of the anchor's accuracy.

| subset | pairs | pools | D | 95% CI | TOST (tightest margin) |
|---|---|---|---|---|---|
| all | 47 | 12 | -0.12 | [-0.39, +0.14] | 0.5 |
| pathology | 29 | 5 | -0.01 | [-0.09, +0.07] | 0.25 |
| vision | 14 | 5 | +0.06 | [-0.18, +0.29] | 0.25 |
| text | 4 | 2 | -1.57 | [-1.87, -1.26] | none |
| image | 43 | 10 | +0.01 | [-0.07, +0.09] | 0.25 |

Matched pairs in which both candidates are individually weaker than the base fusion: 39 of 65.

## R4. Dependence between pools (review point 4)

| subset | D all pairs | CI: by pool | CI: by encoder pair | CI: two-way | two-way df | leave-one-encoder-out range | every LOO CI contains 0 |
|---|---|---|---|---|---|---|---|
| all | -0.10 | [-0.31, +0.12] | [-0.24, +0.05] | [-0.31, +0.12] | df 17 | [-0.16, -0.03] | True |
| pathology | +0.02 | [-0.17, +0.22] | [-0.08, +0.12] | [-0.17, +0.22] | df 5 | [-0.07, +0.07] | True |
| vision | -0.02 | [-0.35, +0.30] | [-0.24, +0.19] | [-0.35, +0.30] | df 7 | [-0.10, +0.11] | True |
| text | -0.92 | [-1.79, -0.06] | [-2.16, +0.31] | [-2.45, +0.61] | df 3 | [-1.32, -0.62] | False |
| image | +0.00 | [-0.15, +0.15] | [-0.10, +0.10] | [-0.15, +0.15] | df 13 | [-0.05, +0.05] | True |

## R5. Selection regret with cross-fitted selection and hybrid rules (review point 1)

Label-using statistics for evaluation fold f are computed on the other folds' held-out rows only; the chosen candidate is scored on fold f. each fold's held-out logits come from a probe trained on the other folds, so selection statistics on folds != f still depend on fold-f labels through those probes. Removing that requires refitting probes in a nested design (GPU); this analysis removes the direct reuse.

Rules: **lowest_cka** = lowest CKA with the anchor (label-free; the registered heuristic); **competence** = highest accuracy; **unique** = largest unique contribution over the base; **disagreement** = highest disagreement with the base; **gated_diversity** = lowest CKA among candidates within 1.5 points of the most accurate; **rank_sum** = best sum of accuracy rank and (inverse) CKA rank; ties to accuracy; **random** = uniform random candidate (expected value).

| subset | pools | random | lowest_cka (cross-fit / in-sample) | competence (cross-fit / in-sample) | unique (cross-fit / in-sample) | disagreement (cross-fit / in-sample) | gated_diversity (cross-fit / in-sample) | rank_sum (cross-fit / in-sample) |
|---|---|---|---|---|---|---|---|---|
| all | 20 | 1.14 | 1.76 / 1.76 | 0.33 / 0.27 | 0.33 / 0.28 | 1.93 / 1.92 | 0.21 / 0.16 | 0.41 / 0.39 |
| pathology | 6 | 0.61 | 0.88 / 0.88 | 0.18 / 0.07 | 0.63 / 0.41 | 1.21 / 1.21 | 0.26 / 0.26 | 0.34 / 0.26 |
| vision | 8 | 1.72 | 2.66 / 2.66 | 0.29 / 0.32 | 0.15 / 0.22 | 2.71 / 2.72 | 0.22 / 0.14 | 0.72 / 0.72 |
| text | 6 | 0.90 | 1.42 / 1.42 | 0.53 / 0.41 | 0.26 / 0.25 | 1.62 / 1.58 | 0.14 / 0.08 | 0.08 / 0.08 |
| image | 14 | 1.25 | 1.90 / 1.90 | 0.24 / 0.21 | 0.36 / 0.30 | 2.07 / 2.07 | 0.24 / 0.19 | 0.56 / 0.52 |

Regret saved relative to the lowest-CKA rule (cross-fitted; positive = the rule beats lowest CKA):

| subset | rule | regret saved (pts) | p (paired t over pools) | better in | worse in |
|---|---|---|---|---|---|
| all | competence | +1.43 | 0.013 | 15 | 5 |
| all | unique | +1.43 | 0.017 | 14 | 4 |
| all | disagreement | -0.18 | 0.097 | 0 | 8 |
| all | gated_diversity | +1.55 | 0.012 | 13 | 3 |
| all | rank_sum | +1.34 | 0.011 | 14 | 2 |
| pathology | competence | +0.71 | 0.231 | 4 | 2 |
| pathology | unique | +0.25 | 0.599 | 2 | 3 |
| pathology | disagreement | -0.33 | 0.336 | 0 | 3 |
| pathology | gated_diversity | +0.62 | 0.229 | 2 | 1 |
| pathology | rank_sum | +0.54 | 0.288 | 2 | 2 |
| vision | competence | +2.37 | 0.067 | 7 | 1 |
| vision | unique | +2.51 | 0.077 | 7 | 1 |
| vision | disagreement | -0.05 | 0.144 | 0 | 3 |
| vision | gated_diversity | +2.45 | 0.088 | 6 | 1 |
| vision | rank_sum | +1.94 | 0.101 | 7 | 0 |
| text | competence | +0.90 | 0.294 | 4 | 2 |
| text | unique | +1.17 | 0.062 | 5 | 0 |
| text | disagreement | -0.20 | 0.258 | 0 | 2 |
| text | gated_diversity | +1.28 | 0.128 | 5 | 1 |
| text | rank_sum | +1.34 | 0.102 | 5 | 0 |
| image | competence | +1.66 | 0.030 | 11 | 3 |
| image | unique | +1.54 | 0.064 | 9 | 4 |
| image | disagreement | -0.17 | 0.229 | 0 | 6 |
| image | gated_diversity | +1.66 | 0.045 | 8 | 2 |
| image | rank_sum | +1.34 | 0.052 | 9 | 2 |

## R6. Algorithm 1 on all pools, cross-fitted (review point 1, Q3)

Accuracy of fusing the subset Algorithm 1 keeps, minus fusing the same number of members chosen by cross-fitted accuracy; and minus fusing all members. Primary fusion rule. Thresholds are a fixed grid, not tuned.

| θ_CKA | θ_unique (%) | pools | mean subset size | Alg 1 − top-k (pts) | p | better in | worse in | Alg 1 − all members |
|---|---|---|---|---|---|---|---|---|
| 0.6 | 0.5 | 20 | 4.0 | -1.41 | 0.015 | 3 | 15 | -1.27 |
| 0.6 | 1.0 | 20 | 3.5 | -1.72 | 0.006 | 3 | 17 | -1.57 |
| 0.6 | 2.0 | 20 | 3.1 | -2.17 | 0.002 | 3 | 17 | -1.87 |
| 0.7 | 0.5 | 20 | 4.8 | -0.88 | 0.002 | 3 | 15 | -0.71 |
| 0.7 | 1.0 | 20 | 4.5 | -1.06 | 6.1e-04 | 3 | 17 | -0.87 |
| 0.7 | 2.0 | 20 | 4.2 | -1.37 | 0.001 | 3 | 17 | -1.08 |
| 0.8 | 0.5 | 20 | 7.1 | -0.41 | 0.012 | 4 | 13 | -0.21 |
| 0.8 | 1.0 | 20 | 6.9 | -0.50 | 0.007 | 4 | 15 | -0.26 |
| 0.8 | 2.0 | 20 | 6.8 | -0.76 | 0.031 | 4 | 15 | -0.43 |
| 0.9 | 0.5 | 20 | 9.3 | -0.05 | 0.148 | 2 | 7 | -0.02 |
| 0.9 | 1.0 | 20 | 9.3 | -0.06 | 0.112 | 2 | 8 | -0.04 |
| 0.9 | 2.0 | 20 | 9.2 | -0.18 | 0.152 | 2 | 9 | -0.15 |

## R7. Other similarity measures on the real pools (review points 5, 6; Q4)

Candidate-level, within-pool-centred regression of utility on each measure and on accuracy (cluster-robust by pool); regret of choosing the least similar candidate by that measure; and the matched pairs re-oriented so that D = utility of the more similar minus the less similar candidate under that measure. `logit_*` measures use the held-out probe logits, a class-discriminative representation.

**all** (G = 20, candidates = 166)

| measure | coef (pts/unit) | p | accuracy coef | p | ρ with utility | regret: least similar | matched D by measure [95% CI] |
|---|---|---|---|---|---|---|---|
| linear_cka_registered | -1.40 | 0.083 | +0.150 | 0.007 | +0.44 | 1.76 | -0.10 [-0.31, +0.12] |
| linear_cka_base | -2.14 | 0.010 | +0.162 | 0.003 | +0.32 | 1.76 | -0.09 [-0.31, +0.12] |
| debiased_cka_anchor | -1.35 | 0.091 | +0.150 | 0.008 | +0.44 | 1.76 | -0.10 [-0.31, +0.12] |
| rbf_cka_anchor | -1.78 | 0.045 | +0.156 | 0.006 | +0.45 | 1.76 | -0.10 [-0.31, +0.12] |
| mutual_knn_anchor | -3.76 | 0.027 | +0.176 | 0.003 | +0.43 | 1.84 | -0.12 [-0.32, +0.09] |
| logit_cka_anchor | -0.62 | 0.807 | +0.134 | 0.052 | +0.48 | 1.86 | -0.08 [-0.30, +0.14] |
| logit_cka_base | -2.73 | 0.322 | +0.157 | 0.021 | +0.36 | 1.86 | -0.17 [-0.37, +0.03] |

**pathology** (G = 6, candidates = 60)

| measure | coef (pts/unit) | p | accuracy coef | p | ρ with utility | regret: least similar | matched D by measure [95% CI] |
|---|---|---|---|---|---|---|---|
| linear_cka_registered | -0.49 | 0.010 | +0.161 | 4.4e-05 | +0.38 | 0.88 | +0.02 [-0.17, +0.22] |
| linear_cka_base | -0.96 | 0.024 | +0.167 | 2.3e-05 | +0.23 | 0.88 | +0.02 [-0.17, +0.22] |
| debiased_cka_anchor | -0.49 | 0.011 | +0.161 | 4.5e-05 | +0.37 | 0.88 | +0.02 [-0.17, +0.22] |
| rbf_cka_anchor | -0.73 | 0.008 | +0.166 | 2.8e-05 | +0.40 | 0.89 | +0.02 [-0.17, +0.22] |
| mutual_knn_anchor | -2.00 | 0.068 | +0.190 | 6.8e-05 | +0.36 | 1.21 | -0.01 [-0.14, +0.11] |
| logit_cka_anchor | -1.93 | 0.027 | +0.195 | 9.2e-06 | +0.59 | 1.21 | +0.05 [-0.15, +0.25] |
| logit_cka_base | -3.28 | 0.063 | +0.223 | 5.6e-04 | +0.40 | 1.21 | -0.08 [-0.24, +0.08] |

**vision** (G = 8, candidates = 64)

| measure | coef (pts/unit) | p | accuracy coef | p | ρ with utility | regret: least similar | matched D by measure [95% CI] |
|---|---|---|---|---|---|---|---|
| linear_cka_registered | -1.28 | 0.282 | +0.147 | 0.073 | +0.64 | 2.66 | -0.02 [-0.35, +0.30] |
| linear_cka_base | -2.24 | 0.094 | +0.163 | 0.049 | +0.51 | 2.66 | -0.02 [-0.34, +0.31] |
| debiased_cka_anchor | -1.20 | 0.304 | +0.146 | 0.075 | +0.64 | 2.66 | -0.02 [-0.35, +0.30] |
| rbf_cka_anchor | -1.70 | 0.191 | +0.155 | 0.066 | +0.64 | 2.66 | -0.02 [-0.35, +0.30] |
| mutual_knn_anchor | -4.52 | 0.162 | +0.182 | 0.044 | +0.64 | 2.66 | -0.02 [-0.35, +0.30] |
| logit_cka_anchor | +1.23 | 0.765 | +0.114 | 0.210 | +0.56 | 2.67 | -0.02 [-0.35, +0.30] |
| logit_cka_base | -2.54 | 0.596 | +0.148 | 0.092 | +0.41 | 2.67 | -0.08 [-0.37, +0.22] |

**text** (G = 6, candidates = 42)

| measure | coef (pts/unit) | p | accuracy coef | p | ρ with utility | regret: least similar | matched D by measure [95% CI] |
|---|---|---|---|---|---|---|---|
| linear_cka_registered | -2.57 | 0.192 | +0.164 | 0.002 | +0.20 | 1.42 | -0.92 [-1.79, -0.06] |
| linear_cka_base | -3.26 | 0.086 | +0.173 | 7.1e-04 | +0.13 | 1.42 | -0.92 [-1.79, -0.06] |
| debiased_cka_anchor | -2.56 | 0.192 | +0.164 | 0.002 | +0.20 | 1.42 | -0.92 [-1.79, -0.06] |
| rbf_cka_anchor | -2.92 | 0.167 | +0.165 | 0.002 | +0.20 | 1.42 | -0.92 [-1.79, -0.06] |
| mutual_knn_anchor | -4.16 | 0.042 | +0.178 | 0.002 | +0.16 | 1.39 | -0.92 [-1.79, -0.06] |
| logit_cka_anchor | -5.11 | 0.118 | +0.216 | 0.005 | +0.29 | 1.42 | -0.92 [-1.79, -0.06] |
| logit_cka_base | -8.85 | 0.041 | +0.284 | 0.004 | +0.28 | 1.42 | -0.92 [-1.79, -0.06] |

**image** (G = 14, candidates = 124)

| measure | coef (pts/unit) | p | accuracy coef | p | ρ with utility | regret: least similar | matched D by measure [95% CI] |
|---|---|---|---|---|---|---|---|
| linear_cka_registered | -1.06 | 0.212 | +0.146 | 0.030 | +0.52 | 1.90 | +0.00 [-0.15, +0.15] |
| linear_cka_base | -1.90 | 0.038 | +0.160 | 0.017 | +0.39 | 1.90 | +0.01 [-0.14, +0.16] |
| debiased_cka_anchor | -1.00 | 0.231 | +0.145 | 0.031 | +0.52 | 1.90 | +0.00 [-0.15, +0.15] |
| rbf_cka_anchor | -1.48 | 0.140 | +0.153 | 0.028 | +0.53 | 1.90 | +0.00 [-0.15, +0.15] |
| mutual_knn_anchor | -3.56 | 0.126 | +0.175 | 0.021 | +0.51 | 2.04 | -0.02 [-0.15, +0.11] |
| logit_cka_anchor | +0.77 | 0.799 | +0.120 | 0.127 | +0.55 | 2.04 | +0.02 [-0.13, +0.17] |
| logit_cka_base | -2.07 | 0.520 | +0.149 | 0.050 | +0.39 | 2.04 | -0.08 [-0.21, +0.05] |

## R8. Other outcome metrics and per-pool intervals (review point 8)

D = metric gain of the higher-CKA minus the lower-CKA candidate; for NLL the gain is the reduction in NLL (positive = better), so all metrics share the sign convention.

| subset | metric | D all pairs | 95% CI | D primary | 95% CI |
|---|---|---|---|---|---|
| all | macro_f1 | -0.112 | [-0.34, +0.11] | -0.350 | [-0.85, +0.15] |
| all | nll | -0.005 | [-0.01, +0.00] | -0.015 | [-0.03, +0.00] |
| pathology | macro_f1 | +0.020 | [-0.16, +0.20] | -0.084 | [-0.58, +0.41] |
| pathology | nll | +0.004 | [-0.01, +0.01] | +0.006 | [-0.02, +0.03] |
| vision | macro_f1 | -0.048 | [-0.41, +0.32] | -0.109 | [-0.89, +0.67] |
| vision | nll | -0.008 | [-0.03, +0.01] | -0.011 | [-0.04, +0.02] |
| text | macro_f1 | -0.981 | [-1.90, -0.06] | -1.233 | [-3.45, +0.98] |
| text | nll | -0.037 | [-0.09, +0.01] | -0.053 | [-0.12, +0.01] |
| image | macro_f1 | -0.007 | [-0.16, +0.15] | -0.098 | [-0.53, +0.33] |
| image | nll | -0.001 | [-0.01, +0.01] | -0.003 | [-0.02, +0.01] |

Per-pool primary pair with a fold-level interval (5 folds, t(4); folds share training data, so these intervals are descriptive):

| pool | domain | pair (higher vs lower CKA) | D | fold 95% CI |
|---|---|---|---|---|
| bracs | pathology | hibou_b vs prov_gigapath | -0.65 | [-2.41, +1.12] |
| sicapv2 | pathology | kaiko_vitb16 vs h_optimus_0 | -0.65 | [-1.56, +0.26] |
| mhist | pathology | uni2_h vs phikon | +0.60 | [-0.19, +1.40] |
| crc_val_7k | pathology | h_optimus_0 vs virchow2 | -0.01 | [-0.05, +0.02] |
| nct_crc_100k | pathology | phikon vs virchow2 | +0.03 | [-0.02, +0.07] |
| pcam | pathology | uni vs virchow2 | +0.14 | [-0.46, +0.73] |
| dtd | vision | siglip_b16 vs convnext_l | +0.00 | [-1.34, +1.34] |
| aircraft | vision | dinov2_b vs openclip_h14 | -2.16 | [-2.88, -1.44] |
| eurosat | vision | eva02_clip vs vit_mae | +0.24 | [+0.04, +0.44] |
| pets | vision | dinov2_b vs openclip_h14 | -0.05 | [-0.57, +0.46] |
| food101 | vision | openclip_h14 vs dinov2_l | +0.60 | [+0.19, +1.00] |
| cifar100 | vision | openclip_h14 vs convnext_l | +0.68 | [+0.31, +1.05] |
| gtsrb | vision | siglip_b16 vs vit_mae | -0.37 | [-0.50, -0.23] |
| country211 | vision | clip_b32 vs dinov2_l | +0.44 | [-0.37, +1.24] |
| emotion | text | gte_base vs glove_avg | -3.05 | [-3.82, -2.28] |
| trec_coarse | text | gte_base vs minilm_l6 | -0.33 | [-0.95, +0.29] |
| newsgroups20 | text | bge_base vs mpnet_base | -1.62 | [-2.79, -0.44] |
| sst5 | text | minilm_l3 vs glove_avg | +0.09 | [-0.99, +1.18] |

## R9. Matched versus unmatched comparisons across all pools (review point 9)

| subset | matched D [95% CI] | unmatched pairs | unmatched D [95% CI] | unmatched: higher-CKA more accurate | D per signed-gap point | p | D at zero gap | p |
|---|---|---|---|---|---|---|---|---|
| all | -0.10 [-0.31, +0.12] | 198 | +1.23 [+0.40, +2.05] | 95% | +0.120 | 0.040 | -0.06 | 0.878 |
| pathology | +0.02 [-0.17, +0.22] | 49 | +0.72 [+0.07, +1.37] | 90% | +0.175 | 1.1e-05 | -0.28 | 0.068 |
| vision | -0.02 [-0.35, +0.30] | 97 | +1.74 [+0.04, +3.43] | 98% | +0.109 | 0.152 | +0.15 | 0.805 |
| text | -0.92 [-1.79, -0.06] | 52 | +0.75 [-0.53, +2.03] | 96% | +0.139 | 0.003 | -0.40 | 0.475 |
| image | +0.00 [-0.15, +0.15] | 146 | +1.40 [+0.31, +2.49] | 95% | +0.114 | 0.089 | +0.07 | 0.864 |
