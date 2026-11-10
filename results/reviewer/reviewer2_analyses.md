# Second-round reviewer analyses (EXPLORATORY)

Post-registration. The code first re-asserts that the saved logits reproduce every registered utility.

## S1. Without the six pilot datasets (Q1)

Excluded: bracs, sicapv2, mhist, dtd, aircraft, eurosat.

| subset | pools | D primary [95% CI] | pairs | D all pairs [95% CI] | TOST | regret lowest CKA | regret competence | regret unique |
|---|---|---|---|---|---|---|---|---|
| all | 14 | -0.29 [-0.96, +0.38] | 45 | -0.12 [-0.38, +0.14] | 0.5 | 1.46 | 0.20 | 0.28 |
| pathology | 3 | +0.05 [-0.14, +0.24] | 26 | +0.01 [-0.04, +0.06] | 0.25 | 0.14 | 0.01 | 0.55 |
| vision | 5 | +0.26 [-0.30, +0.82] | 12 | +0.07 [-0.36, +0.49] | 0.5 | 2.29 | 0.07 | 0.17 |
| text | 6 | -1.23 [-3.48, +1.03] | 7 | -0.92 [-1.79, -0.06] | none | 1.42 | 0.41 | 0.25 |
| image | 8 | +0.18 [-0.12, +0.48] | 38 | +0.03 [-0.08, +0.14] | 0.25 | 1.48 | 0.05 | 0.31 |

## S2. Hybrid rules: all rules tried, gate width, paired tests (concern 3, Q2)

Rules tried in the first exploratory round (complete list, fixed in code before it ran): lowest_cka, competence, unique, disagreement, gated_diversity (gate 1.5), rank_sum.

Cross-fitted regret of the gated rule by gate width (points). Per-fold selection can beat the mean-based oracle on a fold, so regret can be slightly negative.

| subset | gate 0.5 | gate 1.0 | gate 1.5 | gate 3.0 |
|---|---|---|---|---|
| all | 0.18 | 0.22 | 0.21 | 0.34 |
| pathology | 0.10 | 0.18 | 0.26 | 0.43 |
| vision | 0.16 | 0.23 | 0.22 | 0.30 |
| text | 0.29 | 0.25 | 0.14 | 0.31 |
| image | 0.14 | 0.21 | 0.24 | 0.36 |

Paired, cross-fitted: regret of competence minus regret of the rule (positive = rule better):

| subset | rule | mean | 95% CI | p (t) | p (Wilcoxon) | better | worse | tie |
|---|---|---|---|---|---|---|---|---|
| all | gated_diversity | +0.12 | [-0.13, +0.36] | 0.325 | 0.554 | 10 | 7 | 3 |
| all | rank_sum | -0.09 | [-0.71, +0.54] | 0.777 | 0.981 | 7 | 9 | 4 |
| all | unique | +0.00 | [-0.34, +0.34] | 0.999 | 0.959 | 8 | 8 | 4 |
| pathology | gated_diversity | -0.09 | [-0.35, +0.17] | 0.422 | 1.000 | 3 | 3 | 0 |
| pathology | rank_sum | -0.16 | [-0.36, +0.04] | 0.090 | 0.031 | 0 | 6 | 0 |
| pathology | unique | -0.46 | [-1.19, +0.27] | 0.165 | 0.188 | 1 | 4 | 1 |
| vision | gated_diversity | +0.07 | [-0.44, +0.59] | 0.748 | 0.844 | 3 | 3 | 2 |
| vision | rank_sum | -0.43 | [-2.13, +1.27] | 0.569 | 1.000 | 3 | 2 | 3 |
| vision | unique | +0.14 | [-0.28, +0.57] | 0.455 | 1.000 | 3 | 3 | 2 |
| text | gated_diversity | +0.38 | [-0.19, +0.96] | 0.148 | 0.188 | 4 | 1 | 1 |
| text | rank_sum | +0.45 | [-0.19, +1.09] | 0.130 | 0.188 | 4 | 1 | 1 |
| text | unique | +0.27 | [-0.70, +1.24] | 0.504 | 0.438 | 4 | 1 | 1 |
| image | gated_diversity | +0.00 | [-0.28, +0.28] | 0.979 | 0.638 | 6 | 6 | 2 |
| image | rank_sum | -0.31 | [-1.18, +0.55] | 0.448 | 0.239 | 3 | 8 | 3 |
| image | unique | -0.12 | [-0.49, +0.26] | 0.515 | 0.374 | 4 | 7 | 3 |

## S3. Fully nested selection (concern 4, Q2)

Selection statistics from probes refitted by inner 4-fold CV on each outer training split; no label of the evaluation fold is used in selection. Pools with n ≤ 12500: bracs, sicapv2, mhist, crc_val_7k, dtd, aircraft, pets, banking77, trec_coarse, newsgroups20, sst5.

| subset | pools | lowest CKA | competence | unique | disagreement | gated 1.5 | rank sum | gated vs competence | competence vs lowest CKA |
|---|---|---|---|---|---|---|---|---|---|
| all | 11 | 2.11 | 0.20 | 0.50 | 2.17 | 0.20 | 0.23 | +0.01 [-0.20, +0.21] (p 0.957) | +1.90 [+0.09, +3.72] (p 0.041) |
| pathology | 4 | 1.22 | 0.26 | 0.84 | 1.22 | 0.37 | 0.41 | -0.11 [-0.44, +0.22] (p 0.375) | +0.96 [-1.32, +3.24] (p 0.272) |
| vision | 3 | 3.42 | 0.04 | 0.29 | 3.32 | 0.17 | 0.19 | -0.13 [-0.88, +0.62] (p 0.528) | +3.38 [-8.47, +15.23] (p 0.345) |
| text | 4 | 2.02 | 0.28 | 0.33 | 2.26 | 0.06 | 0.09 | +0.22 [-0.32, +0.76] (p 0.283) | +1.74 [-1.29, +4.77] (p 0.165) |
| image | 7 | 2.16 | 0.16 | 0.60 | 2.12 | 0.28 | 0.32 | -0.12 [-0.33, +0.09] (p 0.220) | +2.00 [-0.97, +4.96] (p 0.151) |

## S4. Probe regularisation C (concern 9)

Same folds and matched pairs; C = 1 is registered (asserted to reproduce the registered utilities).

| pool | C | mean candidate accuracy | matched pairs | mean D | regret lowest CKA | regret competence |
|---|---|---|---|---|---|---|
| mhist | 0.1 | 85.38 | 6 | +0.59 | 0.32 | 0.03 |
| mhist | 1.0 | 83.94 | 6 | +0.36 | 0.00 | 0.38 |
| mhist | 10.0 | 82.98 | 6 | +0.55 | 0.19 | 0.35 |
| dtd | 0.1 | 76.26 | 2 | +0.00 | 1.17 | 0.00 |
| dtd | 1.0 | 76.10 | 2 | -0.03 | 0.74 | 0.00 |
| dtd | 10.0 | 75.70 | 2 | -0.11 | 0.59 | 0.11 |
| aircraft | 0.1 | 54.16 | 2 | -1.38 | 9.66 | 0.00 |
| aircraft | 1.0 | 54.22 | 2 | -1.17 | 8.88 | 2.16 |
| aircraft | 10.0 | 54.13 | 2 | -1.27 | 7.74 | 1.77 |

## S5. Is the weak association consistent across pools? (Q4)

| measure | pools with negative slope | LOO coefficient range | LOO p range | pool whose removal weakens it most |
|---|---|---|---|---|
| linear_cka_registered | 16 of 20 | [-1.67, -0.75] | [0.041, 0.202] | aircraft |
| linear_cka_base | 17 of 20 | [-2.36, -1.41] | [0.005, 0.027] | aircraft |
| mutual_knn_anchor | 15 of 20 | [-4.28, -1.82] | [0.002, 0.135] | aircraft |
| rbf_cka_anchor | 16 of 20 | [-2.08, -1.01] | [0.019, 0.110] | aircraft |

## S6. Multiplicity (concern 2)

EXPLORATORY (post-registration; not part of the registered analysis); adjusted p-values are descriptive.

**table4_all_pools** (smallest BH q = 0.073)

| test | p | q (BH) | p (Holm) |
|---|---|---|---|
| linear_cka_registered | 0.083 | 0.128 | 0.331 |
| linear_cka_base | 0.010 | 0.073 | 0.073 |
| debiased_cka_anchor | 0.091 | 0.128 | 0.331 |
| rbf_cka_anchor | 0.045 | 0.106 | 0.227 |
| mutual_knn_anchor | 0.027 | 0.095 | 0.163 |
| logit_cka_anchor | 0.807 | 0.807 | 0.807 |
| logit_cka_base | 0.322 | 0.376 | 0.645 |

**R7_all_subsets** (smallest BH q = 0.093)

| test | p | q (BH) | p (Holm) |
|---|---|---|---|
| all/linear_cka_registered | 0.083 | 0.193 | 1.000 |
| all/linear_cka_base | 0.010 | 0.093 | 0.345 |
| all/debiased_cka_anchor | 0.091 | 0.193 | 1.000 |
| all/rbf_cka_anchor | 0.045 | 0.144 | 1.000 |
| all/mutual_knn_anchor | 0.027 | 0.136 | 0.807 |
| all/logit_cka_anchor | 0.807 | 0.807 | 1.000 |
| all/logit_cka_base | 0.322 | 0.376 | 1.000 |
| pathology/linear_cka_registered | 0.010 | 0.093 | 0.339 |
| pathology/linear_cka_base | 0.024 | 0.136 | 0.756 |
| pathology/debiased_cka_anchor | 0.011 | 0.093 | 0.345 |
| pathology/rbf_cka_anchor | 0.008 | 0.093 | 0.279 |
| pathology/mutual_knn_anchor | 0.068 | 0.183 | 1.000 |
| pathology/logit_cka_anchor | 0.027 | 0.136 | 0.807 |
| pathology/logit_cka_base | 0.063 | 0.183 | 1.000 |
| vision/linear_cka_registered | 0.282 | 0.353 | 1.000 |
| vision/linear_cka_base | 0.094 | 0.193 | 1.000 |
| vision/debiased_cka_anchor | 0.304 | 0.367 | 1.000 |
| vision/rbf_cka_anchor | 0.191 | 0.269 | 1.000 |
| vision/mutual_knn_anchor | 0.162 | 0.265 | 1.000 |
| vision/logit_cka_anchor | 0.765 | 0.807 | 1.000 |
| vision/logit_cka_base | 0.596 | 0.652 | 1.000 |
| text/linear_cka_registered | 0.192 | 0.269 | 1.000 |
| text/linear_cka_base | 0.086 | 0.193 | 1.000 |
| text/debiased_cka_anchor | 0.192 | 0.269 | 1.000 |
| text/rbf_cka_anchor | 0.167 | 0.265 | 1.000 |
| text/mutual_knn_anchor | 0.042 | 0.144 | 1.000 |
| text/logit_cka_anchor | 0.118 | 0.229 | 1.000 |
| text/logit_cka_base | 0.041 | 0.144 | 1.000 |
| image/linear_cka_registered | 0.212 | 0.286 | 1.000 |
| image/linear_cka_base | 0.038 | 0.144 | 1.000 |
| image/debiased_cka_anchor | 0.231 | 0.300 | 1.000 |
| image/rbf_cka_anchor | 0.140 | 0.245 | 1.000 |
| image/mutual_knn_anchor | 0.126 | 0.232 | 1.000 |
| image/logit_cka_anchor | 0.799 | 0.807 | 1.000 |
| image/logit_cka_base | 0.520 | 0.587 | 1.000 |

**D_robustness** (smallest BH q = 0.341)

| test | p | q (BH) | p (Holm) |
|---|---|---|---|
| R1 gap-adjusted D/all | 0.226 | 0.883 | 1.000 |
| R2 caliper 0.5/all | 0.363 | 0.883 | 1.000 |
| R2 caliper 1.0/all | 0.378 | 0.883 | 1.000 |
| R2 caliper 1.5/all | 0.357 | 0.883 | 1.000 |
| R2 caliper 3.0/all | 0.372 | 0.883 | 1.000 |
| R3 strong/all | 0.334 | 0.883 | 1.000 |
| R4 two-way/all | 0.357 | 0.883 | 1.000 |
| R1 gap-adjusted D/pathology | 0.852 | 0.916 | 1.000 |
| R2 caliper 0.5/pathology | 0.305 | 0.883 | 1.000 |
| R2 caliper 1.0/pathology | 0.826 | 0.916 | 1.000 |
| R2 caliper 1.5/pathology | 0.785 | 0.916 | 1.000 |
| R2 caliper 3.0/pathology | 0.302 | 0.883 | 1.000 |
| R3 strong/pathology | 0.753 | 0.916 | 1.000 |
| R4 two-way/pathology | 0.786 | 0.916 | 1.000 |
| R1 gap-adjusted D/vision | 0.575 | 0.916 | 1.000 |
| R2 caliper 0.5/vision | 0.819 | 0.916 | 1.000 |
| R2 caliper 1.0/vision | 0.573 | 0.916 | 1.000 |
| R2 caliper 1.5/vision | 0.864 | 0.916 | 1.000 |
| R2 caliper 3.0/vision | 0.803 | 0.916 | 1.000 |
| R3 strong/vision | 0.525 | 0.916 | 1.000 |
| R4 two-way/vision | 0.864 | 0.916 | 1.000 |
| R1 gap-adjusted D/text | 0.053 | 0.624 | 1.000 |
| R2 caliper 0.5/text | 0.537 | 0.916 | 1.000 |
| R2 caliper 1.0/text | 0.145 | 0.879 | 1.000 |
| R2 caliper 1.5/text | 0.042 | 0.624 | 1.000 |
| R2 caliper 3.0/text | 0.085 | 0.746 | 1.000 |
| R3 strong/text | 0.010 | 0.341 | 0.341 |
| R4 two-way/text | 0.151 | 0.879 | 1.000 |
| R1 gap-adjusted D/image | 0.588 | 0.916 | 1.000 |
| R2 caliper 0.5/image | 0.693 | 0.916 | 1.000 |
| R2 caliper 1.0/image | 0.765 | 0.916 | 1.000 |
| R2 caliper 1.5/image | 0.962 | 0.962 | 1.000 |
| R2 caliper 3.0/image | 0.748 | 0.916 | 1.000 |
| R3 strong/image | 0.721 | 0.916 | 1.000 |
| R4 two-way/image | 0.962 | 0.962 | 1.000 |

## S7. Primary pairs: signed competence gap and fold interval (minor point)

| pool | higher vs lower CKA | CKA gap | signed gap (pts) | D | fold 95% CI |
|---|---|---|---|---|---|
| bracs | hibou_b vs prov_gigapath | 0.18 | +0.50 | -0.65 | [-2.41, +1.12] |
| sicapv2 | kaiko_vitb16 vs h_optimus_0 | 0.17 | -0.57 | -0.65 | [-1.56, +0.26] |
| mhist | uni2_h vs phikon | 0.25 | +0.76 | +0.60 | [-0.19, +1.40] |
| crc_val_7k | h_optimus_0 vs virchow2 | 0.25 | +0.06 | -0.01 | [-0.05, +0.02] |
| nct_crc_100k | phikon vs virchow2 | 0.47 | +0.01 | +0.03 | [-0.02, +0.07] |
| pcam | uni vs virchow2 | 0.39 | +0.68 | +0.14 | [-0.46, +0.73] |
| dtd | siglip_b16 vs convnext_l | 0.17 | +0.90 | +0.00 | [-1.34, +1.34] |
| aircraft | dinov2_b vs openclip_h14 | 0.45 | +0.72 | -2.16 | [-2.88, -1.44] |
| eurosat | eva02_clip vs vit_mae | 0.61 | +1.06 | +0.24 | [+0.04, +0.44] |
| pets | dinov2_b vs openclip_h14 | 0.28 | +0.52 | -0.05 | [-0.57, +0.46] |
| food101 | openclip_h14 vs dinov2_l | 0.16 | +1.02 | +0.60 | [+0.19, +1.00] |
| cifar100 | openclip_h14 vs convnext_l | 0.21 | +1.03 | +0.68 | [+0.31, +1.05] |
| gtsrb | siglip_b16 vs vit_mae | 0.55 | +1.38 | -0.37 | [-0.50, -0.23] |
| country211 | clip_b32 vs dinov2_l | 0.26 | +0.46 | +0.44 | [-0.37, +1.24] |
| emotion | gte_base vs glove_avg | 0.61 | +1.34 | -3.05 | [-3.82, -2.28] |
| trec_coarse | gte_base vs minilm_l6 | 0.15 | -0.95 | -0.33 | [-0.95, +0.29] |
| newsgroups20 | bge_base vs mpnet_base | 0.23 | +0.25 | -1.62 | [-2.79, -0.44] |
| sst5 | minilm_l3 vs glove_avg | 0.20 | -0.11 | +0.09 | [-0.99, +1.18] |

## S8. Encoder validation (concern 9)

| encoder | official source | tensors compared | max abs difference (port vs official) |
|---|---|---|---|
| lunit_vits16 | https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/dino_vit_small_patch16_ep200.torch | 150 / 150 | 0.0 |
| kaiko_vitb16 | https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/0.0.1/vitb16.pth | 150 / 150 | 0.0 |

| comparison | images | CKA (ours vs official) | mean cosine | error |
|---|---|---|---|---|
| dinov2_b_vs_official | 256 | 0.99940 | 0.99744 |  |
| dinov2_l_vs_official | 256 | 0.99961 | 0.99817 |  |
