# Secondary fusion rules and pool weighting (EXPLORATORY)

| rule | D primary [95% CI] (pairs) | D all pairs [95% CI] (pairs) | TOST tightest | regret lowest CKA | regret competence |
|---|---|---|---|---|---|
| mean_logits | -0.34 [-0.84, +0.16] (18) | -0.10 [-0.31, +0.12] (65) | 0.5 | 1.76 | 0.27 |
| mean_probs | -0.28 [-0.66, +0.11] (18) | -0.14 [-0.31, +0.04] (65) | 0.5 | 1.37 | 0.24 |
| class_conditional | -0.55 [-1.11, +0.00] (18) | -0.24 [-0.50, +0.03] (65) | 0.5 | 0.87 | 0.39 |

Equal pool weight (G = 18): -0.21 [-0.51, +0.09], TOST tightest 0.5.
Leave one pool out: estimate -0.14 to -0.03; widest interval [-0.39, +0.15]; equivalent within ±0.5 in every case: True.
