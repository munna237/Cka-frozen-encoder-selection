"""Supplementary table of the 20-pool UNMATCHED comparison (EXPLORATORY).

The unmatched pairs are the pairs of the registered pair construction whose CKA gap
qualifies (>= 0.15 and beyond the subsampling tolerance) but whose competence gap
exceeds the 1.5-point caliper (category "unmatched" in the per-pool records). This
script reads only the per-pool records in results/pools/, so it runs on a CPU from
the repository alone.

It first reproduces the subset-level figures already in
results/reviewer/reviewer_analyses.json (R9_unmatched) and asserts they agree, then
adds what the paper's supplementary table needs: per-pool counts, the share of pairs
in which the higher-CKA member is the more accurate, mean signed competence gap and
mean D, and, per subset, the gap-adjusted D (the intercept of D regressed on the
signed gap, i.e. the difference expected at equal competence) with its 95% interval.

    python code/unmatched_pairs.py   ->  results/reviewer/unmatched_pairs.{json,md}
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("V3_DRIVE", REPO)

import reviewer as R  # noqa: E402
from inference import cluster_robust_ols  # noqa: E402

RESULTS = os.path.join(REPO, "results")


def main():
    recs = R.load(RESULTS)
    out = {"note": R.EXPLORATORY, "definition": "pairs of category 'unmatched' in results/pools/*.json: CKA gap "
           ">= 0.15 and beyond 3 median subsampling SDs, competence gap > 1.5 points", "per_pool": [], "subsets": {}}

    stored = json.load(open(os.path.join(RESULTS, "reviewer", "reviewer_analyses.json")))["R9_unmatched"]
    for s in R.SUBSETS:
        U = R._pairs(R._subset(recs, s), cats=("unmatched",))
        if not U:
            continue
        d = np.array([p["diff"] for p in U]); g = np.array([p["signed_gap"] for p in U])
        cl = np.array([p["pool"] for p in U])
        mean = cluster_robust_ols(d, cl)
        reg = cluster_robust_ols(d, cl, x=g)
        rec = {"pairs": int(len(U)), "pools": int(len(set(cl))),
               "frac_higher_cka_more_accurate": float((g > 0).mean()),
               "mean_signed_gap": float(g.mean()),
               "D": {"estimate": float(mean["estimate"][0]), "ci": [float(mean["ci_low"][0]), float(mean["ci_high"][0])],
                     "p": float(mean["p"][0])},
               "D_at_zero_gap": {"estimate": float(reg["estimate"][0]),
                                 "ci": [float(reg["ci_low"][0]), float(reg["ci_high"][0])], "p": float(reg["p"][0])},
               "slope_per_gap_point": {"estimate": float(reg["estimate"][1]),
                                       "ci": [float(reg["ci_low"][1]), float(reg["ci_high"][1])], "p": float(reg["p"][1])}}
        # agreement with the stored R9 figures
        st = stored[s]
        assert st["unmatched_pairs"] == rec["pairs"]
        assert abs(st["unmatched_all_pairs"]["estimate"] - rec["D"]["estimate"]) < 1e-9
        assert abs(st["unmatched_frac_higher_cka_more_accurate"] - rec["frac_higher_cka_more_accurate"]) < 1e-12
        assert abs(st["unmatched_D_on_signed_gap"]["intercept"] - rec["D_at_zero_gap"]["estimate"]) < 1e-9
        assert abs(st["unmatched_D_on_signed_gap"]["slope_pp_per_point"] - rec["slope_per_gap_point"]["estimate"]) < 1e-9
        out["subsets"][s] = rec

    for r in recs:
        U = R._pairs([r], cats=("unmatched",))
        row = {"pool": r["pool"], "domain": r["domain"], "pairs": len(U)}
        if U:
            g = np.array([p["signed_gap"] for p in U]); d = np.array([p["diff"] for p in U])
            row.update({"frac_higher_cka_more_accurate": float((g > 0).mean()), "mean_signed_gap": float(g.mean()),
                        "mean_D": float(d.mean())})
        out["per_pool"].append(row)
    assert sum(r["pairs"] for r in out["per_pool"]) == out["subsets"]["all"]["pairs"]

    with open(os.path.join(RESULTS, "reviewer", "unmatched_pairs.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    f = lambda v: f"{v:+.2f}"  # noqa: E731
    L = ["# Unmatched pairs across the 20 pools (EXPLORATORY)", "", out["definition"] + ".", "",
         "| subset | pairs | pools | higher-CKA more accurate | mean signed gap | D [95% CI] | D at zero gap [95% CI] | p |",
         "|---|---|---|---|---|---|---|---|"]
    for s, v in out["subsets"].items():
        L.append(f"| {s} | {v['pairs']} | {v['pools']} | {v['frac_higher_cka_more_accurate']:.0%} | "
                 f"{f(v['mean_signed_gap'])} | {f(v['D']['estimate'])} [{f(v['D']['ci'][0])}, {f(v['D']['ci'][1])}] | "
                 f"{f(v['D_at_zero_gap']['estimate'])} [{f(v['D_at_zero_gap']['ci'][0])}, "
                 f"{f(v['D_at_zero_gap']['ci'][1])}] | {v['D_at_zero_gap']['p']:.2f} |")
    L += ["", "| pool | domain | pairs | higher-CKA more accurate | mean signed gap | mean D |", "|---|---|---|---|---|---|"]
    for r in out["per_pool"]:
        if r["pairs"]:
            L.append(f"| {r['pool']} | {r['domain']} | {r['pairs']} | {r['frac_higher_cka_more_accurate']:.0%} | "
                     f"{f(r['mean_signed_gap'])} | {f(r['mean_D'])} |")
        else:
            L.append(f"| {r['pool']} | {r['domain']} | 0 | — | — | — |")
    open(os.path.join(RESULTS, "reviewer", "unmatched_pairs.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
