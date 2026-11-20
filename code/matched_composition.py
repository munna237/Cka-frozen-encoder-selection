"""Composition of the 65 competence-matched pairs relative to the base fusion
(EXPLORATORY). Reads only the per-pool records in results/pools/.

For each matched pair, each candidate's individual accuracy is compared with the
held-out accuracy of the base fusion it would be added to (primary rule, mean over
folds). Reports, overall and by domain: pairs in which both candidates, one, or
neither is weaker than the base; the median and interquartile range of the pair's
mean accuracy minus the base; and how far the stronger member is below the anchor.

    python code/matched_composition.py  ->  results/reviewer/matched_composition.json
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


def main():
    recs = R.load(os.path.join(REPO, "results"))
    rows = []
    for p in R._pairs(recs):
        r = next(r for r in recs if r["pool"] == p["pool"])
        base = float(np.mean(r["base_fold_acc"]["mean_logits"]))
        hi, lo = p["acc_high"] - base, p["acc_low"] - base
        rows.append({"pool": p["pool"], "domain": r["domain"], "high_minus_base": hi, "low_minus_base": lo,
                     "stronger_below_anchor": p["anchor_acc"] - max(p["acc_high"], p["acc_low"])})
    out = {"note": R.EXPLORATORY, "pairs": len(rows), "subsets": {}}
    for s in ("all", "pathology", "vision", "text"):
        rr = [x for x in rows if s == "all" or x["domain"] == s]
        m = np.array([(x["high_minus_base"] + x["low_minus_base"]) / 2 for x in rr])
        out["subsets"][s] = {
            "pairs": len(rr),
            "both_weaker_than_base": sum(max(x["high_minus_base"], x["low_minus_base"]) < 0 for x in rr),
            "one_weaker": sum(min(x["high_minus_base"], x["low_minus_base"]) < 0 <= max(x["high_minus_base"], x["low_minus_base"]) for x in rr),
            "neither_weaker": sum(min(x["high_minus_base"], x["low_minus_base"]) >= 0 for x in rr),
            "pair_mean_minus_base_median": float(np.median(m)),
            "pair_mean_minus_base_iqr": [float(v) for v in np.percentile(m, [25, 75])],
            "stronger_member_below_anchor_median": float(np.median([x["stronger_below_anchor"] for x in rr])),
        }
    # consistency with the figure already in the paper (39 of 65 both weaker; reviewer.r3_strong)
    assert out["subsets"]["all"]["both_weaker_than_base"] == 39 and out["pairs"] == 65
    with open(os.path.join(REPO, "results", "reviewer", "matched_composition.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["subsets"], indent=1))


if __name__ == "__main__":
    main()
