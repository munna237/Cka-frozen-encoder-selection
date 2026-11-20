"""Secondary fusion rules and pool weighting (EXPLORATORY; post-registration).

Reads only the per-pool records in results/pools/, so it runs on a CPU from the
repository alone. Two analyses:

1. The registered estimands (A) and (C) and the regret of lowest-CKA and
   competence selection under each of the three registered fusion rules: the
   primary rule (mean_logits) and the two secondary rules (mean_probs,
   class_conditional). The primary-rule figures are first asserted equal to the
   registered ones in results/battery.json.
2. Pool weighting of the all-pairs estimate: (a) every pool weighted equally, i.e.
   the mean over pools of each pool's mean D with a t(G-1) interval; (b) leave one
   pool out, re-running the cluster-robust estimate and the registered TOST at each
   margin.

    python code/secondary_and_weighting.py  ->  results/reviewer/secondary_and_weighting.{json,md}
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("V3_DRIVE", REPO)

import reviewer as R  # noqa: E402
from inference import cluster_robust_ols, tost_equivalence  # noqa: E402

RULES = ("mean_logits", "mean_probs", "class_conditional")
MARGINS = (1.0, 0.5, 0.25)


def tightest(d, cl):
    ok = [m for m in MARGINS if tost_equivalence(np.asarray(d), np.asarray(cl), margin=m)["equivalent"]]
    return min(ok) if ok else None


def cr(d, cl):
    r = cluster_robust_ols(np.asarray(d, float), np.asarray(cl))
    return {"estimate": float(r["estimate"][0]), "ci": [float(r["ci_low"][0]), float(r["ci_high"][0])],
            "p": float(r["p"][0]), "n": len(d), "G": int(len(set(cl)))}


def regret(recs, rule):
    out = {"lowest_cka": [], "competence": []}
    for r in recs:
        cands = list(r["utility_per_fold"][rule].keys())
        u = {c: float(np.mean(r["utility_per_fold"][rule][c])) for c in cands}
        best = max(u.values())
        low = min(cands, key=lambda c: r["cka_with_anchor"][c])
        comp = max(cands, key=lambda c: r["indiv_acc"][c])
        out["lowest_cka"].append(best - u[low]); out["competence"].append(best - u[comp])
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    recs = R.load(os.path.join(REPO, "results"))
    bat = json.load(open(os.path.join(REPO, "results", "battery.json")))["all"]
    out = {"note": R.EXPLORATORY, "rules": {}, "pool_weighting": {}}

    for rule in RULES:
        P = [p for r in recs for p in r["pairs"] if p["category"] == "matched" and np.isfinite(p[f"diff_{rule}"])]
        d = [p[f"diff_{rule}"] for p in P]; cl = [p["pool"] for p in P]
        prim = [p for p in P if p.get("primary")]
        out["rules"][rule] = {"all_pairs": cr(d, cl), "tost_tightest": tightest(d, cl),
                              "primary": cr([p[f"diff_{rule}"] for p in prim], [p["pool"] for p in prim]),
                              "regret": regret(recs, rule)}

    # the primary rule must reproduce the registered figures
    m = out["rules"]["mean_logits"]
    assert abs(m["all_pairs"]["estimate"] - bat["A_matched_null_all_pairs"]["cluster_robust"]["estimate_pp"]) < 1e-9
    assert abs(m["primary"]["estimate"] - bat["A_matched_null_primary_only"]["cluster_robust"]["estimate_pp"]) < 1e-9
    assert m["all_pairs"]["n"] == 65 and m["primary"]["n"] == 18 and m["tost_tightest"] == 0.5
    assert abs(m["regret"]["lowest_cka"] - 1.7565188699644936) < 1e-6, m["regret"]

    # pool weighting, primary rule
    P = [p for r in recs for p in r["pairs"] if p["category"] == "matched" and np.isfinite(p["diff"])]
    pools = sorted(set(p["pool"] for p in P))
    pm = np.array([np.mean([p["diff"] for p in P if p["pool"] == q]) for q in pools])
    G = len(pm); se = pm.std(ddof=1) / np.sqrt(G); tc = stats.t.ppf(0.975, G - 1); t90 = stats.t.ppf(0.95, G - 1)
    eq = {"estimate": float(pm.mean()), "ci": [float(pm.mean() - tc * se), float(pm.mean() + tc * se)],
          "ci90": [float(pm.mean() - t90 * se), float(pm.mean() + t90 * se)], "G": G}
    eq["tost_tightest"] = next((mg for mg in sorted(MARGINS) if -mg < eq["ci90"][0] and eq["ci90"][1] < mg), None)
    loo = []
    for q in pools:
        Q = [p for p in P if p["pool"] != q]
        d = [p["diff"] for p in Q]; cl = [p["pool"] for p in Q]
        loo.append({"dropped": q, **cr(d, cl), "tost_tightest": tightest(d, cl)})
    out["pool_weighting"] = {
        "equal_pool_weight": eq,
        "leave_one_pool_out": {"estimate_range": [min(x["estimate"] for x in loo), max(x["estimate"] for x in loo)],
                               "ci_low_min": min(x["ci"][0] for x in loo), "ci_high_max": max(x["ci"][1] for x in loo),
                               "all_equivalent_at_0.5": all(x["tost_tightest"] is not None and x["tost_tightest"] <= 0.5 for x in loo),
                               "tightest_by_dropped_pool": {x["dropped"]: x["tost_tightest"] for x in loo},
                               "per_pool": loo}}
    with open(os.path.join(REPO, "results", "reviewer", "secondary_and_weighting.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    f = lambda v: f"{v:+.2f}"  # noqa: E731
    L = ["# Secondary fusion rules and pool weighting (EXPLORATORY)", "",
         "| rule | D primary [95% CI] (pairs) | D all pairs [95% CI] (pairs) | TOST tightest | regret lowest CKA | regret competence |",
         "|---|---|---|---|---|---|"]
    for rule, v in out["rules"].items():
        a, pr = v["all_pairs"], v["primary"]
        L.append(f"| {rule} | {f(pr['estimate'])} [{f(pr['ci'][0])}, {f(pr['ci'][1])}] ({pr['n']}) | "
                 f"{f(a['estimate'])} [{f(a['ci'][0])}, {f(a['ci'][1])}] ({a['n']}) | {v['tost_tightest'] or 'none'} | "
                 f"{v['regret']['lowest_cka']:.2f} | {v['regret']['competence']:.2f} |")
    e = out["pool_weighting"]["equal_pool_weight"]; lo = out["pool_weighting"]["leave_one_pool_out"]
    L += ["", f"Equal pool weight (G = {e['G']}): {f(e['estimate'])} [{f(e['ci'][0])}, {f(e['ci'][1])}], "
          f"TOST tightest {e['tost_tightest']}.",
          f"Leave one pool out: estimate {f(lo['estimate_range'][0])} to {f(lo['estimate_range'][1])}; widest interval "
          f"[{f(lo['ci_low_min'])}, {f(lo['ci_high_max'])}]; equivalent within ±0.5 in every case: {lo['all_equivalent_at_0.5']}."]
    open(os.path.join(REPO, "results", "reviewer", "secondary_and_weighting.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L)); print(json.dumps(lo["tightest_by_dropped_pool"]))


if __name__ == "__main__":
    main()
