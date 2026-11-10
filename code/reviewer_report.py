"""Turn reviewer.run_all() output into reviewer_analyses.md and figures.
Every table is headed EXPLORATORY: none of these analyses was registered."""

from __future__ import annotations

import os

import numpy as np

from reviewer import MEASURES, RULE_DESCR, SUBSETS

SUBS = list(SUBSETS)


def f(v, nd=2, sign=True):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def p(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else (f"{v:.3f}" if v >= 0.001 else f"{v:.1e}")


def ci(d, lo="ci_low", hi="ci_high"):
    return "—" if not d or lo not in d or hi not in d else f"[{f(d[lo])}, {f(d[hi])}]"


def table(head, rows):
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    L += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(L) + "\n"


def build_md(out):
    L = ["# Reviewer analyses (EXPLORATORY)", "",
         "None of these analyses was part of the registration. They read only the saved per-pool records, "
         "held-out logits and frozen features; no probe was refitted. Before anything else the code asserts that "
         f"the saved logits reproduce every registered per-fold utility (max difference "
         f"{out['reconstruction_max_abs_diff']:.1e}), that every matched pair's accuracy-based D is reproduced, and "
         "that the registered selection rules give the registered regrets when applied in-sample.", ""]

    # R1
    L += ["## R1. Competence imbalance within matched pairs (review point 2, Q1)", "",
          "Signed gap = accuracy of the higher-CKA candidate minus the lower-CKA one. Gap-adjusted D is the "
          "intercept of D regressed on the signed gap (cluster-robust by pool): the difference expected at "
          "equal competence.", ""]
    rows = []
    for s in SUBS:
        r = out["R1_competence_gap"].get(s)
        if not r:
            continue
        g = r["signed_gap_cluster_robust"] or {}
        a = r["D_adjusted_all_pairs"] or {}
        rows.append([s, r["pairs"], f(r["signed_gap_mean"]), ci(g), f"{r['frac_higher_cka_more_accurate']:.0%}",
                     f(a.get("intercept_D_at_zero_gap")),
                     "—" if not a else f"[{f(a['ci'][0])}, {f(a['ci'][1])}]", p(a.get("p")),
                     f(a.get("slope_pp_per_gap_point"))])
    L.append(table(["subset", "pairs", "mean signed gap (pts)", "95% CI", "higher-CKA more accurate",
                    "gap-adjusted D", "95% CI", "p", "D per gap point"], rows))

    # R2
    L += ["## R2. Caliper sensitivity (review point 2)", "",
          "Pairs with CKA gap ≥ 0.15 (and beyond the subsampling tolerance) whose competence gap is within the "
          "caliper; 1.5 is the registered caliper.", ""]
    rows = []
    for s in SUBS:
        for c, v in (out["R2_caliper"].get(s) or {}).items():
            if not v:
                rows.append([s, c, "—", "—", "—", "—", "—", "—"])
                continue
            a, pr, t = v["all_pairs"] or {}, v["primary"] or {}, v["tost_all_pairs"] or {}
            rows.append([s, c, a.get("n", "—"), a.get("G", "—"), f(a.get("estimate")), ci(a),
                         f(pr.get("estimate")) + " " + ci(pr), t.get("tightest_margin_cleared") or "none"])
    L.append(table(["subset", "caliper (pts)", "pairs", "G", "D all pairs", "95% CI", "D primary [95% CI]",
                    "TOST (tightest margin)"], rows))

    # R3
    s3 = out["R3_strong"]
    L += ["## R3. Strong candidates only (review point 3, Q2)", "", f"Rule: {s3['rule']}.", ""]
    rows = []
    for s in SUBS:
        v = s3.get(s)
        if not v:
            rows.append([s, 0, "—", "—", "—", "—"])
            continue
        a, t = v["all_pairs"] or {}, v["tost_all_pairs"] or {}
        rows.append([s, v["pairs"], v["pools"], f(a.get("estimate")), ci(a), t.get("tightest_margin_cleared") or "none"])
    L.append(table(["subset", "pairs", "pools", "D", "95% CI", "TOST (tightest margin)"], rows))
    w = s3["matched_pairs_both_weaker_than_base_fusion"]
    L.append(f"Matched pairs in which both candidates are individually weaker than the base fusion: "
             f"{w['count']} of {w['of']}.\n")

    # R4
    L += ["## R4. Dependence between pools (review point 4)", ""]
    rows = []
    for s in SUBS:
        v = out["R4_dependence"].get(s)
        if not v:
            continue
        a, tw, ep = v["pool_clustered"] or {}, v["two_way_pool_x_encoder_pair"], v["encoder_pair_clustered"] or {}
        lr = v["loo_range"]
        rows.append([s, f(a.get("estimate")), ci(a), ci(ep), ci(tw), f"df {tw['df']}",
                     "—" if not lr else f"[{f(lr[0])}, {f(lr[1])}]", v["loo_all_ci_contain_zero"]])
    L.append(table(["subset", "D all pairs", "CI: by pool", "CI: by encoder pair", "CI: two-way",
                    "two-way df", "leave-one-encoder-out range", "every LOO CI contains 0"], rows))

    # R5
    r5 = out["R5_regret_crossfit"]
    L += ["## R5. Selection regret with cross-fitted selection and hybrid rules (review point 1)", "",
          "Label-using statistics for evaluation fold f are computed on the other folds' held-out rows only; "
          "the chosen candidate is scored on fold f. " + r5["caveat"], "",
          "Rules: " + "; ".join(f"**{k}** = {v}" for k, v in RULE_DESCR.items()) + ".", ""]
    rules = [k for k in RULE_DESCR if k != "random"]
    head = ["subset", "pools", "random"] + [f"{k} (cross-fit / in-sample)" for k in rules]
    rows = []
    for s in SUBS:
        v = r5.get(s)
        if not v:
            continue
        rows.append([s, v["pools"], f(v["random_regret"], sign=False)] +
                    [f"{f(v[k + '_crossfit'], sign=False)} / {f(v[k + '_insample'], sign=False)}" for k in rules])
    L.append(table(head, rows))
    rows = []
    for s in SUBS:
        v = r5.get(s)
        if not v:
            continue
        for k in rules:
            c = v.get(f"{k}_vs_lowest_cka_crossfit")
            if c:
                rows.append([s, k, f(c["mean_regret_saved_pp"]), p(c["p_t"]), c["rule_better_in"], c["worse_in"]])
    L.append("Regret saved relative to the lowest-CKA rule (cross-fitted; positive = the rule beats lowest CKA):\n")
    L.append(table(["subset", "rule", "regret saved (pts)", "p (paired t over pools)", "better in", "worse in"], rows))

    # R6
    if "R6_algorithm1" in out:
        L += ["## R6. Algorithm 1 on all pools, cross-fitted (review point 1, Q3)", "",
              "Accuracy of fusing the subset Algorithm 1 keeps, minus fusing the same number of members chosen by "
              "cross-fitted accuracy; and minus fusing all members. Primary fusion rule. Thresholds are a fixed grid, "
              "not tuned.", ""]
        rows = [[x["theta_cka"], x["theta_unique"], x["pools"], f(x["mean_size"], 1, False),
                 f(x["alg1_minus_topk_pp"]), p(x["p_t"]), x["alg1_better_in"], x["alg1_worse_in"],
                 f(x["alg1_minus_all_pp"])] for x in out["R6_algorithm1"]["summary"]]
        L.append(table(["θ_CKA", "θ_unique (%)", "pools", "mean subset size", "Alg 1 − top-k (pts)", "p",
                        "better in", "worse in", "Alg 1 − all members"], rows))

    # R7
    if "R7_similarity" in out:
        L += ["## R7. Other similarity measures on the real pools (review points 5, 6; Q4)", "",
              "Candidate-level, within-pool-centred regression of utility on each measure and on accuracy "
              "(cluster-robust by pool); regret of choosing the least similar candidate by that measure; and the "
              "matched pairs re-oriented so that D = utility of the more similar minus the less similar candidate "
              "under that measure. `logit_*` measures use the held-out probe logits, a class-discriminative "
              "representation.", ""]
        for s in SUBS:
            v = out["R7_similarity"].get(s)
            if not v:
                continue
            rows = []
            for m in MEASURES:
                x = v[m]
                md = x.get("matched_D_oriented_by_measure") or {}
                rows.append([m, f(x["coef_pp_per_unit"]), p(x["p"]), f(x["acc_coef"], 3), p(x["acc_p"]),
                             f(x["spearman_with_utility_within_pool"]), f(x["regret_choose_least_similar_pp"], sign=False),
                             f(md.get("estimate")) + " " + ci(md)])
            L.append(f"**{s}** (G = {v[MEASURES[0]]['G']}, candidates = {v[MEASURES[0]]['n']})\n")
            L.append(table(["measure", "coef (pts/unit)", "p", "accuracy coef", "p", "ρ with utility",
                            "regret: least similar", "matched D by measure [95% CI]"], rows))

    # R8
    r8 = out["R8_metrics"]
    L += ["## R8. Other outcome metrics and per-pool intervals (review point 8)", "", r8["sign_note"] + ".", ""]
    rows = []
    for s in SUBS:
        v = r8.get(s) or {}
        for k in ("macro_f1", "nll"):
            x = v.get(k) or {}
            a, pr = x.get("all_pairs") or {}, x.get("primary") or {}
            rows.append([s, k, f(a.get("estimate"), 3), ci(a), f(pr.get("estimate"), 3), ci(pr)])
    L.append(table(["subset", "metric", "D all pairs", "95% CI", "D primary", "95% CI"], rows))
    rows = [[x["pool"], x["domain"], f"{x['high']} vs {x['low']}", f(x["D"]),
             f"[{f(x['fold_ci_low'])}, {f(x['fold_ci_high'])}]"] for x in r8["per_pool_primary"]]
    L.append("Per-pool primary pair with a fold-level interval (5 folds, t(4); folds share training data, so "
             "these intervals are descriptive):\n")
    L.append(table(["pool", "domain", "pair (higher vs lower CKA)", "D", "fold 95% CI"], rows))

    # R9
    L += ["## R9. Matched versus unmatched comparisons across all pools (review point 9)", ""]
    rows = []
    for s in SUBS:
        v = out["R9_unmatched"].get(s)
        if not v:
            continue
        m, u, g = v["matched_all_pairs"] or {}, v["unmatched_all_pairs"] or {}, v["unmatched_D_on_signed_gap"] or {}
        rows.append([s, f(m.get("estimate")) + " " + ci(m), v["unmatched_pairs"], f(u.get("estimate")) + " " + ci(u),
                     f"{v['unmatched_frac_higher_cka_more_accurate']:.0%}", f(g.get("slope_pp_per_point"), 3),
                     p(g.get("p")), f(g.get("intercept")), p(g.get("intercept_p"))])
    L.append(table(["subset", "matched D [95% CI]", "unmatched pairs", "unmatched D [95% CI]",
                    "unmatched: higher-CKA more accurate", "D per signed-gap point", "p",
                    "D at zero gap", "p"], rows))
    return "\n".join(L)


def figures(out, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    col = {"pathology": "#b03a2e", "vision": "#1f618d", "text": "#1e8449"}
    paths = []
    # per-pool primary D with fold intervals
    P = out["R8_metrics"]["per_pool_primary"]
    if P:
        fig, ax = plt.subplots(figsize=(6.2, 0.32 * len(P) + 1.2))
        for i, x in enumerate(P[::-1]):
            ax.errorbar(x["D"], i, xerr=[[x["D"] - x["fold_ci_low"]], [x["fold_ci_high"] - x["D"]]],
                        fmt="o", color=col.get(x["domain"], "k"), ms=4, capsize=2, lw=1)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvspan(-0.5, 0.5, color="0.9", zorder=0)
        ax.set_yticks(range(len(P)))
        ax.set_yticklabels([x["pool"] for x in P[::-1]], fontsize=7)
        ax.set_xlabel("D (pp), primary pair, fold-level 95% CI (descriptive)", fontsize=8)
        fig.tight_layout()
        p_ = os.path.join(fig_dir, "per_pool_primary_D_with_fold_CI.png")
        fig.savefig(p_, dpi=200); plt.close(fig); paths.append(p_)
    return paths
