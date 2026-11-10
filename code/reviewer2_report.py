"""Markdown report for reviewer2.run_all() output. Everything is EXPLORATORY."""

from __future__ import annotations

from reviewer_report import ci, f, p, table

SUBS = ["all", "pathology", "vision", "text", "image"]


def build_md(o):
    L = ["# Second-round reviewer analyses (EXPLORATORY)", "",
         "Post-registration. The code first re-asserts that the saved logits reproduce every registered utility.", ""]
    s1 = o["S1_pilot_overlap_exclusion"]
    L += ["## S1. Without the six pilot datasets (Q1)", "", f"Excluded: {', '.join(s1['excluded'])}.", ""]
    rows = []
    for s in SUBS:
        v = s1.get(s)
        if not v or not v.get("pools"):
            continue
        pr, a, t = v["primary"] or {}, v["all_pairs"] or {}, v["tost_all_pairs"] or {}
        rg = v["regret"]
        rows.append([s, v["pools"], f(pr.get("estimate")) + " " + ci(pr), a.get("n", "—"), f(a.get("estimate")) + " " + ci(a),
                     t.get("tightest_margin_cleared") or "none", f(rg.get("lowest_cka"), sign=False),
                     f(rg.get("competence"), sign=False), f(rg.get("unique"), sign=False)])
    L.append(table(["subset", "pools", "D primary [95% CI]", "pairs", "D all pairs [95% CI]", "TOST",
                    "regret lowest CKA", "regret competence", "regret unique"], rows))

    s2 = o["S2_gated_rule"]
    L += ["## S2. Hybrid rules: all rules tried, gate width, paired tests (concern 3, Q2)", "",
          "Rules tried in the first exploratory round (complete list, fixed in code before it ran): "
          + ", ".join(s2["rules_tried_round1"]) + ".", "",
          "Cross-fitted regret of the gated rule by gate width (points). Per-fold selection can beat the "
          "mean-based oracle on a fold, so regret can be slightly negative.", ""]
    rows = [[s] + [f(v[g], sign=False) for g in ("0.5", "1.0", "1.5", "3.0")]
            for s, v in s2["gate_width_crossfit"].items()]
    L.append(table(["subset", "gate 0.5", "gate 1.0", "gate 1.5", "gate 3.0"], rows))
    L.append("Paired, cross-fitted: regret of competence minus regret of the rule (positive = rule better):\n")
    rows = []
    for s, d in s2["paired_vs_competence_crossfit"].items():
        for rule, x in d.items():
            rows.append([s, rule, f(x["mean"]), ci(x), p(x.get("p_t")), p(x.get("p_wilcoxon")),
                         x["better_in"], x["worse_in"], x["tie"]])
    L.append(table(["subset", "rule", "mean", "95% CI", "p (t)", "p (Wilcoxon)", "better", "worse", "tie"], rows))

    if "S3_nested_regret" in o:
        s3 = o["S3_nested_regret"]
        L += ["## S3. Fully nested selection (concern 4, Q2)", "",
              f"Selection statistics from probes refitted by inner {s3['inner_k']}-fold CV on each outer training split; "
              f"no label of the evaluation fold is used in selection. Pools with n ≤ {s3['n_max']}: "
              + ", ".join(x["pool"] for x in s3["per_pool"]) + ".", ""]
        rows = []
        for s, v in s3["summary"].items():
            g, c = v["gated_vs_competence"], v["competence_vs_lowest_cka"]
            rows.append([s, v["pools"], f(v["lowest_cka"], sign=False), f(v["competence"], sign=False),
                         f(v["unique"], sign=False), f(v["disagreement"], sign=False), f(v["gated_1.5"], sign=False),
                         f(v["rank_sum"], sign=False), f(g["mean"]) + " " + ci(g) + f" (p {p(g.get('p_t'))})",
                         f(c["mean"]) + " " + ci(c) + f" (p {p(c.get('p_t'))})"])
        L.append(table(["subset", "pools", "lowest CKA", "competence", "unique", "disagreement", "gated 1.5", "rank sum",
                        "gated vs competence", "competence vs lowest CKA"], rows))

    if "S4_c_sensitivity" in o:
        L += ["## S4. Probe regularisation C (concern 9)", "", "Same folds and matched pairs; C = 1 is registered "
              "(asserted to reproduce the registered utilities).", ""]
        rows = []
        for pool, d in o["S4_c_sensitivity"]["pools"].items():
            for C, v in d.items():
                rows.append([pool, C, f(v["mean_indiv_acc"], sign=False), v["matched_pairs"], f(v["matched_D_mean"]),
                             f(v["regret_lowest_cka"], sign=False), f(v["regret_competence"], sign=False)])
        L.append(table(["pool", "C", "mean candidate accuracy", "matched pairs", "mean D",
                        "regret lowest CKA", "regret competence"], rows))

    s5 = o["S5_association_consistency"]
    L += ["## S5. Is the weak association consistent across pools? (Q4)", ""]
    rows = []
    for m, v in s5["per_pool_sign"].items():
        lo = s5["leave_one_pool_out"][m]
        rows.append([m, f"{v['negative_in']} of {v['pools']}", f"[{f(lo['coef_range'][0])}, {f(lo['coef_range'][1])}]",
                     f"[{p(lo['p_range'][0])}, {p(lo['p_range'][1])}]", lo["most_influential"]])
    L.append(table(["measure", "pools with negative slope", "LOO coefficient range", "LOO p range",
                    "pool whose removal weakens it most"], rows))

    s6 = o["S6_multiplicity"]
    L += ["## S6. Multiplicity (concern 2)", "", s6["note"] + ".", ""]
    for fam in ("table4_all_pools", "R7_all_subsets", "D_robustness"):
        rows = [[x["test"], p(x["p"]), p(x["q_bh"]), p(x["p_holm"])] for x in s6[fam]]
        L.append(f"**{fam}** (smallest BH q = {p(s6[fam + '_min_q'])})\n")
        L.append(table(["test", "p", "q (BH)", "p (Holm)"], rows))

    L += ["## S7. Primary pairs: signed competence gap and fold interval (minor point)", ""]
    rows = [[x["pool"], f"{x['high']} vs {x['low']}", f(x["cka_gap"], sign=False), f(x["signed_gap"]), f(x["D"]),
             f"[{f(x['fold_ci_low'])}, {f(x['fold_ci_high'])}]"] for x in o["S7_primary_pairs"]]
    L.append(table(["pool", "higher vs lower CKA", "CKA gap", "signed gap (pts)", "D", "fold 95% CI"], rows))

    if "S8_encoder_validation" in o:
        v = o["S8_encoder_validation"]
        L += ["## S8. Encoder validation (concern 9)", ""]
        rows = []
        for k in ("lunit_vits16", "kaiko_vitb16"):
            x = v.get(k, {})
            rows.append([k, x.get("source", "—"), f"{x.get('same_shape', '—')} / {x.get('tensors_a', '—')}",
                         x.get("max_abs_diff", x.get("error", "—"))])
        L.append(table(["encoder", "official source", "tensors compared", "max abs difference (port vs official)"], rows))
        rows = []
        for k in ("dinov2_b_vs_official", "dinov2_l_vs_official"):
            x = v.get(k)
            if x:
                rows.append([k, x.get("n_images", "—"), f(x.get("cka"), 5, False), f(x.get("mean_cosine"), 5, False),
                             x.get("error", "")])
        if rows:
            L.append(table(["comparison", "images", "CKA (ours vs official)", "mean cosine", "error"], rows))
    return "\n".join(L)
