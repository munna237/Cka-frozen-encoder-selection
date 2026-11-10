"""Paper tables from the SAVED results of notebook 04. Read-only: it never
re-fits a probe or a fusion and never changes battery.json.

Two kinds of output, kept strictly apart:

  REGISTERED   numbers read straight from battery.json (estimands A-F). The only
               change is presentation: equivalence is reported as the TIGHTEST
               margin cleared (summary.md printed max() of the cleared margins,
               i.e. the weakest claim -- a labelling bug in a registered file,
               corrected here at the reporting stage).
  EXPLORATORY  analyses that were not pre-registered (paired regret test,
               leave-one-pool-out). Every such table is labelled EXPLORATORY.

Every recomputation that should equal a registered number is asserted to.
"""

from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np
from scipy import stats

import config as CFG
from inference import cluster_robust_ols

SUBSET_ORDER = ["all", "pathology", "vision", "text", "image"]
RULES = ["cka_diversity", "disagreement_with_base", "competence", "oracle_headroom"]
REVERSE = {"cka_diversity": False, "disagreement_with_base": True,
           "competence": True, "oracle_headroom": True}
KEY = {"cka_diversity": "cka", "disagreement_with_base": "disagreement",
       "competence": "indiv_acc", "oracle_headroom": "headroom_contrib"}


def load(results_dir=None):
    rd = results_dir or CFG.RESULTS_DIR
    with open(os.path.join(rd, "battery.json")) as fh:
        bat = json.load(fh)
    pools = {}
    for f in sorted(glob.glob(os.path.join(rd, "pools", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        pools[r["pool"]] = r
    return bat, pools


def _subset_pools(bat, name):
    return bat[name].get("pools", [])


# ---------------------------------------------------------------- tables
def pools_table(pools):
    rows = []
    for name, r in pools.items():
        prim = next((p for p in r["matched_pairs"] if p.get("primary")), None)
        rows.append({
            "pool": name, "domain": r["domain"], "n": r["n_inputs"], "encoders": r["n_members"],
            "grouped": r["grouped_folds"], "anchor": r["anchor"], "base_member": r["base"][1],
            "matched_pairs": len(r["matched_pairs"]),
            "primary_pair": f"{prim['high']} vs {prim['low']}" if prim else "",
            "primary_cka_gap": round(prim["cka_gap"], 3) if prim else None,
            "primary_comp_gap_pp": round(prim["comp_gap"], 2) if prim else None,
            "primary_D_pp": round(prim["diff"], 2) if prim else None,
        })
    return rows


def estimand_A(bat):
    rows = []
    for s in SUBSET_ORDER:
        b = bat.get(s, {})
        for key, label in (("A_matched_null_primary_only", "primary (1 pair/pool)"),
                           ("A_matched_null_all_pairs", "all matched pairs")):
            a = b.get(key) or {}
            cr = a.get("cluster_robust")
            if not cr:
                rows.append({"subset": s, "analysis": label,
                             "note": a.get("not_estimable", "n/a")})
                continue
            me = a.get("mixed_effects", {})
            rows.append({"subset": s, "analysis": label, "D_pp": cr["estimate_pp"],
                         "ci_low": cr["ci_low"], "ci_high": cr["ci_high"], "p": cr["p"],
                         "pairs": cr["n"], "G": cr["G"], "reference": cr["reference"],
                         "mixed_effects_D": me.get("estimate"), "mixed_effects_p": me.get("p")})
    return rows


def estimand_C(bat):
    """TOST at the registered margins; tightest margin cleared (the correct label)."""
    rows = []
    for s in SUBSET_ORDER:
        c = bat.get(s, {}).get("C_equivalence") or {}
        if "not_estimable" in c or not c:
            rows.append({"subset": s, "note": c.get("not_estimable", "n/a")})
            continue
        margins = sorted((float(k.split("_")[1]) for k in c if k.startswith("margin_")), reverse=True)
        cleared = [m for m in margins if c[f"margin_{m}"]["equivalent"]]
        row = {"subset": s, "G": c["G"],
               "estimate_pp": c[f"margin_{margins[0]}"]["estimate"],
               "ci90_low": c[f"margin_{margins[0]}"]["ci90_low"],
               "ci90_high": c[f"margin_{margins[0]}"]["ci90_high"],
               "tightest_margin_cleared": min(cleared) if cleared else None,
               "summary_md_printed": c.get("largest_margin_cleared")}
        for m in margins:
            row[f"p_tost_{m}"] = c[f"margin_{m}"]["p_tost"]
        rows.append(row)
    return rows


def estimand_B_E(bat):
    rows = []
    for s in SUBSET_ORDER:
        b = bat.get(s, {})
        B = b.get("B_dose_response") or {}
        E = (b.get("E_mechanism_on_real") or {}).get("joint_model") or {}
        rows.append({
            "subset": s,
            "B_slope_pp_per_cka": B.get("slope_pp_per_unit_cka"), "B_p": B.get("p"),
            "B_mde80": B.get("mde_80pct_power"),
            "B_interpretable(MDE<=0.5)": (B.get("mde_80pct_power", 99) <= 0.5) if B else None,
            "E_cka_coef": (E.get("cka") or {}).get("coef_pp_per_unit"),
            "E_cka_p": (E.get("cka") or {}).get("p"),
            "E_disagreement_coef": (E.get("disagreement") or {}).get("coef_pp_per_unit"),
            "E_disagreement_p": (E.get("disagreement") or {}).get("p"),
            "E_acc_coef": (E.get("indiv_acc") or {}).get("coef_pp_per_unit"),
            "E_acc_p": (E.get("indiv_acc") or {}).get("p"),
            "E_G": (b.get("E_mechanism_on_real") or {}).get("G"),
        })
    return rows


def _regrets(cands):
    best = max(c["utility"] for c in cands)
    out = {}
    for r in RULES:
        fn = max if REVERSE[r] else min
        pick = fn(cands, key=lambda c: float(c[KEY[r]]))
        out[r] = best - pick["utility"]
    return out


def estimand_D(bat, pools):
    """Registered regret table + EXPLORATORY paired test CKA vs competence."""
    reg_rows, paired_rows = [], []
    for s in SUBSET_ORDER:
        D = (bat.get(s, {}).get("D_selection_regret") or {})
        pr = D.get("per_rule", {})
        for r in RULES:
            if r in pr:
                reg_rows.append({"subset": s, "criterion": r, **{k: pr[r][k] for k in
                                 ("mean_regret_pp", "median_regret_pp", "p90_regret_pp",
                                  "top1_agreement_with_oracle", "n_pools")}})
        names = [p for p in _subset_pools(bat, s) if p in pools]
        per = {p: _regrets([c for c in pools[p]["candidates"] if np.isfinite(c.get("utility", np.nan))])
               for p in names}
        if not per:
            continue
        for r in RULES:          # recomputation must equal the registered mean
            mine = float(np.mean([per[p][r] for p in per]))
            assert abs(mine - pr[r]["mean_regret_pp"]) < 1e-9, (s, r, mine, pr[r]["mean_regret_pp"])
        d = np.array([per[p]["cka_diversity"] - per[p]["competence"] for p in per])
        t = stats.ttest_1samp(d, 0.0) if len(d) > 1 else None
        try:
            w = stats.wilcoxon(d) if np.any(d != 0) and len(d) > 1 else None
        except ValueError:
            w = None
        paired_rows.append({
            "subset": s, "pools": len(d),
            "mean_extra_regret_cka_vs_competence_pp": float(d.mean()),
            "ci_low": float(d.mean() - stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
            "ci_high": float(d.mean() + stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
            "t_p": float(t.pvalue) if t is not None else None,
            "wilcoxon_p": float(w.pvalue) if w is not None else None,
            "cka_worse": int((d > 1e-9).sum()), "tie": int((np.abs(d) <= 1e-9).sum()),
            "cka_better": int((d < -1e-9).sum()),
        })
    return reg_rows, paired_rows


def leave_one_pool_out(bat, pools, subset):
    """EXPLORATORY: estimand A recomputed with each pool dropped in turn.
    The full-sample recomputation is asserted equal to the registered value."""
    names = [p for p in _subset_pools(bat, subset) if p in pools]
    out = []
    for key, prim in (("A_matched_null_all_pairs", False), ("A_matched_null_primary_only", True)):
        def rows(excl=None):
            rr = [m for p in names if p != excl for m in pools[p]["matched_pairs"]
                  if np.isfinite(m["diff"]) and (m.get("primary") or not prim)]
            return np.array([m["diff"] for m in rr]), np.array([m["pool"] for m in rr])
        y, cl = rows()
        if len(np.unique(cl)) < 2:
            continue
        full = cluster_robust_ols(y, cl)
        reg = bat[subset][key]["cluster_robust"]
        assert abs(float(full["estimate"][0]) - reg["estimate_pp"]) < 1e-9
        assert abs(float(full["ci_low"][0]) - reg["ci_low"]) < 1e-9
        for p in [None] + names:
            y, cl = rows(p)
            if len(np.unique(cl)) < 2:
                continue
            r = cluster_robust_ols(y, cl)
            out.append({"subset": subset, "analysis": "primary" if prim else "all pairs",
                        "dropped": p or "(none)", "D_pp": float(r["estimate"][0]),
                        "ci_low": float(r["ci_low"][0]), "ci_high": float(r["ci_high"][0]),
                        "p": float(r["p"][0]), "G": int(r["G"]), "pairs": int(r["n"])})
    return out


def fit_caps(results_dir=None):
    p = os.path.join(results_dir or CFG.RESULTS_DIR, "fusion_fit_log.csv")
    if not os.path.exists(p):
        return []
    agg = {}
    with open(p) as fh:
        for r in csv.DictReader(fh):
            a = agg.setdefault(r["pool"], {"pool": r["pool"], "classes": int(r["C"]),
                                           "fits": 0, "capped": 0})
            a["fits"] += 1
            a["capped"] += r["converged"] != "True"
    for a in agg.values():
        a["capped_pct"] = round(100 * a["capped"] / a["fits"], 1)
    return list(agg.values())


# ---------------------------------------------------------------- writing
def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 0.001 and v != 0 else f"{v:.{nd}f}"
    return str(v)


def md_table(rows, cols=None):
    if not rows:
        return "_(none)_\n"
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    L += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(L) + "\n"


def build(results_dir=None):
    rd = results_dir or CFG.RESULTS_DIR
    bat, pools = load(rd)
    T = {"pools": pools_table(pools), "A": estimand_A(bat), "C": estimand_C(bat),
         "BE": estimand_B_E(bat)}
    T["D_registered"], T["D_paired_exploratory"] = estimand_D(bat, pools)
    T["LOO_text_exploratory"] = leave_one_pool_out(bat, pools, "text") if "text" in bat else []
    T["LOO_all_exploratory"] = leave_one_pool_out(bat, pools, "all")
    T["fit_caps"] = fit_caps(rd)
    md = [f"# v3 paper tables\n\nRegistered {bat['prereg']['registered_utc']} "
          f"(hash {bat['prereg']['registration_hash'][:12]}); deviations: "
          f"{bat['prereg']['deviations'] or 'none'}.\n",
          "## Pools\n", md_table(T["pools"]),
          "## A. Matched-competence null (REGISTERED; D = utility(higher-CKA) − utility(lower-CKA), pp)\n",
          md_table(T["A"]),
          "## C. Equivalence, TOST (REGISTERED; all matched pairs, cluster-robust)\n",
          "`tightest_margin_cleared` is the claim to make. `summary_md_printed` shows what summary.md "
          "reported (the largest margin cleared, i.e. the weakest claim).\n\n", md_table(T["C"]),
          "## B and E. Dose-response and mechanism (REGISTERED)\n", md_table(T["BE"]),
          "## D. Selection regret (REGISTERED)\n", md_table(T["D_registered"]),
          "## D′. Paired regret, CKA diversity minus competence (EXPLORATORY — not a registered test)\n",
          md_table(T["D_paired_exploratory"]),
          "## Leave-one-pool-out, text subset (EXPLORATORY)\n", md_table(T["LOO_text_exploratory"]),
          "## Leave-one-pool-out, all pools (EXPLORATORY)\n", md_table(T["LOO_all_exploratory"]),
          "## Secondary fusion rule: fits reaching the registered 500-iteration cap\n",
          "Pools analysed before the fit log existed (the pathology pools and DTD, Aircraft, EuroSAT, Pets) "
          "are not listed.\n\n", md_table(T["fit_caps"])]
    text = "\n".join(md)
    with open(os.path.join(rd, "paper_tables.md"), "w") as fh:
        fh.write(text)
    with open(os.path.join(rd, "paper_tables.json"), "w") as fh:
        json.dump(T, fh, indent=1, default=float)
    return T, text
