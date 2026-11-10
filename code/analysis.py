"""The analysis battery: every estimand the resubmission needs, in one place.

Each function takes a `results` record produced by run_pipeline.py and returns
plain dicts so the whole thing can be dumped to JSON and asserted by
verify_claims.py -- the same discipline that made the v2 release checkable.

The five estimands, in the order they should appear in the paper:

  A. matched_null        primary: does CKA predict marginal utility at matched
                         competence? cluster-robust, t(G-1).
  B. dose_response       does the effect GROW with the CKA gap? This is the test
                         v2 could not run (slope SE = 1.20 on 6 points).
  C. equivalence         TOST at +/- margin. Upgrades "failed to reject" to
                         "established equivalent", which needs G >= ~12.
  D. selection_regret    the positive result: what does following the CKA
                         heuristic COST, against competence-only and against
                         other pre-fusion criteria? Not vulnerable to "your CI
                         was too wide", because it is a measured cost, not a null.
  E. mechanism_on_real   does held-out disagreement predict utility at matched
                         competence in REAL pools? v2 could only show this
                         synthetically; the AE called that "not independent".
"""

from __future__ import annotations

import numpy as np

from inference import (cluster_robust_ols, mannwhitney_u, mixed_effects_random_intercept,
                       paired_fold_test, spearman, tost_equivalence)


# --------------------------------------------------------------------------
# A. primary matched-pair null
# --------------------------------------------------------------------------

def matched_null(records: list[dict], margin_pairs: bool = False) -> dict:
    """records: one dict per matched pair with keys
         pool, pair_id, diff (per-pair mean of fold-level utility differences),
         fold_diffs (list/array of per-fold differences), primary (bool)

    Cluster = pool. Observations = one per pair (default) so that multiple pairs
    from the same pool do not masquerade as independent datasets.
    """
    rows = [r for r in records if r.get("primary", True)] if margin_pairs else records
    if not rows:
        return {"n_pairs": 0, "G": 0, "error": "no matched pairs supplied"}
    y = np.array([float(r["diff"]) for r in rows])
    cl = np.array([r["pool"] for r in rows])
    G = len(np.unique(cl))
    out = {"n_pairs": len(rows), "G": G}

    cr = cluster_robust_ols(y, cl)
    # intercept-only model: expose scalars, not 1-element arrays, so downstream
    # JSON and comparisons are unambiguous.
    out["cluster_robust"] = {
        "estimate_pp": float(cr["estimate"][0]), "se": float(cr["se"][0]),
        "t": float(cr["t"][0]), "df": int(cr["df"]), "p": float(cr["p"][0]),
        "ci_low": float(cr["ci_low"][0]), "ci_high": float(cr["ci_high"][0]),
        "n": int(cr["n"]), "G": int(cr["G"]),
        "reference": f"t(G-1) = t({int(cr['df'])})",
    }
    if G >= 2:
        me = mixed_effects_random_intercept(y, cl)
        out["mixed_effects"] = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                for k, v in me.items()}
    # fold-level descriptive statistics (NOT primary; folds within a pool share data)
    allf = np.concatenate([np.asarray(r["fold_diffs"], float) for r in rows]) \
        if all("fold_diffs" in r for r in rows) else np.array([])
    if allf.size:
        out["fold_level_naive"] = paired_fold_test(allf)
        out["fold_level_naive"]["WARNING"] = ("folds within a pool share training data; "
                                              "quote the cluster-robust figure")
    return out


# --------------------------------------------------------------------------
# B. dose-response
# --------------------------------------------------------------------------

def dose_response(records: list[dict]) -> dict:
    """Regress the matched-pair utility difference on the CKA gap.

    Under the diversity heuristic a real effect must GROW with the gap. v2 got
    slope +0.09 with SE 1.20 (p = 0.94) over gaps 0.012-0.572 -- i.e. no power.
    Report the minimum detectable effect alongside the slope so a reviewer can
    see the power directly instead of inferring it from a p-value.
    """
    y = np.array([float(r["diff"]) for r in records])
    x = np.array([float(r["cka_gap"]) for r in records])
    cl = np.array([r["pool"] for r in records])
    res = cluster_robust_ols(y, cl, x=x)
    slope, se = float(res["estimate"][1]), float(res["se"][1])
    mde = 2.8 * se                      # ~80% power at alpha=.05, two-sided
    return {
        "slope_pp_per_unit_cka": slope, "se": se, "t": float(res["t"][1]),
        "df": res["df"], "p": float(res["p"][1]),
        "ci_low": float(res["ci_low"][1]), "ci_high": float(res["ci_high"][1]),
        "mde_80pct_power": mde,
        "cka_gap_range": [float(x.min()), float(x.max())],
        "n_pairs": len(y), "G": res["G"],
        "spearman": spearman(x, y),
        "INTERPRETATION": ("slope CI excludes any effect large enough to matter"
                           if abs(res["ci_high"][1]) < 1.0 and abs(res["ci_low"][1]) < 1.0
                           else "design still under-powered for a 1 pp/unit effect"),
    }


# --------------------------------------------------------------------------
# C. equivalence
# --------------------------------------------------------------------------

def equivalence(records: list[dict], margins=(1.0, 0.5, 0.25)) -> dict:
    """TOST at several margins. Report the largest margin you clear and say so
    plainly; do not claim equivalence at a margin the test does not support
    (v2 correctly refused to at n = 3)."""
    y = np.array([float(r["diff"]) for r in records])
    cl = np.array([r["pool"] for r in records])
    out = {}
    for m in margins:
        r = tost_equivalence(y, cl, margin=m)
        out[f"margin_{m}"] = r
    cleared = [m for m in margins if out[f"margin_{m}"]["equivalent"]]
    out["largest_margin_cleared"] = max(cleared) if cleared else None
    out["G"] = len(np.unique(cl))
    out["NOTE"] = ("equivalence established" if cleared else
                   "equivalence NOT established; report as failure to reject only")
    return out


# --------------------------------------------------------------------------
# D. selection regret -- the positive result
# --------------------------------------------------------------------------

def _select_by(cands: list[dict], key: str, reverse: bool) -> str:
    """Return the NAME of the selected candidate.

    Returns a name, not a dict: an earlier version returned the dict and the
    caller then failed a dict-membership test with 'unhashable type: dict'.
    `reverse=True` selects the largest value of `key` (competence, disagreement,
    headroom); `reverse=False` the smallest (CKA -- i.e. most divergent).
    """
    fn = max if reverse else min
    return fn(cands, key=lambda c: float(c[key]))["name"]


# Keys each regret criterion reads, in order of preference. This indirection
# exists because of a real bug: the rule asked for "disagreement_zs" while
# run_pool emitted "disagreement", every pool raised KeyError, every pool was
# silently appended to skipped_pools, and the criterion simply vanished from the
# output table. The headline comparison of the paper was missing and nothing
# failed loudly. Any key set that resolves to nothing now raises instead.
CRITERION_KEYS = {
    "cka_diversity": ("cka",),
    "disagreement_with_base": ("disagreement",),
    "competence": ("indiv_acc", "acc"),
    "oracle_headroom": ("headroom_contrib", "headroom"),
}


def _resolve(cands: list[dict], keys: tuple) -> str:
    """The first key present in the candidate records; raise if none is."""
    for k in keys:
        if all(k in c and c[k] is not None for c in cands):
            return k
    raise KeyError(f"none of {keys} present in candidate records "
                   f"(available: {sorted(cands[0])})")


def selection_regret(pools: dict[str, list[dict]]) -> dict:
    """pools: {pool_name: [candidate dicts]} where each candidate dict has
         name, cka (with anchor), indiv_acc, disagreement, utility (mean
         realised marginal utility over folds), headroom_contrib

    Criteria compared, all computable BEFORE any fusion is fit:
      cka_diversity          least similar to the anchor. The heuristic under
                             test, and the ONLY criterion that needs no labels.
      disagreement_with_base most different held-out predictions from the base.
                             NOT label-free: the predictions come from linear
                             probes trained on labelled folds, so it uses the
                             same information as `competence`. Do not describe
                             it as zero-shot in the paper.
      competence             individually strongest candidate (labels).
      oracle_headroom        largest unique-correct fraction (labels).
    Regret is measured against the candidate with the largest realised utility.

    Headline numbers: mean regret in accuracy points, and top-1 agreement rate
    with the oracle. A measured regret is immune to the "your null was
    under-powered" objection that sank v2.
    """
    reverse = {"cka_diversity": False, "disagreement_with_base": True,
               "competence": True, "oracle_headroom": True}
    rules = {r: (lambda cs, r=r: _select_by(cs, _resolve(cs, CRITERION_KEYS[r]), reverse[r]))
             for r in CRITERION_KEYS}
    out: dict = {r: {"regret_pp": [], "top1": [], "pools": 0} for r in rules}
    per_pool = {}
    skipped = []
    for pool, cands in pools.items():
        # A pool is unusable for regret if it has <2 candidates with finite utility,
        # or if names are not unique. Skip loudly rather than raising, so one bad
        # pool cannot take down the whole battery.
        cands = [c for c in cands if np.isfinite(c.get("utility", np.nan)) and "name" in c]
        if len(cands) < 2:
            skipped.append({"pool": pool, "reason": f"only {len(cands)} usable candidates"})
            continue
        if len({c["name"] for c in cands}) != len(cands):
            skipped.append({"pool": pool, "reason": "duplicate candidate names"})
            continue
        by_name = {c["name"]: c for c in cands}
        best = max(c["utility"] for c in cands)
        best_name = max(cands, key=lambda c: c["utility"])["name"]
        per_pool[pool] = {"oracle_choice": best_name, "oracle_utility": best}
        for rname, fn in rules.items():
            try:
                pick = fn(cands)
            except (ValueError, KeyError) as e:
                skipped.append({"pool": pool, "reason": f"{rname}: {e}"})
                continue
            if pick not in by_name:                       # defensive: should not happen
                skipped.append({"pool": pool, "reason": f"{rname} picked unknown {pick!r}"})
                continue
            u = by_name[pick]["utility"]
            out[rname]["regret_pp"].append(best - u)
            out[rname]["top1"].append(1.0 if pick == best_name else 0.0)
            out[rname]["pools"] += 1
            per_pool[pool][rname] = pick

    summary = {}
    for rname, d in out.items():
        if not d["regret_pp"]:
            continue
        reg = np.array(d["regret_pp"])
        summary[rname] = {
            "mean_regret_pp": float(reg.mean()),
            "median_regret_pp": float(np.median(reg)),
            "p90_regret_pp": float(np.percentile(reg, 90)),
            "max_regret_pp": float(reg.max()),
            "top1_agreement_with_oracle": float(np.mean(d["top1"])),
            "zero_regret_fraction": float(np.mean(reg == 0)),
            "n_pools": d["pools"],
        }
    ck = summary.get("cka_diversity", {})
    di = summary.get("disagreement_with_base", {})
    return {
        "per_rule": summary,
        "per_pool": per_pool,
        "skipped_pools": skipped,
        "n_pools_scored": len(per_pool),
        "HEADLINE": (f"following CKA diversity costs {ck.get('mean_regret_pp', float('nan')):.2f} pp "
                     f"on average vs. the oracle and picks the right teacher "
                     f"{100 * ck.get('top1_agreement_with_oracle', float('nan')):.0f}% of the time; "
                     f"prediction disagreement costs {di.get('mean_regret_pp', float('nan')):.2f} pp "
                     f"and is right {100 * di.get('top1_agreement_with_oracle', float('nan')):.0f}%")
        if ck and di else "insufficient pools",
    }


# --------------------------------------------------------------------------
# E. mechanism on real pools
# --------------------------------------------------------------------------

def mechanism_on_real(records: list[dict]) -> dict:
    """records: one per candidate (not per pair), with keys
         pool, name, cka, indiv_acc, disagreement (fraction of held-out inputs
         on which the candidate's probe prediction differs from the base fusion;
         probes are trained on labelled folds), utility

    Two regressions, both cluster-robust by pool:
      utility ~ disagreement + indiv_acc   -> does the mechanism hold in real data?
      utility ~ cka + indiv_acc           -> does CKA add anything once you control
                                             for competence?
    v2 reported disagreement going the WRONG way on real pools because of the
    competence entanglement; adding the competence term and using the controlled
    pools is what fixes it.
    """
    y = np.array([float(r["utility"]) for r in records])
    cl = np.array([r["pool"] for r in records])
    dis = np.array([float(r["disagreement"]) for r in records])
    cka = np.array([float(r["cka"]) for r in records])
    acc = np.array([float(r["indiv_acc"]) for r in records])

    def pack(res, idx, name):
        return {"coef_pp_per_unit": float(res["estimate"][idx]),
                "se": float(res["se"][idx]), "t": float(res["t"][idx]),
                "p": float(res["p"][idx]), "df": res["df"],
                "ci": [float(res["ci_low"][idx]), float(res["ci_high"][idx])],
                "term": name}

    r_dis = cluster_robust_ols(y, cl, x=np.stack([dis, acc], axis=1))
    r_cka = cluster_robust_ols(y, cl, x=np.stack([cka, acc], axis=1))
    r_both = cluster_robust_ols(y, cl, x=np.stack([cka, dis, acc], axis=1))
    return {
        "utility_on_disagreement_controlling_competence": pack(r_dis, 1, "disagreement"),
        "competence_coef_in_that_model": pack(r_dis, 2, "indiv_acc"),
        "utility_on_cka_controlling_competence": pack(r_cka, 1, "cka"),
        "joint_model": {"cka": pack(r_both, 1, "cka"),
                        "disagreement": pack(r_both, 2, "disagreement"),
                        "indiv_acc": pack(r_both, 3, "indiv_acc")},
        "marginal_spearmans": {"cka_vs_utility": spearman(cka, y),
                               "disagreement_vs_utility": spearman(dis, y),
                               "competence_vs_utility": spearman(acc, y),
                               "cka_vs_disagreement": spearman(cka, dis)},
        "n_candidates": len(y), "G": len(np.unique(cl)),
    }


def matched_band_comparison(records: list[dict], band=(0.55, 0.75),
                            low_acc=0.70, high_acc=0.90,
                            split: str = "relative", min_per_arm: int = 5) -> dict:
    """The v2 synthetic result (within a matched-CKA band, competent candidates
    give +8.44 pp vs +0.04 for weak ones) reproduced on REAL encoders. This is
    the single most persuasive figure the resubmission can have, and v2 could
    only produce it on synthetic data."""
    in_band = [r for r in records if band[0] <= float(r["cka"]) <= band[1]]
    accs = np.array([float(r["indiv_acc"]) for r in in_band])

    if split == "relative":
        # Absolute thresholds (v2 used 0.70 / 0.90) silently produce an empty arm
        # whenever a pool's accuracies all sit above or below them -- it returned
        # weak n=0 on the first end-to-end run. Splitting at the in-band median
        # makes the comparison well-defined for any pool, at the cost of no longer
        # being comparable to v2's exact numbers. Report which you used.
        if len(accs) < 2 * min_per_arm:
            cut = None
        else:
            cut = float(np.median(accs))
        lo_t = hi_t = cut
    else:
        lo_t, hi_t, cut = low_acc, high_acc, None

    if cut is None and split == "relative":
        weak = strong = []
    elif split == "relative":
        weak = [float(r["utility"]) for r in in_band if float(r["indiv_acc"]) < lo_t]
        strong = [float(r["utility"]) for r in in_band if float(r["indiv_acc"]) > hi_t]
    else:
        weak = [float(r["utility"]) for r in in_band if float(r["indiv_acc"]) < low_acc]
        strong = [float(r["utility"]) for r in in_band if float(r["indiv_acc"]) > high_acc]

    mw = mannwhitney_u(np.array(strong), np.array(weak))
    return {
        "band": band, "n_in_band": len(in_band), "split": split, "cut": cut,
        "acc_thresholds": [lo_t, hi_t],
        "mean_utility_weak_pp": float(np.mean(weak)) if weak else np.nan,
        "mean_utility_strong_pp": float(np.mean(strong)) if strong else np.nan,
        "n_weak": len(weak), "n_strong": len(strong),
        "mannwhitney": mw,
        "NOTE": ("thresholds must be declared before looking at utilities"
                 if len(weak) >= min_per_arm and len(strong) >= min_per_arm
                 else f"band too sparse ({len(weak)} weak vs {len(strong)} strong); "
                      "widen the band or add pools before quoting a p-value"),
    }
