"""Exploratory analyses requested in review (all POST-REGISTRATION).

Everything here reads saved artefacts only: the per-pool records and held-out
logits written by notebook 04 and the frozen features written by notebooks 01-03.
No probe is refitted, so no GPU is needed. Wherever a registered quantity is
re-derived (per-fold utilities, D of every pair) it is asserted equal to the
registered value first, so a reader can trust that the new analyses run on the
same numbers as the paper.

Sections (keys of the returned dict) and the review point each answers:
  R1_competence_gap   point 2 / Q1  signed competence gaps; gap-adjusted D
  R2_caliper          point 2       calipers 0.5 / 1.0 / 1.5 / 3.0 points
  R3_strong           point 3 / Q2  pairs whose candidates are within 3 pts of the anchor
  R4_dependence       point 4       two-way clustering (pool x encoder pair), leave-one-encoder-out
  R5_regret_crossfit  point 1       cross-fitted selection; hybrid rules; random baseline
  R6_algorithm1       point 1 / Q3  Algorithm 1 subsets vs size-matched competence subsets
  R7_similarity       points 5, 6 / Q4  CKA vs base ensemble, debiased and RBF CKA,
                                    mutual-kNN, logit-level CKA, on the real pools
  R8_metrics          point 8       D under macro-F1 and NLL; per-pool fold-level CIs
  R9_unmatched        point 9       matched vs unmatched comparison across all 20 pools
"""

from __future__ import annotations

import glob
import json
import os
import time

import warnings

import numpy as np
from scipy import stats

import config as CFG
import fusion
from inference import cluster_robust_ols, tost_equivalence

SUBSETS = {"all": ("pathology", "vision", "text"), "pathology": ("pathology",),
           "vision": ("vision",), "text": ("text",), "image": ("pathology", "vision")}
ORDER = [ds for dom in ("pathology", "vision", "text") for ds in CFG.DOMAINS[dom][0]]
EXPLORATORY = "EXPLORATORY (post-registration; not part of the registered analysis)"


# ---------------------------------------------------------------- loading
def load(results_dir=None):
    rd = results_dir or CFG.RESULTS_DIR
    recs = {}
    for f in glob.glob(os.path.join(rd, "pools", "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        recs[r["pool"]] = r
    order = [p for p in ORDER if p in recs] + sorted(p for p in recs if p not in ORDER)
    return [recs[p] for p in order]


def load_logits(rec, results_dir=None):
    p = os.path.join(results_dir or CFG.RESULTS_DIR, "pools", f"{rec['pool']}_logits.npz")
    z = np.load(p)
    Z = {k[2:]: z[k] for k in z.files if k.startswith("Z_")}
    return Z, z["y"], z["masks"].astype(bool)


def _subset(recs, name):
    return [r for r in recs if r["domain"] in SUBSETS[name]]


def _cr(y, cl, x=None):
    """Cluster-robust mean (or regression) as a flat dict; None if not estimable."""
    y, cl = np.asarray(y, float), np.asarray(cl)
    if len(y) < 2 or len(np.unique(cl)) < 2:
        return None
    r = cluster_robust_ols(y, cl, x=x)
    i = 0 if x is None else slice(None)
    out = {"n": int(len(y)), "G": int(len(np.unique(cl))), "df": int(r["df"])}
    for k in ("estimate", "se", "p", "ci_low", "ci_high"):
        v = np.atleast_1d(r[k])
        out[k] = [float(t) for t in v] if x is not None else float(v[0])
    return out


def _tost(y, cl, margins=(1.0, 0.5, 0.25)):
    if len(y) < 2 or len(np.unique(cl)) < 2:
        return None
    res = {m: tost_equivalence(np.asarray(y, float), np.asarray(cl), margin=m) for m in margins}
    cleared = [m for m in margins if res[m]["equivalent"]]
    return {"tightest_margin_cleared": min(cleared) if cleared else None,
            **{f"p_tost_{m}": float(res[m]["p_tost"]) for m in margins}}


# ---------------------------------------------------------------- consistency
def check_reconstruction(recs, results_dir=None):
    """Recompute every candidate's per-fold primary-rule utility from the saved
    logits and assert it equals the registered record."""
    worst = 0.0
    for r in recs:
        Z, y, masks = load_logits(r, results_dir)
        base = list(r["base"])
        reg = r["utility_per_fold"]["mean_logits"]
        for c, ufold in reg.items():
            for f, m in enumerate(masks):
                acc_b = (fusion.fuse([Z[b][m] for b in base], "mean_logits").argmax(1) == y[m]).mean() * 100
                acc_c = (fusion.fuse([Z[b][m] for b in base + [c]], "mean_logits").argmax(1) == y[m]).mean() * 100
                worst = max(worst, abs((acc_c - acc_b) - ufold[f]))
    assert worst < 1e-9, f"saved logits do not reproduce the registered utilities (max diff {worst})"
    return worst


# ---------------------------------------------------------------- R1, R2, R3, R9
def _pairs(recs, cats=("matched",), caliper=None):
    out = []
    for r in recs:
        for p in r["pairs"]:
            if p["category"] not in cats or not np.isfinite(p["diff"]):
                continue
            if caliper is not None and p["comp_gap"] > caliper + 1e-12:
                continue
            q = dict(p)
            q["signed_gap"] = p["acc_high"] - p["acc_low"]          # >0: higher-CKA is more accurate
            q["anchor_acc"] = r["indiv_acc"][r["anchor"]]
            q["domain"] = r["domain"]
            out.append(q)
    return out


def _primary(pairs):
    best = {}
    for p in pairs:
        if p["pool"] not in best or p["cka_gap"] > best[p["pool"]]["cka_gap"]:
            best[p["pool"]] = p
    return list(best.values())


def r1_competence_gap(recs):
    out = {"note": EXPLORATORY}
    for s in SUBSETS:
        P = _pairs(_subset(recs, s))
        if not P:
            continue
        g = np.array([p["signed_gap"] for p in P]); d = np.array([p["diff"] for p in P])
        cl = np.array([p["pool"] for p in P])
        prim = _primary(P)
        gp = np.array([p["signed_gap"] for p in prim]); dp = np.array([p["diff"] for p in prim])
        clp = np.array([p["pool"] for p in prim])
        adj = _cr(d, cl, x=g)
        adjp = _cr(dp, clp, x=gp) if len(prim) > 2 else None
        out[s] = {
            "pairs": len(P),
            "signed_gap_mean": float(g.mean()), "signed_gap_sd": float(g.std(ddof=1)) if len(g) > 1 else None,
            "frac_higher_cka_more_accurate": float((g > 0).mean()),
            "signed_gap_cluster_robust": _cr(g, cl),
            "D_adjusted_all_pairs": None if adj is None else
                {"intercept_D_at_zero_gap": adj["estimate"][0], "ci": [adj["ci_low"][0], adj["ci_high"][0]],
                 "p": adj["p"][0], "slope_pp_per_gap_point": adj["estimate"][1], "slope_p": adj["p"][1],
                 "G": adj["G"], "n": adj["n"]},
            "D_adjusted_primary": None if adjp is None else
                {"intercept_D_at_zero_gap": adjp["estimate"][0], "ci": [adjp["ci_low"][0], adjp["ci_high"][0]],
                 "p": adjp["p"][0], "G": adjp["G"]},
        }
    return out


def r2_caliper(recs, calipers=(0.5, 1.0, 1.5, 3.0)):
    out = {"note": EXPLORATORY + "; 1.5 is the registered caliper",
           "eligible": "pairs with CKA gap >= 0.15 and separated by > 3 median subsampling SDs"}
    for s in SUBSETS:
        out[s] = {}
        for c in calipers:
            P = _pairs(_subset(recs, s), cats=("matched", "unmatched"), caliper=c)
            if not P:
                out[s][str(c)] = None
                continue
            d = [p["diff"] for p in P]; cl = [p["pool"] for p in P]
            prim = _primary(P)
            out[s][str(c)] = {"all_pairs": _cr(d, cl), "primary": _cr([p["diff"] for p in prim],
                                                                     [p["pool"] for p in prim]),
                              "tost_all_pairs": _tost(d, cl)}
    return out


def r3_strong(recs, within=3.0):
    out = {"note": EXPLORATORY, "rule": f"both candidates within {within} points of the anchor's accuracy"}
    for s in SUBSETS:
        P = [p for p in _pairs(_subset(recs, s))
             if p["anchor_acc"] - min(p["acc_high"], p["acc_low"]) <= within]
        if not P:
            out[s] = None
            continue
        d = [p["diff"] for p in P]; cl = [p["pool"] for p in P]
        out[s] = {"all_pairs": _cr(d, cl), "tost_all_pairs": _tost(d, cl),
                  "pairs": len(P), "pools": len(set(cl)),
                  "mean_utility_of_members": float(np.mean(
                      [u for r in _subset(recs, s) for c in r["candidates"]
                       for u in [c["utility"]] if any(c["name"] in (p["high"], p["low"]) and p["pool"] == r["pool"]
                                                      for p in P)]))}
    # how many matched pairs have both candidates weaker than the base fusion
    allp = _pairs(recs)
    weak = 0
    for p in allp:
        r = next(r for r in recs if r["pool"] == p["pool"])
        base_acc = float(np.mean(r["base_fold_acc"]["mean_logits"]))
        weak += (p["acc_high"] < base_acc) and (p["acc_low"] < base_acc)
    out["matched_pairs_both_weaker_than_base_fusion"] = {"count": int(weak), "of": len(allp)}
    return out


def r9_unmatched(recs):
    out = {"note": EXPLORATORY}
    for s in SUBSETS:
        U = _pairs(_subset(recs, s), cats=("unmatched",))
        M = _pairs(_subset(recs, s))
        if not U:
            out[s] = None
            continue
        d = np.array([p["diff"] for p in U]); g = np.array([p["signed_gap"] for p in U])
        cl = np.array([p["pool"] for p in U])
        reg = _cr(d, cl, x=g)
        out[s] = {"matched_all_pairs": _cr([p["diff"] for p in M], [p["pool"] for p in M]) if M else None,
                  "unmatched_all_pairs": _cr(d, cl),
                  "unmatched_frac_higher_cka_more_accurate": float((g > 0).mean()),
                  "unmatched_D_on_signed_gap": None if reg is None else
                      {"slope_pp_per_point": reg["estimate"][1], "p": reg["p"][1],
                       "intercept": reg["estimate"][0], "intercept_p": reg["p"][0]},
                  "unmatched_pairs": int(len(U))}
    return out


# ---------------------------------------------------------------- R4
def _two_way(y, c1, c2):
    """Intercept-only mean with Cameron-Gelbach-Miller two-way clustered SE."""
    y = np.asarray(y, float)
    e = y - y.mean()
    n = len(y)

    def V(cl):
        cl = np.asarray(cl)
        u = np.unique(cl)
        G = len(u)
        s = sum(e[cl == g].sum() ** 2 for g in u)
        return G / max(G - 1, 1) * s / n ** 2, G

    v1, g1 = V(c1)
    v2, g2 = V(c2)
    v12, _ = V([f"{a}||{b}" for a, b in zip(c1, c2)])
    var = max(v1 + v2 - v12, v1, v2)          # guard against a negative CGM variance
    se = np.sqrt(var)
    df = min(g1, g2) - 1
    tc = stats.t.ppf(0.975, df)
    m = float(y.mean())
    return {"estimate": m, "se": float(se), "ci_low": m - tc * se, "ci_high": m + tc * se,
            "p": float(2 * stats.t.sf(abs(m / se), df)) if se > 0 else None,
            "G_pool": g1, "G_encoder_pair": g2, "df": df}


def r4_dependence(recs):
    out = {"note": EXPLORATORY}
    for s in SUBSETS:
        P = _pairs(_subset(recs, s))
        if len(P) < 3:
            out[s] = None
            continue
        y = [p["diff"] for p in P]
        pool = [p["pool"] for p in P]
        pair = ["|".join(sorted((p["high"], p["low"]))) for p in P]
        enc = sorted({e for p in P for e in (p["high"], p["low"])})
        loo = []
        for e in enc:
            Q = [p for p in P if e not in (p["high"], p["low"])]
            r = _cr([p["diff"] for p in Q], [p["pool"] for p in Q])
            if r:
                loo.append({"dropped_encoder": e, **{k: r[k] for k in ("estimate", "ci_low", "ci_high", "p", "n", "G")}})
        prim = _primary(P)
        out[s] = {"pool_clustered": _cr(y, pool),
                  "two_way_pool_x_encoder_pair": _two_way(y, pool, pair),
                  "encoder_pair_clustered": _cr(y, pair),
                  "primary_two_way": _two_way([p["diff"] for p in prim], [p["pool"] for p in prim],
                                              ["|".join(sorted((p["high"], p["low"]))) for p in prim])
                  if len(prim) > 2 else None,
                  "leave_one_encoder_out": loo,
                  "loo_range": [min(l["estimate"] for l in loo), max(l["estimate"] for l in loo)] if loo else None,
                  "loo_all_ci_contain_zero": all(l["ci_low"] <= 0 <= l["ci_high"] for l in loo) if loo else None}
    return out


# ---------------------------------------------------------------- R5
RULE_DESCR = {
    "lowest_cka": "lowest CKA with the anchor (label-free; the registered heuristic)",
    "competence": "highest accuracy",
    "unique": "largest unique contribution over the base",
    "disagreement": "highest disagreement with the base",
    "gated_diversity": "lowest CKA among candidates within 1.5 points of the most accurate",
    "rank_sum": "best sum of accuracy rank and (inverse) CKA rank; ties to accuracy",
    "random": "uniform random candidate (expected value)",
}


def _select(rule, names, acc, cka, uniq, dis):
    # Registered rules break ties exactly as the registered analysis does: the
    # first candidate in record order wins (Python's min/max on the value alone).
    if rule == "lowest_cka":
        return min(names, key=lambda c: cka[c])
    if rule == "competence":
        return max(names, key=lambda c: acc[c])
    if rule == "unique":
        return max(names, key=lambda c: uniq[c])
    if rule == "disagreement":
        return max(names, key=lambda c: dis[c])
    if rule == "gated_diversity":
        top = max(acc.values())
        ok = [c for c in names if acc[c] >= top - 1.5]
        return min(ok, key=lambda c: (cka[c], -acc[c]))
    if rule == "rank_sum":
        ra = {c: i for i, c in enumerate(sorted(names, key=lambda c: -acc[c]))}
        rc = {c: i for i, c in enumerate(sorted(names, key=lambda c: cka[c]))}
        return min(names, key=lambda c: (ra[c] + rc[c], ra[c]))
    raise KeyError(rule)


def r5_regret_crossfit(recs, results_dir=None):
    """For each pool and evaluation fold f, every selection statistic that uses
    labels (accuracy, unique contribution, disagreement) is computed on the
    held-out rows of the OTHER folds only; the chosen candidate is scored by its
    utility on fold f. CKA uses no labels and is used as registered.

    Caveat stated in the output: each fold's held-out logits come from a probe
    trained on the other folds, so selection statistics on folds != f still depend
    on fold-f labels through those probes. Removing that requires refitting probes
    in a nested design (GPU); this analysis removes the direct reuse."""
    rules = [r for r in RULE_DESCR if r != "random"]
    per_pool = []
    for r in recs:
        Z, y, masks = load_logits(r, results_dir)
        base = list(r["base"])
        cands = [c["name"] for c in r["candidates"]]
        U = {c: np.array(r["utility_per_fold"]["mean_logits"][c]) for c in cands}
        oracle = max(U[c].mean() for c in cands)
        cka = {c: r["cka_with_anchor"][c] for c in cands}
        row = {"pool": r["pool"], "domain": r["domain"], "oracle_mean_utility": float(oracle),
               "random_regret": float(oracle - np.mean([U[c].mean() for c in cands]))}
        for rule in rules:
            got_cf, got_in = [], []
            for f, m in enumerate(masks):
                tr = ~m
                pred = {c: Z[c].argmax(1) for c in cands + base}
                basep = fusion.fuse([Z[b] for b in base], "mean_logits").argmax(1)
                base_any = np.zeros(len(y), bool)
                for b in base:
                    base_any |= pred[b] == y
                def stats_on(rows):
                    acc = {c: float((pred[c][rows] == y[rows]).mean() * 100) for c in cands}
                    uniq = {c: float(((pred[c][rows] == y[rows]) & ~base_any[rows]).mean() * 100) for c in cands}
                    dis = {c: float((pred[c][rows] != basep[rows]).mean()) for c in cands}
                    return acc, uniq, dis
                a_cf, u_cf, d_cf = stats_on(tr)
                got_cf.append(U[_select(rule, cands, a_cf, cka, u_cf, d_cf)][f])
            # in-sample version, as registered (statistics on all rows)
            a_in = {c: r["indiv_acc"][c] for c in cands}
            u_in = {c: next(x["headroom_contrib"] for x in r["candidates"] if x["name"] == c) for c in cands}
            d_in = {c: next(x["disagreement"] for x in r["candidates"] if x["name"] == c) for c in cands}
            pick = _select(rule, cands, a_in, cka, u_in, d_in)
            row[f"{rule}_regret_crossfit"] = float(oracle - np.mean(got_cf))
            row[f"{rule}_regret_insample"] = float(oracle - U[pick].mean())
        per_pool.append(row)
    bp = os.path.join(results_dir or CFG.RESULTS_DIR, "battery.json")
    if os.path.exists(bp):
        bat = json.load(open(bp))
        regname = {"lowest_cka": "cka_diversity", "competence": "competence",
                   "unique": "oracle_headroom", "disagreement": "disagreement_with_base"}
        for s in SUBSETS:
            reg = (bat.get(s, {}).get("D_selection_regret") or {}).get("per_rule", {})
            rows = [p for p in per_pool if p["domain"] in SUBSETS[s]]
            for mine, theirs in regname.items():
                if theirs in reg and rows:
                    v = float(np.mean([p[f"{mine}_regret_insample"] for p in rows]))
                    assert abs(v - reg[theirs]["mean_regret_pp"]) < 1e-9, \
                        f"in-sample {mine} regret {v} != registered {reg[theirs]['mean_regret_pp']} ({s})"
    out = {"note": EXPLORATORY,
           "caveat": " ".join(r5_regret_crossfit.__doc__.split("Caveat stated in the output: ")[1].split()),
           "registered_rules_reproduced_in_sample": os.path.exists(bp),
           "rules": RULE_DESCR, "per_pool": per_pool}
    for s in SUBSETS:
        rows = [p for p in per_pool if p["domain"] in SUBSETS[s]]
        if not rows:
            continue
        summ = {"pools": len(rows), "random_regret": float(np.mean([p["random_regret"] for p in rows]))}
        for rule in rules:
            for kind in ("crossfit", "insample"):
                v = np.array([p[f"{rule}_regret_{kind}"] for p in rows])
                summ[f"{rule}_{kind}"] = float(v.mean())
            dcf = np.array([p[f"lowest_cka_regret_crossfit"] - p[f"{rule}_regret_crossfit"] for p in rows])
            if rule != "lowest_cka" and len(rows) > 1:
                t = stats.ttest_1samp(dcf, 0.0)
                summ[f"{rule}_vs_lowest_cka_crossfit"] = {
                    "mean_regret_saved_pp": float(dcf.mean()), "p_t": float(t.pvalue),
                    "rule_better_in": int((dcf > 1e-9).sum()), "worse_in": int((dcf < -1e-9).sum())}
        out[s] = summ
    return out


# ---------------------------------------------------------------- R6 (Algorithm 1)
def _alg1(names, C, u, th_cka, th_u):
    keep = set(names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a in keep and b in keep and C[(a, b)] > th_cka:
                weak = a if u[a] < u[b] else b
                if u[weak] < th_u:
                    keep.discard(weak)
    return sorted(keep)


def r6_algorithm1(recs, sim, results_dir=None, th_ckas=(0.6, 0.7, 0.8, 0.9), th_us=(0.5, 1.0, 2.0)):
    """Algorithm 1 (paper appendix) applied to every pool's full member set,
    cross-fitted: unique contributions and accuracies for fold f are computed on
    the other folds' held-out rows; CKA (label-free) from the feature matrices.
    Compared against the size-matched subset of the most accurate members and
    against fusing all members. Fusion: the registered primary rule."""
    rows = []
    for r in recs:
        if r["pool"] not in sim:
            continue
        Z, y, masks = load_logits(r, results_dir)
        names = sorted(Z)
        C = sim[r["pool"]]["pairwise_linear_cka"]
        for thc in th_ckas:
            for thu in th_us:
                acc_alg, acc_top, acc_all, sizes = [], [], [], []
                for f, m in enumerate(masks):
                    tr = ~m
                    pred = {c: Z[c].argmax(1) for c in names}
                    corr = {c: pred[c][tr] == y[tr] for c in names}
                    others_any = {c: np.zeros(tr.sum(), bool) for c in names}
                    for c in names:
                        for o in names:
                            if o != c:
                                others_any[c] |= corr[o]
                    u = {c: float((corr[c] & ~others_any[c]).mean() * 100) for c in names}
                    acc = {c: float(corr[c].mean() * 100) for c in names}
                    S = _alg1(names, C, u, thc, thu)
                    k = len(S)
                    T = sorted(names, key=lambda c: -acc[c])[:k]
                    score = lambda M: float((fusion.fuse([Z[c][m] for c in M], "mean_logits").argmax(1)
                                             == y[m]).mean() * 100)
                    acc_alg.append(score(S)); acc_top.append(score(T)); acc_all.append(score(names))
                    sizes.append(k)
                rows.append({"pool": r["pool"], "domain": r["domain"], "theta_cka": thc, "theta_unique": thu,
                             "mean_size": float(np.mean(sizes)), "n_members": len(names),
                             "acc_alg1": float(np.mean(acc_alg)), "acc_topk_competence": float(np.mean(acc_top)),
                             "acc_all_members": float(np.mean(acc_all))})
    out = {"note": EXPLORATORY, "per_pool": rows, "summary": []}
    for thc in th_ckas:
        for thu in th_us:
            R = [x for x in rows if x["theta_cka"] == thc and x["theta_unique"] == thu]
            if len(R) < 2:
                continue
            d = np.array([x["acc_alg1"] - x["acc_topk_competence"] for x in R])
            t = stats.ttest_1samp(d, 0.0) if np.any(d != 0) else None
            out["summary"].append({"theta_cka": thc, "theta_unique": thu, "pools": len(R),
                                   "mean_size": float(np.mean([x["mean_size"] for x in R])),
                                   "alg1_minus_topk_pp": float(d.mean()),
                                   "p_t": float(t.pvalue) if t is not None else None,
                                   "alg1_worse_in": int((d < -1e-9).sum()), "alg1_better_in": int((d > 1e-9).sum()),
                                   "alg1_minus_all_pp": float(np.mean([x["acc_alg1"] - x["acc_all_members"] for x in R]))})
    return out


# ---------------------------------------------------------------- R7 (similarity measures)
def _center(X):
    return X - X.mean(0, keepdims=True)


def _lin_cka(X, Y):
    X, Y = _center(X), _center(Y)
    num = np.linalg.norm(Y.T @ X) ** 2
    return float(num / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def _unbiased_hsic(K, L):
    n = K.shape[0]
    K = K.copy(); L = L.copy()
    np.fill_diagonal(K, 0); np.fill_diagonal(L, 0)
    t1 = np.sum(K * L)
    t2 = K.sum() * L.sum() / ((n - 1) * (n - 2))
    t3 = 2 * (K.sum(0) @ L.sum(0)) / (n - 2)
    return (t1 + t2 - t3) / (n * (n - 3))


def _debiased_cka(X, Y):
    X, Y = _center(X), _center(Y)
    K, L = X @ X.T, Y @ Y.T
    kl = _unbiased_hsic(K, L)
    return float(kl / np.sqrt(max(_unbiased_hsic(K, K), 1e-300) * max(_unbiased_hsic(L, L), 1e-300)))


def _rbf_cka(X, Y):
    def gram(A):
        sq = np.sum(A * A, 1)
        D = np.maximum(sq[:, None] + sq[None, :] - 2 * A @ A.T, 0)
        sig2 = np.median(D[np.triu_indices_from(D, 1)])
        return np.exp(-D / max(sig2, 1e-12))
    K, L = gram(X), gram(Y)
    n = K.shape[0]
    H = np.eye(n) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    return float(np.sum(Kc * Lc) / (np.linalg.norm(Kc) * np.linalg.norm(Lc)))


def _mutual_knn(X, Y, k=10):
    from sklearn.neighbors import NearestNeighbors
    def nn(A):
        A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
        idx = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A, return_distance=False)
        return idx[:, 1:]
    a, b = nn(X), nn(Y)
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(a))]))


def _unit(X):
    X = _center(np.asarray(X, np.float64))
    return X / max(np.linalg.norm(X), 1e-12)


def similarity_measures(recs, n_sub=3000, n_sub_rbf=2000, results_dir=None, verbose=True):
    """Per pool and candidate: several similarity measures to the anchor and to the
    base ensemble, computed on one fixed subsample of rows (seed 0)."""
    import extract as X_
    out = {}
    for r in recs:
        ds, t0 = r["pool"], time.time()
        man = json.load(open(os.path.join(CFG.MANIFEST_DIR, f"{ds}.json")))
        names = sorted(r["indiv_acc"])
        n = man["n"]
        idx = np.sort(np.random.default_rng(0).choice(n, min(n_sub, n), replace=False))
        idx_r = idx[np.sort(np.random.default_rng(1).choice(len(idx), min(n_sub_rbf, len(idx)), replace=False))]
        F = {}
        for e in names:
            p = X_._paths(ds, e)[1]
            if not os.path.exists(p):
                break
            F[e] = np.asarray(np.load(p)[idx], np.float64)    # one sequential read; scattered reads over Drive are slow
        if len(F) != len(names):
            if verbose:
                print(f"  [{ds}] features missing; skipped")
            continue
        Z, y, masks = load_logits(r, results_dir)
        anchor, base = r["anchor"], list(r["base"])
        Fbase = np.concatenate([_unit(F[b]) for b in base], 1)
        Zbase = np.concatenate([_unit(Z[b]) for b in base], 1)
        pos = {e: i for i, e in enumerate(idx)}
        row_r = np.array([pos[i] for i in idx_r])
        C = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                C[(a, b)] = C[(b, a)] = _lin_cka(F[a], F[b])
        cand = {}
        for c in [x["name"] for x in r["candidates"]]:
            cand[c] = {
                "linear_cka_anchor_subsample": C[(c, anchor)],
                "linear_cka_base": _lin_cka(F[c], Fbase),
                "debiased_cka_anchor": _debiased_cka(F[c], F[anchor]),
                "rbf_cka_anchor": _rbf_cka(F[c][row_r], F[anchor][row_r]),
                "mutual_knn_anchor": _mutual_knn(F[c], F[anchor]),
                "logit_cka_anchor": _lin_cka(Z[c], Z[anchor]),
                "logit_cka_base": _lin_cka(Z[c], Zbase),
            }
        out[ds] = {"pairwise_linear_cka": C, "candidates": cand, "n_rows": int(len(idx)),
                   "n_rows_rbf": int(len(idx_r))}
        if verbose:
            print(f"  [{ds}] {len(cand)} candidates, {len(idx)} rows: {time.time() - t0:.0f}s", flush=True)
    return out


MEASURES = ["linear_cka_registered", "linear_cka_base", "debiased_cka_anchor", "rbf_cka_anchor",
            "mutual_knn_anchor", "logit_cka_anchor", "logit_cka_base"]


def r7_similarity(recs, sim):
    rows = []
    for r in recs:
        if r["pool"] not in sim:
            continue
        for c in r["candidates"]:
            m = dict(sim[r["pool"]]["candidates"][c["name"]])
            m["linear_cka_registered"] = r["cka_with_anchor"][c["name"]]
            rows.append({"pool": r["pool"], "domain": r["domain"], "name": c["name"],
                         "utility": c["utility"], "indiv_acc": c["indiv_acc"], **m})
    out = {"note": EXPLORATORY, "candidates": rows}
    for s in SUBSETS:
        R = [x for x in rows if x["domain"] in SUBSETS[s]]
        if len({x["pool"] for x in R}) < 2:
            continue
        cl = np.array([x["pool"] for x in R])
        u = np.array([x["utility"] for x in R])
        acc = np.array([x["indiv_acc"] for x in R])
        # within-pool centring of accuracy removes pool-level differences in difficulty
        accc = acc - np.array([acc[cl == p].mean() for p in cl])
        res = {}
        for meas in MEASURES:
            v = np.array([x[meas] for x in R])
            vc = v - np.array([v[cl == p].mean() for p in cl])
            reg = cluster_robust_ols(u - np.array([u[cl == p].mean() for p in cl]), cl,
                                     x=np.column_stack([vc, accc]))
            # regret of choosing the least similar candidate by this measure
            reg_regret = []
            for p in sorted(set(cl)):
                Rp = [x for x in R if x["pool"] == p]
                best = max(x["utility"] for x in Rp)
                reg_regret.append(best - min(Rp, key=lambda x: x[meas])["utility"])
            res[meas] = {"coef_pp_per_unit": float(reg["estimate"][1]), "p": float(reg["p"][1]),
                         "acc_coef": float(reg["estimate"][2]), "acc_p": float(reg["p"][2]),
                         "spearman_with_utility_within_pool": float(stats.spearmanr(vc, u - np.array(
                             [u[cl == p].mean() for p in cl])).correlation),
                         "regret_choose_least_similar_pp": float(np.mean(reg_regret)),
                         "G": int(len(set(cl))), "n": int(len(R))}
        # matched pairs re-oriented by each measure
        P = _pairs(_subset(recs, s))
        for meas in MEASURES:
            ds = []
            for p in P:
                if p["pool"] not in sim:
                    continue
                look = {x["name"]: x for x in R if x["pool"] == p["pool"]}
                if p["high"] not in look or p["low"] not in look:
                    continue
                a, b = look[p["high"]][meas], look[p["low"]][meas]
                ds.append((p["pool"], p["diff"] if a >= b else -p["diff"]))
            if len({d[0] for d in ds}) >= 2:
                res[meas]["matched_D_oriented_by_measure"] = _cr([d[1] for d in ds], [d[0] for d in ds])
        out[s] = res
    return out


# ---------------------------------------------------------------- R8
def r8_metrics(recs, results_dir=None):
    from sklearn.metrics import f1_score
    out = {"note": EXPLORATORY, "per_pool_primary": []}
    pairD = {"macro_f1": [], "nll": [], "accuracy_check": []}
    for r in recs:
        Z, y, masks = load_logits(r, results_dir)
        base = list(r["base"])
        def metr(M, m):
            zl = fusion.fuse([Z[c][m] for c in M], "mean_logits")
            pr = fusion.fuse([Z[c][m] for c in M], "mean_probs")
            f1 = f1_score(y[m], zl.argmax(1), average="macro", labels=np.unique(y)) * 100
            nll = -np.mean(np.log(np.clip(pr[np.arange(m.sum()), y[m]], 1e-12, 1)))
            acc = (zl.argmax(1) == y[m]).mean() * 100
            return f1, nll, acc
        cache = {}
        def util(c):
            if c not in cache:
                vals = []
                for f, m in enumerate(masks):
                    fb, nb, ab = metr(base, m)
                    fc, nc, ac = metr(base + [c], m)
                    vals.append((fc - fb, nb - nc, ac - ab))      # NLL utility: reduction in NLL
                cache[c] = np.array(vals)
            return cache[c]
        for p in r["pairs"]:
            if p["category"] != "matched" or not np.isfinite(p["diff"]):
                continue
            dh, dl = util(p["high"]), util(p["low"])
            dacc = (dh[:, 2] - dl[:, 2]).mean()
            assert abs(dacc - p["diff"]) < 1e-9, f"{r['pool']}: accuracy D not reproduced"
            pairD["macro_f1"].append((r["pool"], r["domain"], (dh[:, 0] - dl[:, 0]).mean(), p.get("primary")))
            pairD["nll"].append((r["pool"], r["domain"], (dh[:, 1] - dl[:, 1]).mean(), p.get("primary")))
            if p.get("primary"):
                fd = np.array(p["fold_diffs"])
                tc = stats.t.ppf(0.975, len(fd) - 1)
                se = fd.std(ddof=1) / np.sqrt(len(fd))
                out["per_pool_primary"].append({"pool": r["pool"], "domain": r["domain"], "D": float(fd.mean()),
                                                "fold_ci_low": float(fd.mean() - tc * se),
                                                "fold_ci_high": float(fd.mean() + tc * se),
                                                "high": p["high"], "low": p["low"]})
    for s in SUBSETS:
        out[s] = {}
        for k in ("macro_f1", "nll"):
            rows = [x for x in pairD[k] if x[1] in SUBSETS[s]]
            prim = [x for x in rows if x[3]]
            out[s][k] = {"all_pairs": _cr([x[2] for x in rows], [x[0] for x in rows]),
                         "primary": _cr([x[2] for x in prim], [x[0] for x in prim])}
    out["sign_note"] = ("D = metric gain of the higher-CKA minus the lower-CKA candidate; for NLL the gain is "
                        "the reduction in NLL (positive = better), so all metrics share the sign convention")
    return out


# ---------------------------------------------------------------- accuracy table
def accuracy_table(recs):
    """Held-out linear-probe accuracy of every encoder in every pool, for comparison
    with published linear-probe results (protocols differ; see the paper)."""
    rows = []
    for r in recs:
        for e, a in sorted(r["indiv_acc"].items(), key=lambda t: -t[1]):
            rows.append({"pool": r["pool"], "domain": r["domain"], "encoder": e, "accuracy": a,
                         "anchor": e == r["anchor"]})
    return rows


# ---------------------------------------------------------------- driver
def run_all(results_dir=None, with_features=True, verbose=True):
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # zero-variance subsets
    rd = results_dir or CFG.RESULTS_DIR
    recs = load(rd)
    t0 = time.time()
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "status": EXPLORATORY,
           "pools": [r["pool"] for r in recs]}
    out["reconstruction_max_abs_diff"] = check_reconstruction(recs, rd)
    if verbose:
        print("registered utilities reproduced from the saved logits (max diff "
              f"{out['reconstruction_max_abs_diff']:.1e})", flush=True)
    for key, fn in (("R1_competence_gap", r1_competence_gap), ("R2_caliper", r2_caliper),
                    ("R3_strong", r3_strong), ("R4_dependence", r4_dependence),
                    ("R9_unmatched", r9_unmatched)):
        out[key] = fn(recs)
        if verbose:
            print(f"{key}: done", flush=True)
    out["R5_regret_crossfit"] = r5_regret_crossfit(recs, rd)
    if verbose:
        print("R5_regret_crossfit: done", flush=True)
    out["R8_metrics"] = r8_metrics(recs, rd)
    if verbose:
        print("R8_metrics: done", flush=True)
    out["accuracy_table"] = accuracy_table(recs)
    if with_features:
        sim = similarity_measures(recs, results_dir=rd, verbose=verbose)
        out["R7_similarity"] = r7_similarity(recs, sim)
        out["R6_algorithm1"] = r6_algorithm1(recs, sim, rd)
        if verbose:
            print("R6, R7: done", flush=True)
    out["wall_minutes"] = round((time.time() - t0) / 60, 1)
    return out
