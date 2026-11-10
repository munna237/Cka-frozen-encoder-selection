"""End-to-end pipeline over one pool.

Input : a dict of frozen feature matrices {member_name: (n, d)}, labels y,
        optional groups (patient/slide/institution) for grouped folds.
Output: a single JSON-serialisable record containing everything the analysis
        battery in analysis.py needs, plus every intermediate quantity so that
        verify_claims.py can assert any published number against a fresh run.

Deliberately CPU-only and dependency-light (numpy/scipy/sklearn): feature
extraction is the only GPU step and it is a separate script. That split is what
makes this runnable inside a Colab session that disconnects, and testable in CI.
"""

from __future__ import annotations

import json
import time

import numpy as np

import analysis
import folds as F
import fusion
import pairs as P
from cka import cka_subsampling_sd, linear_cka, pairwise_cka_matrix


def run_pool(pool_name: str, feats: dict, y: np.ndarray, groups=None,
             k: int = 5, seed: int = 0, n_cap_cka: int = 10000,
             n_cap_analysis: int | None = None,
             comp_tol: float = 1.5, cka_min: float = 0.15, cka_sd_floor: float = 3.0,
             fusion_rules=("mean_logits", "mean_probs", "class_conditional"),
             probe_backend: str = "torch", verbose: bool = True,
             return_logits: bool = False) -> dict:
    """Run the whole pipeline for one pool.

    n_cap_cka      : rows used for the pairwise CKA matrix. Linear CKA costs
                     O(n*di*dj), so 276 pairs at n=30000 is minutes on CPU; the
                     measured subsampling SD is 0.003-0.006, far below any gap we
                     interpret, so capping is a free saving.
    n_cap_analysis : rows used for probes, fusion and accuracy. Capping here
                     changes absolute accuracies slightly but NOT the matched-pair
                     differences, which is the estimand. Set it to keep a 24-member
                     pool inside Colab RAM and a session.
    """
    t0 = time.time()
    names = sorted(feats)
    y = np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)

    # Apply the analysis cap FIRST and to everything, so features, labels, groups
    # and all downstream quantities stay row-aligned. A fixed seed makes it
    # reproducible; the cap is recorded in the output.
    if n_cap_analysis is not None and y.shape[0] > n_cap_analysis:
        n_orig = y.shape[0]
        rng_cap = np.random.default_rng(seed)
        keep = np.sort(rng_cap.choice(n_orig, size=n_cap_analysis, replace=False))
        feats = {m: np.asarray(v)[keep] for m, v in feats.items()}
        y = y[keep]
        if groups is not None:
            groups = groups[keep]
        print(f"[cap] analysis subsampled {n_orig} -> {len(keep)} rows (seed {seed})")
    n = y.shape[0]

    # ---- 1. folds (grouped when group labels exist) --------------------
    masks = F.make_folds(y, groups=groups, k=k, seed=seed)

    # ---- 2. held-out logits per member --------------------------------
    Z = {}
    for i_m, m in enumerate(names):
        t_m = time.time()
        Z[m] = F.per_fold_logits(feats[m], y, masks, backend=probe_backend)
        if verbose:
            print(f"  [{pool_name}] probe {i_m + 1}/{len(names)} {m:18s} "
                  f"acc {F.accuracy(Z[m], y) * 100:6.2f}%  ({time.time() - t_m:5.1f}s)",
                  flush=True)
    indiv_acc = {m: F.accuracy(Z[m], y) * 100.0 for m in names}

    # ---- 3. CKA -------------------------------------------------------
    C, cka_names = pairwise_cka_matrix(feats, n_cap=n_cap_cka, seed=seed)
    pos = {nm: i for i, nm in enumerate(cka_names)}
    cka_full = {}
    for a in names:
        for b in names:
            cka_full[(a, b)] = cka_full[(b, a)] = float(C[pos[a], pos[b]])
    mean_cka = {m: float(np.mean([cka_full[(m, o)] for o in names if o != m]))
                for m in names}
    anchor = P.choose_anchor(mean_cka, indiv_acc)
    cka_with_anchor = {m: (0.0 if m == anchor else cka_full[(m, anchor)]) for m in names}

    # CKA measurement noise: the check whose absence let a 0.012 gap be treated
    # as meaningful in v2. Computed on a subsample, so it is cheap.
    idx = np.sort(np.random.default_rng(seed).choice(n, size=min(4000, n), replace=False))
    cka_sd = {}
    for m in names:
        if m == anchor:
            continue
        cka_sd[m] = cka_subsampling_sd(feats[m][idx], feats[anchor][idx],
                                       n_draws=20, n_sub=min(2000, len(idx)), seed=seed)
    # BUG GUARD: a zero or non-finite SD would silently disable the separation
    # check (median([]) is nan, and nan comparisons are always False, so EVERY
    # pair would pass as separated). Fall back to v2's measured subsampling SD of
    # 0.003-0.006 and record that a fallback was used.
    finite_sds = [v for v in cka_sd.values() if np.isfinite(v) and v > 1e-9]
    if finite_sds:
        cka_sd_used = float(np.median(finite_sds))
        cka_sd_fallback = False
    else:
        cka_sd_used = 0.005
        cka_sd_fallback = True
    cka_tol = cka_sd_floor * cka_sd_used

    # ---- 4. base, pairs ----------------------------------------------
    base = P.fix_base(anchor, cka_with_anchor)
    base_names = list(base)
    specs = P.enumerate_pairs(pool_name, cka_with_anchor, indiv_acc,
                              comp_tol=comp_tol, cka_min=cka_min,
                              anchor=anchor, cka_tol=cka_tol)

    # ---- 5. utility of every member w.r.t. the fixed base -------------
    # Computed per fold and per rule.
    #
    # COST NOTE (this is the expensive step, and it is worth understanding why):
    # for each fusion rule we fit base + every candidate, per fold. The base fit
    # is IDENTICAL across candidates, so it is cached -- without that cache the
    # base was refit once per candidate, which for a 24-member pool means ~22
    # redundant fits per fold per rule. On Colab that is the difference between
    # minutes and hours per pool.
    util: dict[str, dict[str, np.ndarray]] = {}
    base_acc_per_rule_fold: dict[str, np.ndarray] = {}
    for rule in fusion_rules:
        base_fold_acc = np.zeros(k)
        base_models: list = [None] * k
        for f, m in enumerate(masks):
            Ztr = [Z[b][~m] for b in base_names]
            Zva = [Z[b][m] for b in base_names]
            if rule == "class_conditional":
                model = fusion.IdentifiedClassConditionalFusion(seed=seed).fit(Ztr, y[~m])
                base_models[f] = model                      # cached, reused below
                out = model.predict_logits(Zva)
            else:
                out = fusion.fuse(Zva, rule)
            base_fold_acc[f] = (np.asarray(out).argmax(1) == y[m]).mean() * 100.0
        base_acc_per_rule_fold[rule] = base_fold_acc

        cand_fold_acc = {}
        for c in names:
            if c in base_names:
                continue
            acc = np.zeros(k)
            for f, m in enumerate(masks):
                members = base_names + [c]
                Ztr = [Z[b][~m] for b in members]
                Zva = [Z[b][m] for b in members]
                if rule == "class_conditional":
                    model = fusion.IdentifiedClassConditionalFusion(seed=seed).fit(Ztr, y[~m])
                    out = model.predict_logits(Zva)
                else:
                    out = fusion.fuse(Zva, rule)
                acc[f] = (np.asarray(out).argmax(1) == y[m]).mean() * 100.0
            cand_fold_acc[c] = acc
        util[rule] = {c: cand_fold_acc[c] - base_fold_acc for c in cand_fold_acc}

    primary_rule = "mean_logits" if "mean_logits" in fusion_rules else fusion_rules[0]

    # ---- 6. prediction disagreement with the base -----------------------
    # Uses held-out probe predictions (probes are trained on labelled folds),
    # so this is NOT a label-free criterion. CKA is the only label-free one.
    base_pred = fusion.fuse([Z[b] for b in base_names], primary_rule).argmax(1)
    disagreement = {c: float((F.preds_from_logits(Z[c]) != base_pred).mean())
                    for c in names if c not in base_names}

    # ---- 7. oracle headroom contribution ------------------------------
    base_correct = np.zeros(n, dtype=bool)
    for b in base_names:
        base_correct |= (F.preds_from_logits(Z[b]) == y)
    headroom_contrib = {}
    for c in names:
        if c in base_names:
            continue
        c_correct = (F.preds_from_logits(Z[c]) == y)
        # fraction of inputs the candidate uniquely rescues over the base
        headroom_contrib[c] = float((c_correct & ~base_correct).mean() * 100.0)
    oracle_acc = float(np.mean(base_correct | np.logical_or.reduce(
        [(F.preds_from_logits(Z[c]) == y) for c in names if c not in base_names]
        or [base_correct])) * 100.0)

    # ---- 8. per-pair estimands ----------------------------------------
    pair_records = []
    for s in specs:
        rec = s.to_dict()
        rec["cka_sd_high"] = cka_sd.get(s.high)
        rec["cka_sd_low"] = cka_sd.get(s.low)
        for rule in fusion_rules:
            uh = util[rule].get(s.high)
            ul = util[rule].get(s.low)
            if uh is None or ul is None:
                continue
            rec[f"fold_diffs_{rule}"] = (uh - ul).tolist()
            rec[f"diff_{rule}"] = float(np.mean(uh - ul))
        rec["fold_diffs"] = rec.get(f"fold_diffs_{primary_rule}", [])
        rec["diff"] = rec.get(f"diff_{primary_rule}", float("nan"))
        rec["fusion_rule"] = primary_rule
        pair_records.append(rec)

    prim = P.primary_pair(specs)
    for rec in pair_records:
        rec["primary"] = bool(prim is not None and rec["high"] == prim.high
                              and rec["low"] == prim.low)

    matched = [r for r in pair_records if r["category"] == "matched"
               and np.isfinite(r["diff"])]

    # ---- 9. candidate-level records for the mechanism analysis ---------
    cand_records = []
    for c in names:
        if c in base_names:
            continue
        cand_records.append({
            "pool": pool_name, "name": c,
            "cka": cka_with_anchor[c],
            "indiv_acc": indiv_acc[c],
            "disagreement": disagreement[c],
            "headroom_contrib": headroom_contrib[c],
            "utility": float(np.mean(util[primary_rule][c])),
            "utility_per_fold": util[primary_rule][c].tolist(),
        })

    out = {
        "pool": pool_name,
        "n_members": len(names), "n_inputs": n, "k_folds": k, "seed": seed,
        "n_cap_cka": n_cap_cka, "n_cap_analysis": n_cap_analysis,
        "grouped_folds": groups is not None,
        "probe_backend": probe_backend,
        "anchor": anchor, "base": base_names,
        "indiv_acc": indiv_acc,
        "cka_with_anchor": cka_with_anchor,
        "cka_sd_with_anchor": cka_sd,
        "cka_tolerance_used": cka_tol,
        "cka_sd_fallback_used": cka_sd_fallback,
        "cka_sd_median": cka_sd_used,
        "oracle_acc": oracle_acc,
        "base_fold_acc": {r: v.tolist() for r, v in base_acc_per_rule_fold.items()},
        "utility_per_fold": {r: {c: v.tolist() for c, v in util[r].items()}
                             for r in util},
        "pair_specs_summary": P.summarise_pool(specs),
        "pairs": pair_records,
        "matched_pairs": matched,
        "candidates": cand_records,
        "primary_fusion_rule": primary_rule,
        "fusion_rules": list(fusion_rules),
        "wall_seconds": round(time.time() - t0, 1),
    }
    if return_logits:
        # not JSON-serialisable on purpose: the caller saves these with np.savez
        out["_logits"] = Z
        out["_y"] = y
        out["_masks"] = np.stack(masks)
    return out


def analyse_pools(pool_records: list[dict], margins=(1.0, 0.5, 0.25)) -> dict:
    """Run the full battery across pools. This is what produces the paper's numbers."""
    matched = []
    for rec in pool_records:
        matched.extend(rec["matched_pairs"])
    cands = []
    for rec in pool_records:
        cands.extend(rec["candidates"])

    primary = [m for m in matched if m.get("primary")]
    pools_for_regret = {}
    for rec in pool_records:
        pools_for_regret[rec["pool"]] = rec["candidates"]

    def safe(name, fn, *args, need=None, **kw):
        """One estimand failing (e.g. too few pools with matched pairs for a
        cluster-robust SE) must not take down the others. It is reported as
        not estimable, with the reason, instead."""
        if need is not None and not need:
            return {"not_estimable": "insufficient data for this estimand"}
        try:
            return fn(*args, **kw)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as e:
            return {"not_estimable": f"{type(e).__name__}: {e}"}

    out = {
        "n_matched_pairs_all": len(matched),
        "n_matched_pairs_primary": len(primary),
        "G_pools": len({m["pool"] for m in matched}),
        "A_matched_null_all_pairs": safe("A_all", analysis.matched_null, matched, need=matched),
        "A_matched_null_primary_only": safe("A_primary", analysis.matched_null, primary, need=primary),
        "B_dose_response": safe("B", analysis.dose_response, matched, need=len(matched) >= 3),
        "C_equivalence": safe("C", analysis.equivalence, matched, margins=margins, need=matched),
        "D_selection_regret": safe("D", analysis.selection_regret, pools_for_regret),
        "E_mechanism_on_real": safe("E", analysis.mechanism_on_real, cands, need=len(cands) >= 5),
        "F_matched_band": safe("F", analysis.matched_band_comparison, cands, need=len(cands) >= 5),
    }
    return out


def dump(obj, path):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)
