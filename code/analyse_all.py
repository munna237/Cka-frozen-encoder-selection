"""Run the pre-registered analysis over every pool and write the paper's numbers.

    results/pools/<dataset>.json      full per-pool record (run_pipeline.run_pool)
    results/pools/<dataset>_logits.npz  held-out logits, labels, fold masks
    results/battery.json              estimands A-F for each pre-registered subset
    results/summary.md                human-readable tables
    results/figures/*.png
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

import config as CFG
import extract as X
import run_pipeline as RP

SUBSETS = {
    "all": ("pathology", "vision", "text"),
    "pathology": ("pathology",),
    "vision": ("vision",),
    "text": ("text",),
    "image": ("pathology", "vision"),
}


def _manifest(ds):
    with open(os.path.join(CFG.MANIFEST_DIR, f"{ds}.json")) as fh:
        return json.load(fh)


def load_pool(ds, domain):
    man = _manifest(ds)
    encs = CFG.DOMAINS[domain][1]
    feats, missing = {}, []
    for e in encs:
        if X.is_done(ds, e, man):
            feats[e] = np.load(X._paths(ds, e)[1]).astype(np.float64)
        else:
            missing.append(e)
    groups = None
    if CFG.PREREG["grouped_folds_where_groups_exist"] and man.get("groups"):
        groups = np.asarray(man["groups"])
    return feats, np.asarray(man["labels"]), groups, man, missing


def _feature_key(ds, feats, man, prereg_hash):
    h = hashlib.sha256()
    h.update(man["hash"].encode())
    h.update(prereg_hash.encode())
    for e in sorted(feats):
        with open(X._paths(ds, e)[2]) as fh:
            h.update(json.load(fh).get("saved", e).encode())
        h.update(e.encode())
    return h.hexdigest()[:16]


def run_all(prereg_status, domains=("pathology", "vision", "text"), force=False):
    P = CFG.PREREG
    out_dir = os.path.join(CFG.RESULTS_DIR, "pools")
    os.makedirs(out_dir, exist_ok=True)
    records, skipped = [], []
    for domain in domains:
        for ds in CFG.DOMAINS[domain][0]:
            jp = os.path.join(CFG.MANIFEST_DIR, f"{ds}.json")
            if not os.path.exists(jp):
                skipped.append({"dataset": ds, "reason": "no manifest (not extracted)"})
                continue
            feats, y, groups, man, missing = load_pool(ds, domain)
            if len(feats) < P["min_encoders_per_pool"]:
                skipped.append({"dataset": ds, "reason": f"only {len(feats)} encoders "
                                f"(< {P['min_encoders_per_pool']}); missing {missing}"})
                continue
            key = _feature_key(ds, feats, man, prereg_status["registration_hash"])
            rp = os.path.join(out_dir, f"{ds}.json")
            if os.path.exists(rp) and not force:
                with open(rp) as fh:
                    rec = json.load(fh)
                if rec.get("feature_key") == key:
                    print(f"[{ds}] cached result ({len(feats)} encoders)")
                    records.append(rec)
                    continue
            print(f"\n[{ds}] {domain}: {len(feats)} encoders, n={len(y)}, "
                  f"{'grouped' if groups is not None else 'ungrouped'} folds"
                  + (f"; missing {missing}" if missing else ""), flush=True)
            t0 = time.time()
            rec = RP.run_pool(ds, feats, y, groups=groups, k=P["k_folds"], seed=P["seed"],
                              n_cap_cka=P["n_cap_cka"], comp_tol=P["comp_tol_pp"],
                              cka_min=P["cka_min_gap"], cka_sd_floor=P["cka_sd_floor"],
                              fusion_rules=tuple([P["primary_fusion_rule"]] +
                                                 P["secondary_fusion_rules"]),
                              probe_backend=P["probe_backend"], return_logits=True)
            np.savez_compressed(os.path.join(out_dir, f"{ds}_logits.npz"),
                                y=rec.pop("_y"), masks=rec.pop("_masks"),
                                **{f"Z_{k}": v for k, v in rec.pop("_logits").items()})
            rec.update(domain=domain, feature_key=key, missing_encoders=missing,
                       manifest_hash=man["hash"], prereg=prereg_status)
            RP.dump(rec, rp)
            print(f"[{ds}] done in {(time.time() - t0) / 60:.1f} min: anchor={rec['anchor']} "
                  f"base={rec['base']} matched pairs={len(rec['matched_pairs'])}", flush=True)
            with open(rp) as fh:                       # use exactly what was saved
                records.append(json.load(fh))
    return records, skipped


def battery(records, prereg_status):
    out = {"prereg": prereg_status, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    for name, doms in SUBSETS.items():
        recs = [r for r in records if r["domain"] in doms]
        if len(recs) < 2:
            out[name] = {"error": f"only {len(recs)} pools"}
            continue
        try:
            res = RP.analyse_pools(recs, margins=tuple(CFG.PREREG["equivalence_margins_pp"]))
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {e}"}
        res["pools"] = [r["pool"] for r in recs]
        out[name] = res
    return out


def _f(v, nd=2):
    try:
        return f"{float(v):+.{nd}f}"
    except Exception:
        return "n/a"


def _p(v):
    try:
        v = float(v)
        return f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"
    except Exception:
        return "n/a"


def summary_md(records, bat, skipped):
    L = ["# v3 results summary", "",
         f"Pre-registration: {bat['prereg']['registered_utc']} "
         f"(hash {bat['prereg']['registration_hash'][:12]}); deviations: "
         f"{bat['prereg']['deviations'] or 'none'}", ""]
    L += ["## Pools", "", "| pool | domain | n | encoders | grouped | anchor | base | matched pairs | primary D (pp) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        prim = [p for p in r["matched_pairs"] if p.get("primary")]
        d = _f(prim[0]["diff"]) if prim else "-"
        L.append(f"| {r['pool']} | {r['domain']} | {r['n_inputs']} | {r['n_members']} | "
                 f"{r['grouped_folds']} | {r['anchor']} | {r['base'][1]} | "
                 f"{len(r['matched_pairs'])} | {d} |")
    if skipped:
        L += ["", "Skipped: " + "; ".join(f"{s['dataset']} ({s['reason']})" for s in skipped)]
    for name in SUBSETS:
        b = bat.get(name, {})
        L += ["", f"## Subset: {name}", ""]
        if "error" in b:
            L.append(f"_{b['error']}_")
            continue
        ne = [k for k, v in b.items() if isinstance(v, dict) and "not_estimable" in v]
        if ne:
            L.append("_Not estimable in this subset: " + "; ".join(
                f"{k} ({b[k]['not_estimable']})" for k in ne) + "_")
            L.append("")
        a = (b.get("A_matched_null_primary_only") or {}).get("cluster_robust", {})
        aa = (b.get("A_matched_null_all_pairs") or {}).get("cluster_robust", {})
        L.append(f"- **A (primary, one pair per pool)**: D = {_f(a.get('estimate_pp'))} pp, "
                 f"95% CI [{_f(a.get('ci_low'))}, {_f(a.get('ci_high'))}], p = {_p(a.get('p'))}, "
                 f"G = {a.get('G')}")
        L.append(f"- A (all matched pairs, clustered): D = {_f(aa.get('estimate_pp'))} pp, "
                 f"95% CI [{_f(aa.get('ci_low'))}, {_f(aa.get('ci_high'))}], "
                 f"pairs = {aa.get('n')}, G = {aa.get('G')}")
        B = b.get("B_dose_response") or {}
        B = B if "slope_pp_per_unit_cka" in B else {}
        L.append(f"- B (dose-response): slope {_f(B.get('slope_pp_per_unit_cka'))} pp/unit CKA, "
                 f"SE {_f(B.get('se'))}, p = {_p(B.get('p'))}, MDE80 = {_f(B.get('mde_80pct_power'))}; "
                 f"interpretable: {B.get('mde_80pct_power', 99) <= 0.5 if B else 'n/a'}")
        C = b.get("C_equivalence") or {}
        L.append(f"- C (TOST, all matched pairs): largest margin cleared = "
                 f"{C.get('largest_margin_cleared')} pp (G = {C.get('G')})")
        D = (b.get("D_selection_regret") or {}).get("per_rule", {})
        if D:
            L += ["", "| criterion | mean regret (pp) | median | p90 | top-1 agreement | pools |",
                  "|---|---|---|---|---|---|"]
            for k, v in D.items():
                L.append(f"| {k} | {v['mean_regret_pp']:.2f} | {v['median_regret_pp']:.2f} | "
                         f"{v['p90_regret_pp']:.2f} | {v['top1_agreement_with_oracle']:.0%} | {v['n_pools']} |")
        Em = b.get("E_mechanism_on_real") or {}
        if "joint_model" in Em:
            j = Em["joint_model"]
            L.append("")
            L.append(f"- E (joint model, cluster-robust): CKA {_f(j['cka']['coef_pp_per_unit'])} "
                     f"(p = {_p(j['cka']['p'])}), disagreement {_f(j['disagreement']['coef_pp_per_unit'])} "
                     f"(p = {_p(j['disagreement']['p'])}), accuracy {_f(j['indiv_acc']['coef_pp_per_unit'], 3)} "
                     f"(p = {_p(j['indiv_acc']['p'])}); candidates = {Em['n_candidates']}, G = {Em['G']}")
    return "\n".join(L) + "\n"


def figures(records, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    colors = {"pathology": "#c0392b", "vision": "#2471a3", "text": "#239b56"}
    # 1. dose-response
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in records:
        xs = [p["cka_gap"] for p in r["matched_pairs"]]
        ys = [p["diff"] for p in r["matched_pairs"]]
        ax.scatter(xs, ys, s=10, alpha=0.5, color=colors[r["domain"]])
    for d, c in colors.items():
        ax.scatter([], [], color=c, label=d)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("CKA gap (higher-CKA minus lower-CKA candidate)")
    ax.set_ylabel("utility difference D (pp)")
    ax.legend(frameon=False)
    ax.set_title("Matched pairs: does D grow with the CKA gap?")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dose_response.png"), dpi=200)
    plt.close(fig)
    # 2. candidate utility vs CKA and vs competence (within-pool centred)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for r in records:
        c = r["candidates"]
        u = np.array([x["utility"] for x in c])
        k = np.array([x["cka"] for x in c])
        a = np.array([x["indiv_acc"] for x in c])
        axs[0].scatter(k, u - u.mean(), s=12, alpha=0.6, color=colors[r["domain"]])
        axs[1].scatter(a - a.mean(), u - u.mean(), s=12, alpha=0.6, color=colors[r["domain"]])
    axs[0].set_xlabel("CKA with anchor")
    axs[1].set_xlabel("individual accuracy (pp, centred within pool)")
    axs[0].set_ylabel("marginal utility (pp, centred within pool)")
    for ax in axs:
        ax.axhline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "utility_vs_cka_and_competence.png"), dpi=200)
    plt.close(fig)


def write_all(records, bat, skipped):
    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)
    RP.dump(bat, os.path.join(CFG.RESULTS_DIR, "battery.json"))
    md = summary_md(records, bat, skipped)
    with open(os.path.join(CFG.RESULTS_DIR, "summary.md"), "w") as fh:
        fh.write(md)
    figures(records, os.path.join(CFG.RESULTS_DIR, "figures"))
    return md


def verify_determinism(records, bat):
    """Recompute the battery from the saved pool JSONs and assert it matches."""
    again = battery(records, bat["prereg"])
    a = json.dumps({k: v for k, v in bat.items() if k not in ("generated",)},
                   sort_keys=True, default=float)
    b = json.dumps({k: v for k, v in again.items() if k not in ("generated",)},
                   sort_keys=True, default=float)
    ok = a == b
    print("battery recomputation from saved pool records:", "IDENTICAL" if ok else "DIFFERS")
    return ok
