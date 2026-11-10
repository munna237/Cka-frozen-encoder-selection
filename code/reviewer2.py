"""Second-round exploratory analyses (all POST-REGISTRATION).

  S1 pilot_overlap_exclusion  Q1  registered estimands without the six pilot datasets
  S2 gated_rule               concern 3 / Q2  all rules tried; gated vs competence (paired);
                                   gate width 0.5 / 1 / 1.5 / 3 points
  S3 nested_regret            concern 4 / Q2  fully nested selection: probes for the selection
                                   statistics are refitted inside each outer training split
  S4 c_sensitivity            concern 9   probe C in {0.1, 1, 10} on the smallest pools
  S5 association_consistency  Q4  per-pool slopes and leave-one-pool-out for the weak association
  S6 multiplicity             concern 2   Benjamini-Hochberg and Holm over the exploratory p-values
  S7 primary_pairs            minor       signed gap and fold interval of every primary pair
  S8 encoder_validation       concern 9   ported weights vs official releases (Lunit, Kaiko),
                                   DINOv2 at 224 px vs the official model

Reads the same saved artefacts as reviewer.py; S3, S4 and S8 additionally refit
linear probes or load weights, on CPU (GPU if present). Results carry the same
EXPLORATORY label.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
from scipy import stats

import config as CFG
import folds as FO
import fusion
import reviewer as RV
from inference import cluster_robust_ols, tost_equivalence

EXPLORATORY = RV.EXPLORATORY
PILOT_DATASETS = ("bracs", "sicapv2", "mhist", "dtd", "aircraft", "eurosat")


def _paired(d):
    d = np.asarray(d, float)
    out = {"n": int(len(d)), "mean": float(d.mean()), "better_in": int((d > 1e-9).sum()),
           "worse_in": int((d < -1e-9).sum()), "tie": int((np.abs(d) <= 1e-9).sum())}
    if len(d) > 1 and np.any(d != 0):
        t = stats.ttest_1samp(d, 0.0)
        tc = stats.t.ppf(0.975, len(d) - 1)
        se = d.std(ddof=1) / np.sqrt(len(d))
        out.update(p_t=float(t.pvalue), ci_low=float(d.mean() - tc * se), ci_high=float(d.mean() + tc * se))
        try:
            out["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------- S1
def pilot_overlap_exclusion(recs):
    keep = [r for r in recs if r["pool"] not in PILOT_DATASETS]
    out = {"note": EXPLORATORY, "excluded": list(PILOT_DATASETS), "pools": [r["pool"] for r in keep]}
    for s in RV.SUBSETS:
        R = RV._subset(keep, s)
        P = RV._pairs(R)
        prim = RV._primary(P)
        regret = {}
        for r in R:
            c = r["candidates"]
            best = max(x["utility"] for x in c)
            for name, key, rev in (("lowest_cka", "cka", False), ("competence", "indiv_acc", True),
                                   ("unique", "headroom_contrib", True)):
                pick = (max if rev else min)(c, key=lambda x: x[key])
                regret.setdefault(name, []).append(best - pick["utility"])
        d = [p["diff"] for p in P]; cl = [p["pool"] for p in P]
        out[s] = {"primary": RV._cr([p["diff"] for p in prim], [p["pool"] for p in prim]),
                  "all_pairs": RV._cr(d, cl), "tost_all_pairs": RV._tost(d, cl),
                  "regret": {k: float(np.mean(v)) for k, v in regret.items()},
                  "pools": len(R)}
    return out


# ---------------------------------------------------------------- S2
def gated_rule(recs, rv_json, results_dir=None, gates=(0.5, 1.0, 1.5, 3.0)):
    """Cross-fitted regret of the gated rule for several gate widths, and paired
    comparisons with competence. Rules tried in the first exploratory round: the
    four registered rules, the gated rule (gate 1.5) and the rank-sum rule; that
    is the complete list, all fixed in code before that round was run."""
    rows = rv_json["R5_regret_crossfit"]["per_pool"]
    out = {"note": EXPLORATORY,
           "rules_tried_round1": ["lowest_cka", "competence", "unique", "disagreement",
                                  "gated_diversity (gate 1.5)", "rank_sum"],
           "paired_vs_competence_crossfit": {}}
    for s in RV.SUBSETS:
        R = [x for x in rows if x["domain"] in RV.SUBSETS[s]]
        if len(R) < 2:
            continue
        out["paired_vs_competence_crossfit"][s] = {
            rule: _paired([x["competence_regret_crossfit"] - x[f"{rule}_regret_crossfit"] for x in R])
            for rule in ("gated_diversity", "rank_sum", "unique")}
    # gate width sensitivity (cross-fitted, computed afresh)
    per = []
    for r in recs:
        Z, y, masks = RV.load_logits(r, results_dir)
        cands = [c["name"] for c in r["candidates"]]
        U = {c: np.array(r["utility_per_fold"]["mean_logits"][c]) for c in cands}
        oracle = max(U[c].mean() for c in cands)
        cka = {c: r["cka_with_anchor"][c] for c in cands}
        pred = {c: Z[c].argmax(1) for c in cands}
        row = {"pool": r["pool"], "domain": r["domain"]}
        for g in gates:
            got = []
            for f, m in enumerate(masks):
                acc = {c: float((pred[c][~m] == y[~m]).mean() * 100) for c in cands}
                top = max(acc.values())
                ok = [c for c in cands if acc[c] >= top - g]
                got.append(U[min(ok, key=lambda c: (cka[c], -acc[c]))][f])
            row[f"gate_{g}"] = float(oracle - np.mean(got))
        per.append(row)
    out["gate_width_crossfit"] = {}
    for s in RV.SUBSETS:
        R = [x for x in per if x["domain"] in RV.SUBSETS[s]]
        if R:
            out["gate_width_crossfit"][s] = {str(g): float(np.mean([x[f"gate_{g}"] for x in R])) for g in gates}
    # consistency check: gate 1.5 must reproduce round 1
    r1 = {x["pool"]: x["gated_diversity_regret_crossfit"] for x in rows}
    worst = max(abs(x["gate_1.5"] - r1[x["pool"]]) for x in per if x["pool"] in r1)
    assert worst < 1e-9, f"gate 1.5 does not reproduce round-1 gated regret ({worst})"
    out["per_pool_gate_width"] = per
    return out


# ---------------------------------------------------------------- S3
def _load_features(ds, names):
    import extract as X_
    F = {}
    for e in names:
        p = X_._paths(ds, e)[1]
        if not os.path.exists(p):
            return None
        F[e] = np.load(p)
    return F


def nested_regret(recs, results_dir=None, n_max=12500, inner_k=4, gates=(1.5,), verbose=True,
                  backend="torch"):
    """Fully nested selection. For outer fold f, every label-using selection
    statistic is computed from INNER held-out logits: probes refitted by inner
    k-fold CV on the outer training rows only, so no label of fold f is used
    anywhere in selection. The chosen candidate is scored by its registered
    utility on fold f. Pools with more than n_max inputs are skipped (cost)."""
    per = []
    for r in recs:
        if r["n_inputs"] > n_max:
            continue
        t0 = time.time()
        man = json.load(open(os.path.join(CFG.MANIFEST_DIR, f"{r['pool']}.json")))
        Z, y, masks = RV.load_logits(r, results_dir)
        base = list(r["base"])
        cands = [c["name"] for c in r["candidates"]]
        F = _load_features(r["pool"], cands + base)
        if F is None:
            if verbose:
                print(f"  [{r['pool']}] features missing; skipped")
            continue
        groups = np.asarray(man["groups"]) if (man.get("groups") and r["grouped_folds"]) else None
        U = {c: np.array(r["utility_per_fold"]["mean_logits"][c]) for c in cands}
        oracle = max(U[c].mean() for c in cands)
        cka = {c: r["cka_with_anchor"][c] for c in cands}
        picks = {k: [] for k in ("lowest_cka", "competence", "unique", "disagreement", "rank_sum",
                                 *[f"gated_{g}" for g in gates])}
        for f, m in enumerate(masks):
            R_ = np.where(~m)[0]
            yR = y[R_]
            gR = None if groups is None else groups[R_]
            im = FO.make_folds(yR, k=inner_k, groups=gR, seed=CFG.SEED + 100 + f)
            Zi = {e: FO.per_fold_logits(F[e][R_], yR, im, backend=backend) for e in cands + base}
            pred = {e: Zi[e].argmax(1) for e in Zi}
            bp = fusion.fuse([Zi[b] for b in base], "mean_logits").argmax(1)
            base_any = np.zeros(len(R_), bool)
            for b in base:
                base_any |= pred[b] == yR
            acc = {c: float((pred[c] == yR).mean() * 100) for c in cands}
            uniq = {c: float(((pred[c] == yR) & ~base_any).mean() * 100) for c in cands}
            dis = {c: float((pred[c] != bp).mean()) for c in cands}
            for k in picks:
                if k.startswith("gated_"):
                    g = float(k.split("_")[1]); top = max(acc.values())
                    ok = [c for c in cands if acc[c] >= top - g]
                    sel = min(ok, key=lambda c: (cka[c], -acc[c]))
                else:
                    sel = RV._select(k, cands, acc, cka, uniq, dis)
                picks[k].append(U[sel][f])
        row = {"pool": r["pool"], "domain": r["domain"], "n": r["n_inputs"],
               **{f"{k}_regret_nested": float(oracle - np.mean(v)) for k, v in picks.items()},
               "minutes": round((time.time() - t0) / 60, 1)}
        per.append(row)
        if verbose:
            print(f"  [{r['pool']}] nested selection done in {row['minutes']} min", flush=True)
    out = {"note": EXPLORATORY, "inner_k": inner_k, "n_max": n_max, "per_pool": per, "summary": {}}
    for s in RV.SUBSETS:
        R = [x for x in per if x["domain"] in RV.SUBSETS[s]]
        if not R:
            continue
        summ = {"pools": len(R)}
        for k in ("lowest_cka", "competence", "unique", "disagreement", "rank_sum", *[f"gated_{g}" for g in gates]):
            summ[k] = float(np.mean([x[f"{k}_regret_nested"] for x in R]))
        summ["gated_vs_competence"] = _paired([x["competence_regret_nested"] - x[f"gated_{gates[0]}_regret_nested"] for x in R])
        summ["competence_vs_lowest_cka"] = _paired([x["lowest_cka_regret_nested"] - x["competence_regret_nested"] for x in R])
        out["summary"][s] = summ
    return out


# ---------------------------------------------------------------- S4
def c_sensitivity(recs, pools=("dtd", "aircraft", "mhist"), Cs=(0.1, 1.0, 10.0), results_dir=None,
                  backend="torch", verbose=True):
    """Refit the per-fold probes with other C values on the registered folds and
    recompute every candidate's utility, the matched pairs' D (same pairs) and the
    regrets of the registered rules. C = 1 must reproduce the registered values."""
    out = {"note": EXPLORATORY, "pools": {}}
    for r in recs:
        if r["pool"] not in pools:
            continue
        Z, y, masks = RV.load_logits(r, results_dir)
        names = sorted(Z)
        F = _load_features(r["pool"], names)
        if F is None:
            continue
        base = list(r["base"]); cands = [c["name"] for c in r["candidates"]]
        res = {}
        for C in Cs:
            t0 = time.time()
            Zc = Z if C == 1.0 else {e: _logits_C(F[e], y, masks, C, backend) for e in names}
            acc = {e: float((Zc[e].argmax(1) == y).mean() * 100) for e in names}
            def util(c):
                v = []
                for m in masks:
                    ab = (fusion.fuse([Zc[b][m] for b in base], "mean_logits").argmax(1) == y[m]).mean()
                    ac = (fusion.fuse([Zc[b][m] for b in base + [c]], "mean_logits").argmax(1) == y[m]).mean()
                    v.append((ac - ab) * 100)
                return float(np.mean(v))
            u = {c: util(c) for c in cands}
            if C == 1.0:
                reg = {c["name"]: c["utility"] for c in r["candidates"]}
                assert max(abs(u[c] - reg[c]) for c in cands) < 1e-9, "C=1 must reproduce registered utilities"
            D = [u[p["high"]] - u[p["low"]] for p in r["pairs"] if p["category"] == "matched"]
            best = max(u.values())
            cka = {c: r["cka_with_anchor"][c] for c in cands}
            res[str(C)] = {"mean_indiv_acc": float(np.mean([acc[c] for c in cands])),
                           "matched_D_mean": float(np.mean(D)) if D else None, "matched_pairs": len(D),
                           "regret_lowest_cka": best - u[min(cands, key=lambda c: cka[c])],
                           "regret_competence": best - u[max(cands, key=lambda c: acc[c])],
                           "minutes": round((time.time() - t0) / 60, 1)}
            if verbose:
                print(f"  [{r['pool']}] C={C}: done in {res[str(C)]['minutes']} min", flush=True)
        out["pools"][r["pool"]] = res
    return out


def _logits_C(X, y, masks, C, backend):
    import folds as F_
    return F_.per_fold_logits(X, y, list(masks), C=C, backend=backend)


# ---------------------------------------------------------------- S5
def association_consistency(rv_json, measures=("linear_cka_registered", "linear_cka_base", "mutual_knn_anchor",
                                               "rbf_cka_anchor")):
    rows = rv_json["R7_similarity"]["candidates"]
    pools = sorted({x["pool"] for x in rows})
    out = {"note": EXPLORATORY, "per_pool_sign": {}, "leave_one_pool_out": {}}
    for meas in measures:
        signs, slopes = [], {}
        for p in pools:
            R = [x for x in rows if x["pool"] == p]
            if len(R) < 4:
                continue
            X = np.column_stack([np.ones(len(R)), [x[meas] for x in R], [x["indiv_acc"] for x in R]])
            b = np.linalg.lstsq(X, np.array([x["utility"] for x in R]), rcond=None)[0]
            slopes[p] = float(b[1]); signs.append(np.sign(b[1]))
        loo = []
        for drop in pools:
            R = [x for x in rows if x["pool"] != drop]
            cl = np.array([x["pool"] for x in R]); u = np.array([x["utility"] for x in R])
            v = np.array([x[meas] for x in R]); a = np.array([x["indiv_acc"] for x in R])
            cen = lambda z: z - np.array([z[cl == q].mean() for q in cl])
            reg = cluster_robust_ols(cen(u), cl, x=np.column_stack([cen(v), cen(a)]))
            loo.append({"dropped": drop, "coef": float(reg["estimate"][1]), "p": float(reg["p"][1])})
        out["per_pool_sign"][meas] = {"negative_in": int(sum(s < 0 for s in signs)),
                                      "positive_in": int(sum(s > 0 for s in signs)), "pools": len(signs),
                                      "per_pool_slope": slopes}
        out["leave_one_pool_out"][meas] = {"coef_range": [min(l["coef"] for l in loo), max(l["coef"] for l in loo)],
                                           "p_range": [min(l["p"] for l in loo), max(l["p"] for l in loo)],
                                           "most_influential": max(loo, key=lambda l: l["coef"])["dropped"],
                                           "rows": loo}
    return out


# ---------------------------------------------------------------- S6
def _bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def _holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, p[i] * (n - rank))); adj[i] = run
    return adj


def multiplicity(rv_json):
    """Families: (a) Table 4: the seven similarity measures over all pools;
    (b) every candidate-level coefficient p-value of R7 (7 measures x 5 subsets);
    (c) every exploratory interval-based p-value reported for D in R1-R4, R9."""
    out = {"note": EXPLORATORY + "; adjusted p-values are descriptive"}
    r7 = rv_json["R7_similarity"]
    fam_a = [(m, r7["all"][m]["p"]) for m in RV.MEASURES]
    fam_b = [(f"{s}/{m}", r7[s][m]["p"]) for s in RV.SUBSETS if s in r7 for m in RV.MEASURES]
    fam_c = []
    for s in RV.SUBSETS:
        a = (rv_json["R1_competence_gap"].get(s) or {}).get("D_adjusted_all_pairs")
        if a: fam_c.append((f"R1 gap-adjusted D/{s}", a["p"]))
        for c, v in (rv_json["R2_caliper"].get(s) or {}).items():
            if v and v.get("all_pairs"): fam_c.append((f"R2 caliper {c}/{s}", v["all_pairs"]["p"]))
        v = rv_json["R3_strong"].get(s)
        if v and v.get("all_pairs"): fam_c.append((f"R3 strong/{s}", v["all_pairs"]["p"]))
        v = rv_json["R4_dependence"].get(s)
        if v and v["two_way_pool_x_encoder_pair"].get("p") is not None:
            fam_c.append((f"R4 two-way/{s}", v["two_way_pool_x_encoder_pair"]["p"]))
    for name, fam in (("table4_all_pools", fam_a), ("R7_all_subsets", fam_b), ("D_robustness", fam_c)):
        fam = [(k, p) for k, p in fam if p is not None and np.isfinite(p)]
        ps = [p for _, p in fam]
        q, h = _bh(ps), _holm(ps)
        out[name] = [{"test": k, "p": float(p), "q_bh": float(qq), "p_holm": float(hh)}
                     for (k, p), qq, hh in zip(fam, q, h)]
        out[name + "_min_q"] = float(min(q)) if len(q) else None
    return out


# ---------------------------------------------------------------- S7
def primary_pairs(recs):
    rows = []
    for r in recs:
        for p in r["pairs"]:
            if p.get("primary"):
                fd = np.array(p["fold_diffs"]); tc = stats.t.ppf(0.975, len(fd) - 1)
                se = fd.std(ddof=1) / np.sqrt(len(fd))
                rows.append({"pool": r["pool"], "high": p["high"], "low": p["low"], "cka_gap": p["cka_gap"],
                             "signed_gap": p["acc_high"] - p["acc_low"], "D": p["diff"],
                             "fold_ci_low": float(fd.mean() - tc * se), "fold_ci_high": float(fd.mean() + tc * se)})
    return rows


# ---------------------------------------------------------------- S8
def _state(model):
    return {k: v.detach().float().cpu() for k, v in model.state_dict().items()}


def _compare_states(a, b):
    common = sorted(set(a) & set(b))
    same_shape = [k for k in common if a[k].shape == b[k].shape]
    diffs = [float((a[k] - b[k]).abs().max()) for k in same_shape]
    return {"tensors_a": len(a), "tensors_b": len(b), "common": len(common), "same_shape": len(same_shape),
            "max_abs_diff": max(diffs) if diffs else None,
            "only_in_a": sorted(set(a) - set(b))[:6], "only_in_b": sorted(set(b) - set(a))[:6]}


def _dinov2_official(name):
    """Meta's DINOv2 model code and weights. Tries torch.hub; if GitHub's API refuses
    (rate limit), downloads the repository as an archive and loads it locally."""
    import torch
    try:
        return torch.hub.load("facebookresearch/dinov2", name, trust_repo=True)
    except Exception:
        import io, urllib.request, zipfile
        root = os.path.join(os.environ.get("TORCH_HOME", "/tmp"), "dinov2_src")
        src = os.path.join(root, "dinov2-main")
        if not os.path.isdir(src):
            data = urllib.request.urlopen(
                "https://codeload.github.com/facebookresearch/dinov2/zip/refs/heads/main", timeout=120).read()
            zipfile.ZipFile(io.BytesIO(data)).extractall(root)
        return torch.hub.load(src, name, source="local", trust_repo=True)


def encoder_validation(images=None, device="cpu", verbose=True):
    """Ported weights vs official releases, tensor by tensor; and the DINOv2
    encoders as we run them (224 px, resampled position embedding) vs the official
    model on the same images (CKA of the class token)."""
    import torch
    import timm
    from cka import linear_cka
    import encoders as E
    out = {"note": "validation of encoder loading; not an analysis of the study"}
    # Lunit DINO ViT-S/16: official release vs the timm port we use
    try:
        url = ("https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/"
               "pretrained-weights/dino_vit_small_patch16_ep200.torch")
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
        sd = sd.get("state_dict", sd.get("model", sd)) if isinstance(sd, dict) else sd
        sd = {k.replace("module.", "").replace("backbone.", ""): v.float() for k, v in sd.items()}
        port = timm.create_model("hf-hub:1aurent/vit_small_patch16_224.lunit_dino", pretrained=True, num_classes=0)
        out["lunit_vits16"] = {"source": url, **_compare_states(sd, _state(port))}
    except Exception as e:
        out["lunit_vits16"] = {"error": f"{type(e).__name__}: {e}"}
    if verbose: print("  lunit:", out["lunit_vits16"].get("max_abs_diff", out["lunit_vits16"].get("error")), flush=True)
    # Kaiko ViT-B/16: official torch.hub entry vs the timm port we use
    try:
        # the official release file named by kaiko's own hubconf.py (RELEASE_TAG 0.0.1); loading the
        # file directly avoids torch.hub's GitHub API calls, which are rate-limited
        url = "https://github.com/kaiko-ai/towards_large_pathology_fms/releases/download/0.0.1/vitb16.pth"
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
        sd = sd.get("state_dict", sd.get("model", sd)) if isinstance(sd, dict) else sd
        sd = {k.replace("module.", ""): v.float() for k, v in sd.items() if hasattr(v, "float")}
        port = timm.create_model("hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
                                 pretrained=True, num_classes=0, dynamic_img_size=True)
        out["kaiko_vitb16"] = {"source": url, **_compare_states(sd, _state(port))}
    except Exception as e:
        out["kaiko_vitb16"] = {"error": f"{type(e).__name__}: {e}"}
    if verbose: print("  kaiko:", out["kaiko_vitb16"].get("max_abs_diff", out["kaiko_vitb16"].get("error")), flush=True)
    # DINOv2 weights: Meta's official release files vs timm's checkpoint at its native 518 px, tensor by
    # tensor (no GitHub API involved). The only change our 224-px run makes on top of these weights is
    # timm's bicubic resampling of the position embedding, checked on images below when the hub is reachable.
    for ours, tag, tid in (("dinov2_b", "vitb14", "vit_base_patch14_dinov2.lvd142m"),
                           ("dinov2_l", "vitl14", "vit_large_patch14_dinov2.lvd142m")):
        try:
            url = f"https://dl.fbaipublicfiles.com/dinov2/dinov2_{tag}/dinov2_{tag}_pretrain.pth"
            sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
            sd = {k: v.float() for k, v in sd.items() if k != "mask_token"}
            native = timm.create_model(tid, pretrained=True, num_classes=0)
            out[f"{ours}_weights_vs_official"] = {"source": url, **_compare_states(sd, _state(native))}
            del native
        except Exception as e:
            out[f"{ours}_weights_vs_official"] = {"error": f"{type(e).__name__}: {e}"}
        if verbose:
            x = out[f"{ours}_weights_vs_official"]
            print(f"  {ours} weights:", x.get("max_abs_diff", x.get("error")), flush=True)
    # DINOv2: our wrapper (timm, img_size=224) vs the official hub model on the same images
    if images is not None and len(images):
        for ours, hub in (("dinov2_b", "dinov2_vitb14"), ("dinov2_l", "dinov2_vitl14")):
            try:
                enc = E.build(ours, CFG.VISION_ENCODERS[ours], device)
                x = torch.from_numpy(np.ascontiguousarray(images))
                f_ours = enc(x)
                off = _dinov2_official(hub).eval().to(device)
                mean = torch.tensor(enc.mean, device=device).view(1, 3, 1, 1)
                std = torch.tensor(enc.std, device=device).view(1, 3, 1, 1)
                with torch.no_grad():
                    xi = (x.to(device).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
                    f_off = off(xi).float().cpu().numpy()            # class token, pos-embed interpolated
                out[f"{ours}_vs_official"] = {"n_images": int(len(images)),
                                              "cka": float(linear_cka(f_ours.astype(np.float64), f_off.astype(np.float64))),
                                              "mean_cosine": float(np.mean(np.sum(f_ours * f_off, 1) /
                                                     (np.linalg.norm(f_ours, axis=1) * np.linalg.norm(f_off, axis=1)))),
                                              "info": {k: enc.info.get(k) for k in ("resized_from", "dynamic_from")}}
                del enc, off
            except Exception as e:
                out[f"{ours}_vs_official"] = {"error": f"{type(e).__name__}: {e}"}
            if verbose: print(f"  {ours}:", out[f"{ours}_vs_official"].get("cka", out[f"{ours}_vs_official"].get("error")), flush=True)
    return out


# ---------------------------------------------------------------- driver
def run_all(rv_json_path=None, results_dir=None, nested=True, c_sens=True, validate=True, verbose=True):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    rd = results_dir or CFG.RESULTS_DIR
    recs = RV.load(rd)
    rvj = json.load(open(rv_json_path or os.path.join(rd, "reviewer", "reviewer_analyses.json")))
    RV.check_reconstruction(recs, rd)
    out = {"status": EXPLORATORY, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    out["S1_pilot_overlap_exclusion"] = pilot_overlap_exclusion(recs)
    out["S2_gated_rule"] = gated_rule(recs, rvj, rd)
    out["S5_association_consistency"] = association_consistency(rvj)
    out["S6_multiplicity"] = multiplicity(rvj)
    out["S7_primary_pairs"] = primary_pairs(recs)
    if verbose:
        print("S1, S2, S5, S6, S7: done", flush=True)
    if nested:
        out["S3_nested_regret"] = nested_regret(recs, rd, verbose=verbose)
    if c_sens:
        out["S4_c_sensitivity"] = c_sensitivity(recs, results_dir=rd, verbose=verbose)
    if validate:
        imgs = None
        try:
            import imgcache as IC
            arr, _ = IC.ensure_images("dtd")
            imgs = np.asarray(arr[:256])
        except Exception as e:
            print("  (DINOv2 comparison skipped: no DTD image cache:", e, ")")
        out["S8_encoder_validation"] = encoder_validation(imgs, verbose=verbose)
    return out
