# ==========================================================================
# Section 6: general-vision replication with a single fixed base per benchmark.
#
# This script needs per-encoder frozen-feature caches (<dataset>_<encoder>.npz,
# each holding F and y), built by general_vision_extract.py. Those caches are
# several GB and are NOT included in this archive; set CACHE to wherever you
# build them. Everything else in the archive is self-contained.
#
# The probe below deliberately omits multi_class='multinomial': all six
# general-vision benchmarks are multiclass (C >= 10), where the multinomial and
# default parameterisations coincide. The pathology pipeline in core.py sets it
# explicitly because MHIST is binary, where they do not.
# ==========================================================================

import os, json, itertools
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats

CACHE = os.environ.get("GV_CACHE", "./gv_cache")
DRIVE_CACHE = os.environ.get("GV_CACHE_ALT", "./gv_cache_alt")

def _resolve(ds, k):
    for root in (CACHE, DRIVE_CACHE):
        p = f"{root}/{ds}_{k}.npz"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"{ds}_{k}.npz not found in {CACHE} or {DRIVE_CACHE}. "
        "Build it with general_vision_extract.py, or set GV_CACHE.")
ENC = ["dinov2_l", "eva02_clip", "swin_l", "convnext_l", "resnet50", "vit_mae"]
ANCHOR = "dinov2_l"
DATASETS = ["dtd", "aircraft", "eurosat"]
SEED = 0
torch.set_num_threads(2)

def load(ds, k):
    z = np.load(_resolve(ds, k))
    return z["F"], z["y"]

def cka(X, Y):
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    X -= X.mean(0, keepdims=True); Y -= Y.mean(0, keepdims=True)
    d = np.linalg.norm(X.T@X, "fro") * np.linalg.norm(Y.T@Y, "fro")
    return float(np.linalg.norm(Y.T@X, "fro")**2 / d) if d else 0.0

def probe(F, y, sp):
    Z = np.zeros((len(y), len(np.unique(y))))
    for tr, va in sp:
        sc = StandardScaler().fit(F[tr])
        c = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(F[tr]), y[tr])
        d = c.decision_function(sc.transform(F[va]))
        Z[va] = np.column_stack([-d, d]) if d.ndim == 1 else d
    return Z

def softmax(x, ax=-1):
    x = x - x.max(ax, keepdims=True); e = np.exp(x); return e / e.sum(ax, keepdims=True)

def fuse(Zs, y, sp, C):
    """Parameter-free logit averaging: no fitted parameters, so the
    comparison cannot depend on a learned rule."""
    out = sum(Zs) / len(Zs)
    return out

results = {}
for ds in DATASETS:
    feats, y = {}, None
    for k in ENC:
        F, yy = load(ds, k); feats[k] = F
        if y is None: y = yy
    C = len(np.unique(y))
    sp = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(feats[ANCHOR], y))
    Z = {k: probe(feats[k], y, sp) for k in ENC}
    acc = {k: float((Z[k].argmax(1) == y).mean()*100) for k in ENC}
    ck = {k: cka(feats[ANCHOR], feats[k]) for k in ENC if k != ANCHOR}

    # ---- ONE fixed base per benchmark: the anchor plus the median-CKA encoder
    others = sorted([k for k in ENC if k != ANCHOR], key=lambda k: ck[k])
    BASE = others[len(others)//2]
    cands = [k for k in ENC if k not in (ANCHOR, BASE)]
    base_out = fuse([Z[ANCHOR], Z[BASE]], y, sp, C)
    b_acc = float((base_out.argmax(1) == y).mean()*100)
    print(f"\n{ds.upper()}  fixed base = {ANCHOR}+{BASE} ({b_acc:.2f}%)  "
          f"candidates: {cands}")

    marg = {}
    for k in cands:
        o = fuse([Z[ANCHOR], Z[BASE], Z[k]], y, sp, C)
        marg[k] = np.array([float(((o[va].argmax(1)==y[va]).mean()
                                   - (base_out[va].argmax(1)==y[va]).mean())*100)
                            for _, va in sp])
        print(f"   {k:12s} CKA={ck[k]:.3f} acc={acc[k]:6.2f} marg={marg[k].mean():+.2f}")

    pairs = []
    for a, b in itertools.combinations(cands, 2):
        hi, lo = (a, b) if ck[a] > ck[b] else (b, a)
        pairs.append(dict(hi=hi, lo=lo, cka_gap=abs(ck[a]-ck[b]),
                          comp_gap=abs(acc[a]-acc[b]),
                          diff=(marg[hi]-marg[lo]).tolist()))
    results[ds] = dict(base=BASE, base_acc=b_acc, cka=ck, acc=acc, pairs=pairs)

print("\n" + "="*74)
print("FIXED-BASE CONTRAST")
print("="*74)
for tag, sel in [("controlled", lambda p: p["comp_gap"] <= 5 and p["cka_gap"] >= 0.15),
                 ("unmatched",  lambda p: p["comp_gap"] > 5)]:
    D = []
    for ds, r in results.items():
        ok = [p for p in r["pairs"] if sel(p)]
        if not ok: continue
        best = max(ok, key=lambda p: p["cka_gap"] if tag == "controlled" else p["comp_gap"])
        D.append(np.array(best["diff"]))
        print(f"  {tag:11s} {ds:9s} {best['hi']} vs {best['lo']}  "
              f"CKAgap {best['cka_gap']:.3f} compgap {best['comp_gap']:.2f}  "
              f"diff {np.mean(best['diff']):+.2f}")
    if D:
        a = np.concatenate(D); t, p = stats.ttest_1samp(a, 0)
        print(f"  -> POOLED {a.mean():+.3f} pp  wins {int((a>0).sum())}/{len(a)}  p={p:.4f}\n")

json.dump(results, open("general_vision_fixedbase_raw.json", "w"), indent=2, default=float)
print("written general_vision_fixedbase_raw.json")
