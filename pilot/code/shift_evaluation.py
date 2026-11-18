# ==========================================================================
# R2 REQUEST 2b -- is the loss under patient-level shift a failure of the
# FUSION, or a failure of the fixed TEACHER SELECTION?
#
# The first run fixed the teacher set (Phikon+UNI+ResNet50), chosen under
# random 5-fold CV, and then evaluated it under patient-level shift. On every
# SICAPv2 fold the best individual teacher turned out to be CTransPath, which
# is not in that set, while ResNet50 -- which is -- collapsed from 76% to
# 61-70%. So the comparison confounds two things.
#
# This script separates them. Within each shift fold we evaluate:
#   FIXED      the paper's combination, selected in-distribution
#   ORACLE     the best 3-teacher combination on the test fold (upper bound)
#   RESELECT   teachers chosen on the TRAINING partition of that fold only,
#              by individual accuracy -- what a practitioner would actually do
#   ALL4       all four teachers fused, no selection at all
#
# RESELECT is the honest operational condition: no test information is used.
# ==========================================================================
import os, json, itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold

DRIVE = "/content/drive/MyDrive"
HET = os.path.join(DRIVE, "project_root")
TEACHERS = ["uni", "phikon_v2", "ctranspath", "resnet50_imagenet"]
PAPER_SET = ["phikon_v2", "uni", "resnet50_imagenet"]


def feat(ds, t):
    if t == "ctranspath":
        for fn in ["features.npy", "trainval_features.npy"]:
            p = os.path.join(HET, f"expanded/{ds}/ctranspath_fixed", fn)
            if os.path.exists(p): return np.load(p)
    for fn in ["all_features.npy", "trainval_features.npy"]:
        p = os.path.join(HET, f"cached_features_{ds}", t, fn)
        if os.path.exists(p): return np.load(p)
    raise FileNotFoundError(f"{ds}/{t}")


def labels(ds):
    for t in TEACHERS:
        for fn in ["all_labels.npy", "trainval_labels.npy"]:
            p = os.path.join(HET, f"cached_features_{ds}", t, fn)
            if os.path.exists(p): return np.load(p)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def probe(Ftr, ytr, Fte):
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    clf.fit(sc.transform(Ftr), ytr)
    d = clf.decision_function(sc.transform(Fte))
    return np.column_stack([-d, d]) if d.ndim == 1 else d


def fit_fusion(X, y, C, K, epochs=800, lr0=.05, wd=1e-3, lam=.01, patience=100):
    W = np.zeros((C, K)); W[:, 0] = .5
    tl = np.zeros(K); b = np.zeros(C)
    P = [W, tl, b]; m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]
    Y = np.eye(C)[y]; N = len(y); best = (1e9, None, 0)
    for ep in range(epochs):
        lr = lr0 * (1 + np.cos(np.pi * ep / epochs)) / 2
        temps = np.clip(np.exp(tl), .1, 10.)
        act = ((np.exp(tl) >= .1) & (np.exp(tl) <= 10.)).astype(float)
        sc_ = X / temps[None, :, None]; al = softmax(W, axis=1)
        G = (softmax(np.einsum("ck,nkc->nc", al, sc_) + b[None, :], 1) - Y) / N
        dA = np.einsum("nc,nkc->ck", G, sc_) + (lam / C) * (np.log(al + 1e-8) + 1)
        dW = al * (dA - (dA * al).sum(1, keepdims=True))
        dtl = -np.einsum("nc,ck,nkc->k", G, al, sc_) * act
        db = G.sum(0)
        for i, (p, g) in enumerate(zip(P, [dW, dtl, db])):
            p *= (1 - lr * wd)
            m[i] = .9*m[i] + .1*g; v[i] = .999*v[i] + .001*g*g
            p -= lr * (m[i]/(1-.9**(ep+1))) / (np.sqrt(v[i]/(1-.999**(ep+1)))+1e-8)
        W, tl, b = P
        t2 = np.clip(np.exp(tl), .1, 10.)
        o = np.einsum("ck,nkc->nc", softmax(W, 1), X/t2[None, :, None]) + b
        l = -np.log(softmax(o, 1)[np.arange(N), y] + 1e-12).mean()
        if l < best[0]: best = (l, (W.copy(), tl.copy(), b.copy()), ep)
        elif ep - best[2] >= patience: break
    W, tl, b = best[1]
    return W, np.clip(np.exp(tl), .1, 10.), b


def fuse(Ztr, ytr, Zte, combo, C):
    Xtr = np.stack([Ztr[t] for t in combo], 1)
    Xte = np.stack([Zte[t] for t in combo], 1)
    W, t2, b = fit_fusion(Xtr, ytr, C, len(combo))
    return np.einsum("ck,nkc->nc", softmax(W, 1), Xte/t2[None, :, None]) + b


def run(ds, folds, tag):
    y = labels(ds); C = len(np.unique(y))
    F = {t: feat(ds, t) for t in TEACHERS}
    rows = []
    for fi, (tri, tei) in enumerate(folds):
        Zte = {t: probe(F[t][tri], y[tri], F[t][tei]) for t in TEACHERS}
        Ztr = {t: probe(F[t][tri], y[tri], F[t][tri]) for t in TEACHERS}

        # inner CV on the TRAINING partition only -- no test information
        inner = list(StratifiedKFold(3, shuffle=True,
                                     random_state=0).split(F["uni"][tri], y[tri]))
        inner_acc = {}
        for t in TEACHERS:
            preds = np.zeros(len(tri), dtype=int)
            for a, b_ in inner:
                z = probe(F[t][tri][a], y[tri][a], F[t][tri][b_])
                preds[b_] = z.argmax(1)
            inner_acc[t] = float((preds == y[tri]).mean() * 100)

        te_acc = {t: float((Zte[t].argmax(1) == y[tei]).mean()*100) for t in TEACHERS}
        best_single = max(TEACHERS, key=lambda t: te_acc[t])
        reselect = sorted(TEACHERS, key=lambda t: -inner_acc[t])[:3]

        res = {"fold": fi + 1, "best_single": best_single,
               "best_single_acc": round(te_acc[best_single], 2),
               "inner_ranking": [f"{t}:{inner_acc[t]:.1f}" for t in
                                 sorted(TEACHERS, key=lambda t: -inner_acc[t])],
               "test_ranking": [f"{t}:{te_acc[t]:.1f}" for t in
                                sorted(TEACHERS, key=lambda t: -te_acc[t])]}

        for name, combo in [("FIXED", PAPER_SET), ("RESELECT", reselect),
                            ("ALL4", TEACHERS)]:
            o = fuse(Ztr, y[tri], Zte, combo, C)
            a = float((o.argmax(1) == y[tei]).mean()*100)
            res[name] = dict(combo=list(combo), acc=round(a, 2),
                             gain=round(a - te_acc[best_single], 2))

        oracle = None
        for combo in itertools.combinations(TEACHERS, 3):
            o = fuse(Ztr, y[tri], Zte, list(combo), C)
            a = float((o.argmax(1) == y[tei]).mean()*100)
            if oracle is None or a > oracle[1]: oracle = (list(combo), a)
        res["ORACLE"] = dict(combo=oracle[0], acc=round(oracle[1], 2),
                             gain=round(oracle[1] - te_acc[best_single], 2))

        print(f"\n  {ds} {tag} fold {fi+1}: best single {best_single} "
              f"{te_acc[best_single]:.2f}%")
        print(f"    inner-CV ranking (train only): {res['inner_ranking']}")
        print(f"    test ranking                 : {res['test_ranking']}")
        for name in ["FIXED", "RESELECT", "ALL4", "ORACLE"]:
            r = res[name]
            print(f"    {name:9s} {'+'.join(t[:4] for t in r['combo']):24s} "
                  f"{r['acc']:6.2f}%  gain {r['gain']:+6.2f}")
        rows.append(res)
    return rows


out = {}
print("=" * 78)
print("SICAPv2 -- patient-level shift")
print("=" * 78)
te = pd.read_excel(os.path.join(DRIVE, "SICAPv2/partition/Test/Test.xlsx"))
tr = pd.read_excel(os.path.join(DRIVE, "SICAPv2/partition/Test/Train.xlsx"))
names = list(tr["image_name"]) + list(te["image_name"])
groups = np.array([n.split("_")[0] for n in names])
y = labels("sicapv2")
folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
out["sicapv2"] = run("sicapv2", folds, "patient")

print("\n" + "=" * 78)
print("MHIST -- official partition")
print("=" * 78)
ann = pd.read_csv(os.path.join(DRIVE, "MHIST/annotations (1).csv"))
part = ann["Partition"].values
out["mhist"] = run("mhist", [(np.where(part == "train")[0],
                              np.where(part == "test")[0])], "official")

print("\n" + "=" * 78)
print("SUMMARY -- mean gain over the best single teacher")
print("=" * 78)
for ds, rows in out.items():
    print(f"\n  {ds}")
    for name in ["FIXED", "RESELECT", "ALL4", "ORACLE"]:
        g = np.array([r[name]["gain"] for r in rows])
        print(f"    {name:9s} mean {g.mean():+6.2f} pp   "
              f"wins {int((g > 0).sum())}/{len(g)}   "
              f"folds {np.round(g, 2).tolist()}")
    hits = sum(1 for r in rows
               if r["RESELECT"]["combo"] == r["ORACLE"]["combo"])
    print(f"    re-selection matched the oracle combination on "
          f"{hits}/{len(rows)} folds")

with open("/content/reselect_results.json", "w") as f:
    json.dump(out, f, indent=2)
try:
    import shutil
    shutil.copy("/content/reselect_results.json",
                os.path.join(HET, "expanded/reselect_results.json"))
except Exception as e:
    print("Drive copy failed:", e)
print("\nsaved reselect_results.json")
