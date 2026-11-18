# ==========================================================================
# Exact CKA features + bootstrap uncertainty
# Paste this whole file into a Colab cell and run.
# ==========================================================================

import os, json
import numpy as np

HET = "./project_root"
OUT = "/content/cka_exact"
os.makedirs(OUT, exist_ok=True)
DATASETS = ["bracs", "sicapv2", "mhist"]
TEACHERS = ["uni", "phikon_v2", "ctranspath", "resnet50_imagenet"]

def feat(ds, t):
    if t == "ctranspath":
        for fn in ["features.npy", "trainval_features.npy"]:
            p = os.path.join(HET, f"expanded/{ds}/ctranspath_fixed", fn)
            if os.path.exists(p): return np.load(p)
    for fn in ["trainval_features.npy", "all_features.npy"]:
        p = os.path.join(HET, f"cached_features_{ds}", t, fn)
        if os.path.exists(p): return np.load(p)
    raise FileNotFoundError(f"{ds}/{t}")

def cka(X, Y):
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    X -= X.mean(0, keepdims=True); Y -= Y.mean(0, keepdims=True)
    d = np.linalg.norm(X.T@X, "fro") * np.linalg.norm(Y.T@Y, "fro")
    return float(np.linalg.norm(Y.T@X, "fro")**2 / d) if d else 0.0

report = {}
for ds in DATASETS:
    F = {t: feat(ds, t) for t in TEACHERS}
    n = len(F["uni"])
    ent = {"n_full": int(n), "full": {}, "bootstrap": {}}
    print(f"\n{ds.upper()}  n = {n}")
    for a, b in [("uni","phikon_v2"),("uni","ctranspath"),("uni","resnet50_imagenet"),
                 ("phikon_v2","ctranspath"),("phikon_v2","resnet50_imagenet"),
                 ("ctranspath","resnet50_imagenet")]:
        full = cka(F[a], F[b])                       # FULL-N value: the paper's number
        boots = []
        for s in range(50):                          # subsample uncertainty
            idx = np.random.default_rng(s).choice(n, size=min(2000, n), replace=False)
            boots.append(cka(F[a][idx], F[b][idx]))
        ent["full"][f"{a}|{b}"] = round(full, 4)
        ent["bootstrap"][f"{a}|{b}"] = dict(mean=round(float(np.mean(boots)), 4),
                                            sd=round(float(np.std(boots)), 4),
                                            lo=round(float(np.percentile(boots, 2.5)), 4),
                                            hi=round(float(np.percentile(boots, 97.5)), 4))
        print(f"   {a:10s}-{b:18s} full={full:.4f}  "
              f"subsample {np.mean(boots):.4f} +/- {np.std(boots):.4f}")
    report[ds] = ent

    # ship the FULL features at reduced precision so the table is regenerable
    d = os.path.join(OUT, ds); os.makedirs(d, exist_ok=True)
    for t in TEACHERS:
        np.save(os.path.join(d, f"{t}_features_full_fp16.npy"),
                F[t].astype(np.float16))

json.dump(report, open("/content/cka_exact_report.json", "w"), indent=2)
tot = sum(os.path.getsize(os.path.join(dp, f))
          for dp, _, fs in os.walk(OUT) for f in fs)
print(f"\nfull features (fp16): {tot/1e6:.0f} MB at {OUT}")
print("written /content/cka_exact_report.json")
