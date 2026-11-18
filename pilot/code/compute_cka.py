# ==========================================================================
# COMPLETE CKA MATRIX -- all 6 teacher pairs x 3 datasets, using the refit
# CTransPath features. Linear CKA plus an RBF check (for Table 8).
#
# ONE CELL. No GPU needed. A few minutes, mostly Drive reads.
# ==========================================================================
import os
import json
import numpy as np

DRIVE = "/content/drive/MyDrive"
HET = os.path.join(DRIVE, "project_root")
SUB_N = 2000          # subsample for the released bundle and RBF check
rng = np.random.default_rng(0)

TEACHERS = ["uni", "phikon_v2", "ctranspath", "resnet50_imagenet"]
NICE = {"uni": "UNI", "phikon_v2": "Phikon-v2",
        "ctranspath": "CTransPath", "resnet50_imagenet": "ResNet50"}
DATASETS = ["bracs", "sicapv2", "mhist"]

# values as printed in the submitted paper (Table 7)
PUBLISHED = {
    "bracs":   {("uni", "phikon_v2"): 0.834, ("uni", "ctranspath"): 0.246,
                ("uni", "resnet50_imagenet"): 0.450,
                ("phikon_v2", "ctranspath"): 0.322,
                ("phikon_v2", "resnet50_imagenet"): 0.425,
                ("ctranspath", "resnet50_imagenet"): 0.228},
    "sicapv2": {("uni", "phikon_v2"): 0.696, ("uni", "ctranspath"): 0.317,
                ("uni", "resnet50_imagenet"): 0.477,
                ("phikon_v2", "ctranspath"): 0.422,
                ("phikon_v2", "resnet50_imagenet"): 0.490,
                ("ctranspath", "resnet50_imagenet"): 0.307},
    "mhist":   {("uni", "phikon_v2"): 0.688, ("uni", "ctranspath"): 0.006,
                ("uni", "resnet50_imagenet"): 0.496,
                ("phikon_v2", "ctranspath"): 0.004,
                ("phikon_v2", "resnet50_imagenet"): 0.440,
                ("ctranspath", "resnet50_imagenet"): 0.005},
}


# ---------------------------------------------------------------- loading
def feat_path(ds, t):
    """Refit features for CTransPath; original cached features otherwise."""
    if t == "ctranspath":
        p = os.path.join(HET, f"expanded/{ds}/ctranspath_fixed")
        for fn in ["features.npy", "trainval_features.npy"]:
            if os.path.exists(os.path.join(p, fn)):
                return os.path.join(p, fn)
        return None
    base = os.path.join(HET, f"cached_features_{ds}", t)
    for fn in ["trainval_features.npy", "all_features.npy"]:
        if os.path.exists(os.path.join(base, fn)):
            return os.path.join(base, fn)
    return None


def load_feats(ds):
    out = {}
    for t in TEACHERS:
        p = feat_path(ds, t)
        if p is None:
            print(f"    MISSING: {ds}/{t}")
            continue
        out[t] = np.load(p).astype(np.float64)
    return out


# ---------------------------------------------------------------- CKA
def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    d = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(np.linalg.norm(Y.T @ X, "fro") ** 2 / d) if d else 0.0


def rbf_cka(X, Y, n=1500):
    """RBF-kernel CKA with the median-distance bandwidth heuristic."""
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    A, B = X[idx], Y[idx]

    def K(M):
        sq = (M ** 2).sum(1)
        D = sq[:, None] + sq[None, :] - 2 * M @ M.T
        np.maximum(D, 0, out=D)
        sigma2 = np.median(D[D > 0]) if (D > 0).any() else 1.0
        return np.exp(-D / (2 * sigma2))

    m = len(A)
    H = np.eye(m) - np.ones((m, m)) / m
    Kc, Lc = H @ K(A) @ H, H @ K(B) @ H
    den = np.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())
    return float((Kc * Lc).sum() / den) if den else 0.0


# ---------------------------------------------------------------- run
results, bundle_dir = {}, "/content/cka_bundle"
os.makedirs(bundle_dir, exist_ok=True)

for ds in DATASETS:
    print("\n" + "=" * 78)
    print(f"  {ds.upper()}")
    print("=" * 78)
    F = load_feats(ds)
    if len(F) < 4:
        print("  incomplete teacher set, skipping")
        continue
    n = min(len(v) for v in F.values())
    for t, v in F.items():
        print(f"    {NICE[t]:12s} {v.shape}")
    if len({len(v) for v in F.values()}) > 1:
        print("    *** row counts differ -- alignment problem, not extracting ***")
        continue

    idx = np.sort(rng.choice(n, size=min(SUB_N, n), replace=False))
    ds_out, rows = {}, []
    print()
    print(f"    {'pair':30s}{'linear':>9s}{'RBF':>8s}{'published':>11s}{'delta':>9s}")
    print("    " + "-" * 67)
    for i, a in enumerate(TEACHERS):
        for b in TEACHERS[i + 1:]:
            lin = linear_cka(F[a], F[b])
            rbf = rbf_cka(F[a][idx], F[b][idx])
            pub = PUBLISHED[ds].get((a, b))
            d = lin - pub if pub is not None else float("nan")
            flag = "  <-- CHANGED" if pub is not None and abs(d) > 0.02 else ""
            print(f"    {NICE[a]+' - '+NICE[b]:30s}{lin:9.4f}{rbf:8.4f}"
                  f"{pub if pub is not None else float('nan'):11.3f}{d:+9.3f}{flag}")
            ds_out[f"{a}|{b}"] = dict(linear=round(lin, 4), rbf=round(rbf, 4),
                                      published=pub, delta=round(d, 4))
            rows.append((NICE[a], NICE[b], lin, rbf, pub))
    results[ds] = ds_out

    # export a small verifiable subset (float16 reproduces CKA to 4 dp)
    d_ = os.path.join(bundle_dir, ds)
    os.makedirs(d_, exist_ok=True)
    np.save(os.path.join(d_, "cka_subset_indices.npy"), idx)
    for t, v in F.items():
        np.save(os.path.join(d_, f"{t}_features_subset.npy"),
                v[idx].astype(np.float16))

# ---------------------------------------------------------------- summary
print("\n" + "=" * 78)
print("  DOMAIN vs ARCHITECTURE")
print("  pathology-pretrained pairs (UNI / Phikon-v2 / CTransPath) vs")
print("  any pair involving the ImageNet ResNet50")
print("=" * 78)
PATH = {"uni", "phikon_v2", "ctranspath"}
for ds, r in results.items():
    within, across = [], []
    for k, v in r.items():
        a, b = k.split("|")
        (within if {a, b} <= PATH else across).append(v["linear"])
    print(f"  {ds:9s} within-pathology mean={np.mean(within):.3f} "
          f"(range {min(within):.3f}-{max(within):.3f})   "
          f"vs ResNet50 mean={np.mean(across):.3f} "
          f"(range {min(across):.3f}-{max(across):.3f})")

with open("/content/cka_matrix_corrected.json", "w") as f:
    json.dump(results, f, indent=2)

try:
    dest = os.path.join(HET, "expanded/cka_matrix_corrected.json")
    with open(dest, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {dest}")
except Exception as e:
    print(f"\ncould not write to Drive ({e}); local copy at "
          f"/content/cka_matrix_corrected.json")

tot = sum(os.path.getsize(os.path.join(dp, f))
          for dp, _, fs in os.walk(bundle_dir) for f in fs)
print(f"verifiable feature bundle: {tot/1e6:.1f} MB at {bundle_dir}")
print("\nUpload cka_matrix_corrected.json to the chat.")
