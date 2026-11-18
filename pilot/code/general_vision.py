# ==========================================================================
# W3 v3 -- COMPETENCE-MATCHED COMPARISONS.  ONE CELL.
#
# v2 produced a pooled result of +1.48 pp (p = 0.006) that appears to show CKA
# predicting fusion utility in general vision. It does not. The low-CKA
# candidate in every v2 comparison (MAE ViT) was also the weakest encoder, so
# competence and divergence were entangled exactly as they were in the
# pathology pool. Utility difference tracked the competence gap at r = 0.95 and
# ran opposite to the CKA gap at r = -0.93. The single competence-matched
# comparison available (EuroSAT, 0.90 pp apart) gave -0.03 pp across a CKA gap
# of 0.573.
#
# v3 selects, on each dataset, the candidate PAIR that maximises the CKA gap
# subject to a competence gap below MAX_COMP_GAP. That is the controlled test.
# It also reports the confounded pair for contrast, so both appear side by
# side.
#
# Features are cached to disk, so re-running is cheap.
# ==========================================================================
import os, time, json, itertools, traceback
from contextlib import contextmanager
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT, CACHE = "/content/w3", "/content/w3/cache"
os.makedirs(CACHE, exist_ok=True)
BATCH, WORKERS, SEED = 64, 4, 0
MAX_COMP_GAP = 5.0        # percentage points; pairs above this are "confounded"
MIN_CKA_GAP = 0.15        # a controlled pair must still span a real CKA range

ENCODERS = {
    "dinov2_l":   "vit_large_patch14_dinov2.lvd142m",
    "eva02_clip": "eva02_large_patch14_clip_224.merged2b",
    "swin_l":     "swin_large_patch4_window7_224.ms_in22k",
    "convnext_l": "convnext_large.fb_in22k",
    "resnet50":   "resnet50.a1_in1k",
    "vit_mae":    "vit_base_patch16_224.mae",
}
ANCHOR = "dinov2_l"

DATASETS = {
    "dtd":      lambda tf: datasets.DTD("/content/data", split="train",
                                        transform=tf, download=True),
    "aircraft": lambda tf: datasets.FGVCAircraft("/content/data", split="train",
                                                 transform=tf, download=True),
    "eurosat":  lambda tf: datasets.EuroSAT("/content/data", transform=tf,
                                            download=True),
}


class Log:
    I = {"info": "   ", "ok": " ok", "warn": "  !", "error": "XXX", "step": " ->"}

    def __init__(self, name):
        self.t0 = time.time()
        self.path = os.path.join(OUT, name + "_" +
                                 datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
        self.fh = open(self.path, "a", buffering=1)
        self._raw("=== " + name + " " + str(datetime.now())[:19] + " ===")

    def _raw(self, t):
        print(t, flush=True)
        try: self.fh.write(t + "\n")
        except Exception: pass

    def _l(self, k, m):
        self._raw("[%6.1fm][%s] %s" % ((time.time()-self.t0)/60, self.I[k], m))

    def info(self, m): self._l("info", str(m))
    def ok(self, m): self._l("ok", str(m))
    def warn(self, m): self._l("warn", str(m))
    def error(self, m): self._l("error", str(m))
    def section(self, t):
        self._raw(""); self._raw("=" * 74); self._raw("  " + t); self._raw("=" * 74)
    def kv(self, **p):
        for k, v in p.items(): self._l("info", "%-26s = %s" % (k, v))
    @contextmanager
    def timer(self, label):
        t = time.time(); self._l("step", label + " ...")
        try: yield
        except Exception:
            self._l("error", label + " FAILED"); self._raw(traceback.format_exc()); raise
        self._l("ok", "%s done in %.1fm" % (label, (time.time()-t)/60))
    def close(self):
        self._raw("=== finished in %.1f min ===" % ((time.time()-self.t0)/60))
        self.fh.close()


LOG = Log("w3_v3_matched")


# ------------------------------------------------------------------ features
@torch.no_grad()
def get_features(ds_name, key):
    """Cached: extraction happens once per (dataset, encoder)."""
    fp = os.path.join(CACHE, f"{ds_name}_{key}.npz")
    if os.path.exists(fp):
        z = np.load(fp)
        return z["F"], z["y"]
    import timm
    m = timm.create_model(ENCODERS[key], pretrained=True,
                          num_classes=0).eval().to(DEV)
    cfg = timm.data.resolve_data_config({}, model=m)
    tf = timm.data.create_transform(**cfg)
    dl = DataLoader(DATASETS[ds_name](tf), batch_size=BATCH, shuffle=False,
                    num_workers=WORKERS, pin_memory=True)
    F, Y = [], []
    for x, y in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            f = m(x.to(DEV, non_blocking=True))
        if f.ndim == 3: f = f.mean(1)
        F.append(f.float().cpu().numpy()); Y.append(np.asarray(y))
    del m; torch.cuda.empty_cache()
    F, Y = np.concatenate(F), np.concatenate(Y)
    np.savez_compressed(fp, F=F.astype(np.float32), y=Y)
    return F, Y


def linear_cka(X, Y):
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    d = np.linalg.norm(X.T@X, "fro") * np.linalg.norm(Y.T@Y, "fro")
    return float(np.linalg.norm(Y.T@X, "fro")**2 / d) if d else 0.0


def perfold_logits(F, y, splits):
    Z = np.zeros((len(y), len(np.unique(y))))
    for tr, va in splits:
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
        clf.fit(sc.transform(F[tr]), y[tr])
        d = clf.decision_function(sc.transform(F[va]))
        Z[va] = np.column_stack([-d, d]) if d.ndim == 1 else d
    return Z


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def fit_fusion(Xtr, ytr, C, K, epochs=800, lr0=.05, wd=1e-3, lam=.01, patience=100):
    W = np.zeros((C, K)); W[:, 0] = .5
    tl = np.zeros(K); b = np.zeros(C)
    P_ = [W, tl, b]; m_ = [np.zeros_like(p) for p in P_]; v_ = [np.zeros_like(p) for p in P_]
    Y = np.eye(C)[ytr]; N = len(ytr); best = (1e9, None, 0)
    for ep in range(epochs):
        lr = lr0 * (1 + np.cos(np.pi * ep / epochs)) / 2
        temps = np.clip(np.exp(tl), .1, 10.)
        act = ((np.exp(tl) >= .1) & (np.exp(tl) <= 10.)).astype(float)
        sc_ = Xtr / temps[None, :, None]
        al = softmax(W, axis=1)
        out = np.einsum("ck,nkc->nc", al, sc_) + b[None, :]
        G = (softmax(out, 1) - Y) / N
        dA = np.einsum("nc,nkc->ck", G, sc_) + (lam / C) * (np.log(al + 1e-8) + 1)
        dW = al * (dA - (dA * al).sum(1, keepdims=True))
        dtl = -np.einsum("nc,ck,nkc->k", G, al, sc_) * act
        db = G.sum(0)
        for i, (p, g) in enumerate(zip(P_, [dW, dtl, db])):
            p *= (1 - lr * wd)
            m_[i] = .9*m_[i] + .1*g; v_[i] = .999*v_[i] + .001*g*g
            p -= lr * (m_[i]/(1-.9**(ep+1))) / (np.sqrt(v_[i]/(1-.999**(ep+1)))+1e-8)
        W, tl, b = P_
        t2 = np.clip(np.exp(tl), .1, 10.)
        o = np.einsum("ck,nkc->nc", softmax(W, 1), Xtr/t2[None, :, None]) + b
        l = -np.log(softmax(o, 1)[np.arange(N), ytr] + 1e-12).mean()
        if l < best[0]: best = (l, (W.copy(), tl.copy(), b.copy()), ep)
        elif ep - best[2] >= patience: break
    W, tl, b = best[1]
    return W, np.clip(np.exp(tl), .1, 10.), b


def run_fusion(Zs, y, splits, C):
    X = np.stack(Zs, 1); out = np.zeros((len(y), C))
    for tr, va in splits:
        W, t2, b = fit_fusion(X[tr], y[tr], C, X.shape[1])
        out[va] = np.einsum("ck,nkc->nc", softmax(W, 1),
                            X[va]/t2[None, :, None]) + b
    return out


# ------------------------------------------------------------------ driver
results = {}
try:
    for ds_name in DATASETS:
        LOG.section(ds_name.upper())
        feats, y = {}, None
        for key in ENCODERS:
            with LOG.timer(f"{ds_name}/{key}"):
                F, yy = get_features(ds_name, key)
            feats[key] = F
            if y is None: y = yy
        C = len(np.unique(y))
        splits = list(StratifiedKFold(5, shuffle=True,
                                      random_state=SEED).split(feats[ANCHOR], y))
        cka = {k: linear_cka(feats[ANCHOR], feats[k])
               for k in ENCODERS if k != ANCHOR}
        Z = {k: perfold_logits(feats[k], y, splits) for k in ENCODERS}
        acc = {k: float((Z[k].argmax(1) == y).mean()*100) for k in ENCODERS}
        LOG.info("CKA:  " + ", ".join(f"{k}={v:.3f}" for k, v in cka.items()))
        LOG.info("acc:  " + ", ".join(f"{k}={v:.2f}" for k, v in acc.items()))

        cands = [k for k in ENCODERS if k != ANCHOR]
        pairs = []
        for a, b in itertools.combinations(cands, 2):
            pairs.append(dict(hi=a if cka[a] > cka[b] else b,
                              lo=b if cka[a] > cka[b] else a,
                              cka_gap=abs(cka[a]-cka[b]),
                              comp_gap=abs(acc[a]-acc[b])))
        matched = [p for p in pairs if p["comp_gap"] <= MAX_COMP_GAP
                   and p["cka_gap"] >= MIN_CKA_GAP]
        matched.sort(key=lambda p: -p["cka_gap"])
        confounded = sorted(pairs, key=lambda p: -p["comp_gap"])[:1]

        if not matched:
            LOG.warn(f"no pair with competence gap <= {MAX_COMP_GAP} and "
                     f"CKA gap >= {MIN_CKA_GAP}")
        ds_rows = []
        for tag, plist in [("CONTROLLED", matched[:1]), ("CONFOUNDED", confounded)]:
            for p in plist:
                hi, lo = p["hi"], p["lo"]
                base_pool = [k for k in cands if k not in (hi, lo)]
                base_k = min(base_pool, key=lambda k: abs(cka[k] - np.mean(
                    [cka[hi], cka[lo]])))
                base = run_fusion([Z[ANCHOR], Z[base_k]], y, splits, C)
                b_acc = float((base.argmax(1) == y).mean()*100)
                fh_ = run_fusion([Z[ANCHOR], Z[base_k], Z[hi]], y, splits, C)
                fl_ = run_fusion([Z[ANCHOR], Z[base_k], Z[lo]], y, splits, C)
                mh = float((fh_.argmax(1) == y).mean()*100) - b_acc
                ml = float((fl_.argmax(1) == y).mean()*100) - b_acc
                foldh = [float((fh_[va].argmax(1) == y[va]).mean()*100) for _, va in splits]
                foldl = [float((fl_[va].argmax(1) == y[va]).mean()*100) for _, va in splits]
                LOG.info(f"  [{tag}] base={ANCHOR}+{base_k} ({b_acc:.2f}%)  "
                         f"hi={hi} CKA={cka[hi]:.3f} acc={acc[hi]:.2f} marg={mh:+.2f} | "
                         f"lo={lo} CKA={cka[lo]:.3f} acc={acc[lo]:.2f} marg={ml:+.2f} | "
                         f"CKAgap={p['cka_gap']:.3f} compgap={p['comp_gap']:.2f} "
                         f"diff={mh-ml:+.2f}")
                ds_rows.append(dict(tag=tag, hi=hi, lo=lo, base=base_k,
                                    base_acc=b_acc, cka_hi=cka[hi], cka_lo=cka[lo],
                                    acc_hi=acc[hi], acc_lo=acc[lo],
                                    marg_hi=mh, marg_lo=ml,
                                    cka_gap=p["cka_gap"], comp_gap=p["comp_gap"],
                                    folds_hi=foldh, folds_lo=foldl))
        results[ds_name] = dict(cka=cka, acc=acc, rows=ds_rows)

    # ---------------------------------------------------------- summary
    for tag in ["CONTROLLED", "CONFOUNDED"]:
        diffs, used = [], []
        for ds, r in results.items():
            for row in r["rows"]:
                if row["tag"] != tag: continue
                d = np.array(row["folds_hi"]) - np.array(row["folds_lo"])
                diffs.append(d); used.append(
                    f"{ds}({row['hi']} vs {row['lo']}, CKAgap {row['cka_gap']:.2f}, "
                    f"compgap {row['comp_gap']:.1f})")
        if not diffs: continue
        a = np.concatenate(diffs)
        t_, p_ = stats.ttest_1samp(a, 0)
        LOG.section(f"POOLED -- {tag} pairs")
        for u in used: LOG.info("  " + u)
        LOG.kv(n_folds=len(a), mean_diff=f"{a.mean():+.3f} pp",
               higher_cka_wins=f"{int((a > 0).sum())}/{len(a)}",
               t=f"{t_:.2f}", p=f"{p_:.4f}")

    LOG.section("REFERENCE")
    LOG.info("pathology, competence-matched: +0.05 pp, 5/15 folds, p = 0.86")
    with open(os.path.join(OUT, "w3_results_v3.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    LOG.ok("saved " + os.path.join(OUT, "w3_results_v3.json"))
    try:
        import shutil
        d = "./expanded"
        os.makedirs(d, exist_ok=True)
        shutil.copy(os.path.join(OUT, "w3_results_v3.json"), d)
        LOG.ok("copied to Drive")
    except Exception as e:
        LOG.warn("Drive copy failed: " + str(e))
except Exception as e:
    LOG.error(str(e)); LOG._raw(traceback.format_exc()); raise
finally:
    LOG.close()
