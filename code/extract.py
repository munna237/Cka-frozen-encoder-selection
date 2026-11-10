"""Resumable feature extraction.

Layout on Drive:
    features/<dataset>/<encoder>.npy        (n, d) float32, rows in manifest order
    features/<dataset>/<encoder>.json       provenance: encoder info, manifest hash, stats
    features/<dataset>/_shards_<encoder>/   temporary; deleted after assembly

A Colab disconnect loses at most one shard (SHARD_ROWS rows) of one encoder.
Re-running the notebook cell resumes exactly where it stopped.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import config as CFG
import encoders as E

TIMM054_DIR = os.environ.get("V3_TIMM054", "/content/timm054")


def _paths(ds, enc):
    d = os.path.join(CFG.FEAT_DIR, ds)
    return d, os.path.join(d, f"{enc}.npy"), os.path.join(d, f"{enc}.json"), \
        os.path.join(d, f"_shards_{enc}")


def is_done(ds, enc, man):
    _, npy, meta, _ = _paths(ds, enc)
    if not (os.path.exists(npy) and os.path.exists(meta)):
        return False
    with open(meta) as fh:
        m = json.load(fh)
    return m.get("manifest_hash") == man["hash"] and m.get("complete") is True


def _shard_ranges(n):
    return [(s, min(s + CFG.SHARD_ROWS, n)) for s in range(0, n, CFG.SHARD_ROWS)]


def _assemble(ds, enc, man, info, t_start):
    d, npy, meta, sh = _paths(ds, enc)
    ranges = _shard_ranges(man["n"])
    parts = [np.load(os.path.join(sh, f"shard_{i:04d}.npy")) for i in range(len(ranges))]
    F = np.concatenate(parts).astype(CFG.FEATURE_DTYPE)
    assert F.shape[0] == man["n"], f"{F.shape[0]} rows != manifest {man['n']}"
    stats = E.feature_stats(F)
    if not stats["finite"]:
        raise RuntimeError(f"{ds}/{enc}: non-finite features; not saving")
    np.save(npy + ".tmp.npy", F)
    os.replace(npy + ".tmp.npy", npy)
    with open(meta, "w") as fh:
        json.dump({"dataset": ds, "encoder": enc, "manifest_hash": man["hash"],
                   "shape": list(F.shape), "dtype": CFG.FEATURE_DTYPE, "complete": True,
                   "encoder_info": info, "stats": stats,
                   "minutes": round((time.time() - t_start) / 60, 1),
                   "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, fh, indent=1, default=str)
    shutil.rmtree(sh, ignore_errors=True)
    return stats


def extract_inprocess(ds, enc_name, spec, images, man, device, enc_obj=None):
    import torch
    d, npy, meta, sh = _paths(ds, enc_name)
    t0 = time.time()
    enc = enc_obj or E.build(enc_name, spec, device)   # build first: a failed load leaves no trace
    os.makedirs(sh, exist_ok=True)
    ranges = _shard_ranges(man["n"])
    todo = [i for i in range(len(ranges))
            if not os.path.exists(os.path.join(sh, f"shard_{i:04d}.npy"))]
    print(f"    {enc_name}: {enc.info.get('n_params', 0) / 1e6:.0f}M params, "
          f"{len(todo)}/{len(ranges)} shards to go", flush=True)
    for k, i in enumerate(todo):
        a, b = ranges[i]
        out = []
        for s in range(a, b, CFG.EXTRACT_BATCH):
            x = torch.from_numpy(np.ascontiguousarray(images[s:min(s + CFG.EXTRACT_BATCH, b)]))
            out.append(enc(x))
        np.save(os.path.join(sh, f"shard_{i:04d}.npy"), np.concatenate(out).astype(np.float32))
        el = time.time() - t0
        print(f"      shard {i + 1}/{len(ranges)}  {b}/{man['n']}  "
              f"{(b - a) * (k + 1) / max(el, 1e-9):6.0f} img/s  "
              f"ETA {el / (k + 1) * (len(todo) - k - 1) / 60:5.1f} min", flush=True)
    info = dict(enc.info)
    stats = _assemble(ds, enc_name, man, info, t0)
    return stats


# ---------------------------------------------------------------------------
# CTransPath: separate process with the original timm 0.5.4
# ---------------------------------------------------------------------------

def ensure_timm054():
    if os.path.exists(os.path.join(TIMM054_DIR, "timm", "__init__.py")):
        return
    print("  installing timm==0.5.4 into an isolated directory (for CTransPath only) ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "--target", TIMM054_DIR, "timm==0.5.4"], check=True)


def _run_streaming(cmd, env):
    """Run a worker and echo its output line by line INTO THE NOTEBOOK.
    (Colab does not display a child process's own stdout/stderr.)"""
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    tail = []
    for line in p.stdout:
        line = line.rstrip("\n")
        tail = (tail + [line])[-40:]
        if ("Warning" in line or "warn(" in line) and "Error" not in line:
            continue
        print(line, flush=True)
    p.wait()
    return p.returncode, "\n".join(tail)


# Hugging Face models whose remote code needs an older transformers than Colab's.
# They run in a separate process against an isolated install (like CTransPath).
ISOLATED_HF = {"hibou_b": "4.46.3"}
OLD_TF_DIR = os.environ.get("V3_OLD_TF", "/content/transformers_4_46")


def ensure_old_transformers(version="4.46.3"):
    marker = os.path.join(OLD_TF_DIR, f".installed_{version}")
    if os.path.exists(marker):
        return
    print(f"  installing transformers=={version} into an isolated directory "
          f"(for remote-code models only) ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    "--target", OLD_TF_DIR, f"transformers=={version}",
                    "tokenizers==0.20.3", "huggingface_hub==0.26.5"], check=True)
    open(marker, "w").close()


def extract_hf_isolated(ds, enc_name, spec, images_path, man, device):
    d, npy, meta, sh = _paths(ds, enc_name)
    os.makedirs(sh, exist_ok=True)
    ensure_old_transformers(ISOLATED_HF[enc_name])
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = OLD_TF_DIR + os.pathsep + CFG.CODE_DIR
    env["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    cmd = [sys.executable, os.path.join(CFG.CODE_DIR, "hf_worker.py"), "--model", spec["id"],
           "--pool", spec.get("pool", "cls"), "--images", images_path, "--shards", sh,
           "--shard_rows", str(CFG.SHARD_ROWS), "--batch", str(CFG.EXTRACT_BATCH),
           "--device", device]
    if spec.get("trust_remote_code"):
        cmd.append("--trust_remote_code")
    rc, tail = _run_streaming(cmd, env)
    if rc != 0:
        if not any(f.startswith("shard_") for f in os.listdir(sh)):
            shutil.rmtree(sh, ignore_errors=True)        # nothing done: leave no "partial"
        raise RuntimeError(f"{enc_name} worker failed: {tail.splitlines()[-1] if tail else rc}")
    with open(os.path.join(sh, "load_report.json")) as fh:
        rep = json.load(fh)
    info = {"name": enc_name, "kind": "hf-isolated", "id": spec["id"],
            "mean": rep["mean"], "std": rep["std"], "pool": rep["pool"],
            "n_params": rep["n_params"], "load_report": rep}
    return _assemble(ds, enc_name, man, info, t0)


def extract_ctranspath(ds, images_path, man, device):
    d, npy, meta, sh = _paths(ds, "ctranspath")
    os.makedirs(sh, exist_ok=True)
    if not os.path.exists(CFG.CTRANSPATH_CKPT):
        raise FileNotFoundError(CFG.CTRANSPATH_CKPT)
    ensure_timm054()
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONPATH"] = TIMM054_DIR + os.pathsep + CFG.CODE_DIR
    worker = os.path.join(CFG.CODE_DIR, "ctranspath_worker.py")
    cmd = [sys.executable, worker, "--images", images_path, "--ckpt", CFG.CTRANSPATH_CKPT,
           "--shards", sh, "--shard_rows", str(CFG.SHARD_ROWS),
           "--batch", str(CFG.EXTRACT_BATCH), "--device", device]
    rc, _ = _run_streaming(cmd, env)
    if rc != 0:
        raise RuntimeError("CTransPath worker failed (see output above); nothing saved")
    with open(os.path.join(sh, "load_report.json")) as fh:
        rep = json.load(fh)
    info = {"name": "ctranspath", "kind": "ctranspath", "id": CFG.CTRANSPATH_CKPT,
            "mean": E.IMAGENET_MEAN, "std": E.IMAGENET_STD, "pool": "avg (timm 0.5.4 head)",
            "load_report": rep}
    return _assemble(ds, "ctranspath", man, info, t0)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_domain(datasets, encoders, device="cuda", only=None, skip_encoders=()):
    """Extract every (dataset, encoder) that is not already complete."""
    import imgcache as IC
    log = []
    E.hf_login()
    for ds in datasets:
        if only and ds not in only:
            continue
        pending = None
        print(f"\n=== {ds} ===", flush=True)
        try:
            man = IC.build_manifest(ds)
        except Exception as e:
            print(f"  !! manifest failed: {type(e).__name__}: {e}")
            log.append({"dataset": ds, "encoder": None, "status": f"manifest failed: {e}"})
            continue
        pending = [e for e in encoders if e not in skip_encoders and not is_done(ds, e, man)]
        if not pending:
            print("  all encoders complete")
            continue
        images, man = IC.ensure_images(ds)
        images_path = IC._cache_paths(ds)[0]
        for enc_name in pending:
            spec = encoders[enc_name]
            t = time.time()
            try:
                if spec["kind"] == "ctranspath":
                    stats = extract_ctranspath(ds, images_path, man, device)
                elif enc_name in ISOLATED_HF:
                    stats = extract_hf_isolated(ds, enc_name, spec, images_path, man, device)
                else:
                    stats = extract_inprocess(ds, enc_name, spec, images, man, device)
                msg = (f"ok  dim={stats['dim']}  eff_rank={stats['eff_rank']:.0f} "
                       f"({stats['eff_rank_frac']:.0%})  {(time.time() - t) / 60:.1f} min")
            except Exception as e:
                msg = f"SKIPPED: {type(e).__name__}: {str(e)[:300]}"
            print(f"  -> {enc_name}: {msg}", flush=True)
            log.append({"dataset": ds, "encoder": enc_name, "status": msg})
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    os.makedirs(CFG.LOG_DIR, exist_ok=True)
    with open(os.path.join(CFG.LOG_DIR, f"extract_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as fh:
        json.dump(log, fh, indent=1)
    return log


def status_table(datasets, encoders):
    """Which (dataset, encoder) features exist and are complete."""
    import imgcache as IC
    rows = []
    for ds in datasets:
        jp = os.path.join(CFG.MANIFEST_DIR, f"{ds}.json")
        man = json.load(open(jp)) if os.path.exists(jp) else None
        row = {"dataset": ds, "n": man["n"] if man else None}
        for e in encoders:
            row[e] = ("ok" if man and is_done(ds, e, man) else
                      ("partial" if os.path.isdir(_paths(ds, e)[3]) else "-"))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# consistency with the v2 features (pathology only)
# ---------------------------------------------------------------------------

V2_PREFERRED = ("trainval_features.npy", "all_features.npy", "features.npy")


def _find_v2(ds, enc):
    """Locate the v2 feature file covering the SAME rows as the v3 manifest.

    v3 fix: the previous version took the alphabetically first *features*.npy,
    which on BRACS was test_features.npy (the 570-image held-out institutions),
    so the comparison reported "row count differs" for UNI, Phikon-v2 and
    ResNet-50. Split-specific files (test_/train_/val_) are now never used."""
    if enc == "ctranspath":
        d = CFG.V2_CTRANSPATH_FIXED[ds]
    else:
        d = os.path.join(CFG.V2_CACHE[ds], CFG.V2_NAME[enc])
    if not os.path.isdir(d):
        return None, None
    present = set(os.listdir(d))
    fp = next((os.path.join(d, f) for f in V2_PREFERRED if f in present), None)
    if fp is None:
        cands = sorted(f for f in present if f.endswith(".npy") and "features" in f
                       and not f.startswith(("test_", "train_", "val_")))
        fp = os.path.join(d, cands[0]) if cands else None
    if fp is None:
        return None, None
    lp = os.path.join(d, os.path.basename(fp).replace("features", "labels"))
    return fp, (lp if os.path.exists(lp) else None)


def compare_with_v2(ds, enc, n_rows=3000):
    """Linear CKA between v3 and v2 features on the same rows.

    ~1.0 means the v3 pipeline reproduces v2's features for this encoder (the
    only differences are decode details). A low value means row order or
    preprocessing differs, which must be explained before using either set."""
    from cka import linear_cka
    if ds not in CFG.V2_CACHE or enc not in CFG.V2_NAME:
        return None
    fp, lp = _find_v2(ds, enc)
    _, npy, _, _ = _paths(ds, enc)
    if fp is None or not os.path.exists(npy):
        return {"dataset": ds, "encoder": enc, "status": "missing v2 or v3 features"}
    old = np.load(fp, mmap_mode="r")
    new = np.load(npy, mmap_mode="r")
    man = json.load(open(os.path.join(CFG.MANIFEST_DIR, f"{ds}.json")))
    res = {"dataset": ds, "encoder": enc, "v2_file": os.path.relpath(fp, CFG.DRIVE),
           "n_v2": int(old.shape[0]), "n_v3": int(new.shape[0])}
    if old.shape[0] != new.shape[0]:
        res["status"] = "row count differs"
        return res
    if lp:
        y_old = np.load(lp)
        res["labels_identical"] = bool(np.array_equal(y_old, np.asarray(man["labels"])))
    idx = np.sort(np.random.default_rng(0).choice(new.shape[0], min(n_rows, new.shape[0]),
                                                  replace=False))
    res["cka_v2_v3"] = float(linear_cka(np.asarray(old[idx], np.float64),
                                        np.asarray(new[idx], np.float64)))
    res["status"] = ("reproduces v2" if res["cka_v2_v3"] > 0.98 else
                     "close to v2" if res["cka_v2_v3"] > 0.90 else
                     "DIFFERS FROM v2 - investigate before analysis")
    return res


# ---------------------------------------------------------------------------
# ResNet-50 provenance: which variant did v2's "resnet50_imagenet" use?
# Diagnostic only -- it does not change the registered v3 encoder.
# ---------------------------------------------------------------------------

RESNET_CANDIDATES = {
    "torchvision IMAGENET1K_V1 (v3)": ("timm", "resnet50.tv_in1k"),
    "torchvision IMAGENET1K_V2":      ("timm", "resnet50.tv2_in1k"),
    "timm a1_in1k":                   ("timm", "resnet50.a1_in1k"),
    "timm fb_swsl_ig1b_ft_in1k":      ("timm", "resnet50.fb_swsl_ig1b_ft_in1k"),
}


def resnet_provenance(ds="mhist", n_rows=1500, device="cuda"):
    """Compare v2's ResNet-50 features with several ImageNet ResNet-50 variants
    under two preprocessings, on the same rows.

    preprocessing "v3": the shared 224 cache (Resize 224 + CenterCrop 224)
    preprocessing "tv": torchvision's classic eval transform (Resize 256 +
                        CenterCrop 224) applied to the original images
    The row with CKA ~1.0 identifies what v2 used."""
    import torch
    from PIL import Image
    from cka import linear_cka
    import imgcache as IC
    fp, _ = _find_v2(ds, "resnet50_in1k")
    if fp is None:
        raise FileNotFoundError(f"no v2 ResNet-50 features for {ds}")
    old = np.load(fp, mmap_mode="r")
    images, man = IC.ensure_images(ds)
    idx = np.sort(np.random.default_rng(0).choice(man["n"], min(n_rows, man["n"]), replace=False))
    ref = np.asarray(old[idx], np.float64)

    def tv_batch(rows):
        out = []
        for i in rows:
            img = Image.open(man["reader"][1][i]).convert("RGB")
            w, h = img.size
            s = 256 / min(w, h)
            img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
            W, H = img.size
            l, t = (W - 224) // 2, (H - 224) // 2
            out.append(np.asarray(img.crop((l, t, l + 224, t + 224)), np.uint8))
        return np.stack(out)

    can_tv = man["reader"][0] == "files"
    rows = []
    for label, (kind, mid) in RESNET_CANDIDATES.items():
        try:
            enc = E.build(label, dict(kind=kind, id=mid), device)
        except Exception as e:
            rows.append({"variant": label, "preprocessing": "-", "cka_with_v2": None,
                         "note": f"load failed: {type(e).__name__}"})
            continue
        for prep in (("v3", "tv") if can_tv else ("v3",)):
            feats = []
            for s in range(0, len(idx), 64):
                b = idx[s:s + 64]
                x = np.asarray(images[b]) if prep == "v3" else tv_batch(b)
                feats.append(enc(torch.from_numpy(np.ascontiguousarray(x))))
            Fn = np.concatenate(feats).astype(np.float64)
            rows.append({"variant": label, "preprocessing":
                         "Resize224+Crop224 (v3)" if prep == "v3" else "Resize256+Crop224",
                         "cka_with_v2": float(linear_cka(ref, Fn)), "note": ""})
        del enc
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return rows
