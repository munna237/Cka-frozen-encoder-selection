"""Dataset manifests and decoded 224x224 uint8 image caches.

Why decode once: v2 spent ~19 min per encoder just decoding BRACS RoIs
(139-173 MP each) over Drive's FUSE mount. Decoding each image once to a
224x224 uint8 array and letting every encoder read that array makes the
per-encoder cost GPU-bound. Preprocessing is identical to v2's CTransPath
extraction (Resize(224) shorter side, CenterCrop(224)); each encoder then
applies only its own mean/std normalisation.

Row order is defined ONCE, by the manifest, and every encoder's features are
written in that order. The manifest stores a hash of (uid, label) so the
analysis can assert alignment instead of assuming it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess  # noqa: F401  (kept for callers)
import time

import numpy as np

import config as CFG

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Datasets fetched from the internet. Their decoded 224px caches are saved to
# Drive (CRC-VAL 1.1 GB, NCT 20k 3.0 GB, PCam 20k 3.0 GB) so a Colab reset never
# forces a re-download again. Set False if Drive space is tight.
DOWNLOADED = {"crc_val_7k", "nct_crc_100k", "pcam"}
PERSIST_DOWNLOADED = True


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _mk(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def manifest_hash(uids, labels) -> str:
    h = hashlib.sha256()
    for u, l in zip(uids, labels):
        h.update(f"{u}\t{int(l)}\n".encode())
    return h.hexdigest()


def stratified_cap(labels, cap, seed=CFG.SEED, groups=None):
    """Indices (sorted) of a stratified subsample of size `cap`.
    If groups are given, whole groups are kept or dropped together so that the
    grouped folds downstream stay meaningful."""
    labels = np.asarray(labels)
    n = len(labels)
    if cap is None or n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups)
        ug = np.unique(groups)
        rng.shuffle(ug)
        keep, total = [], 0
        for g in ug:
            idx = np.where(groups == g)[0]
            if total + len(idx) > cap and total > 0:
                continue
            keep.append(idx)
            total += len(idx)
            if total >= cap * 0.98:
                break
        return np.sort(np.concatenate(keep))
    from sklearn.model_selection import train_test_split
    idx, _ = train_test_split(np.arange(n), train_size=cap, stratify=labels,
                              random_state=seed)
    return np.sort(idx)


def download(url, dst, retries=8, chunk=1 << 22):
    """Resumable download that reports progress IN THE NOTEBOOK.

    (The previous version used curl in a subprocess; Colab does not show a
    subprocess's progress bar, so an 11.7 GB download looked frozen.)
    A partial file is kept as <dst>.part and resumed with an HTTP Range request,
    so interrupting the cell loses nothing."""
    import requests
    if os.path.exists(dst) and os.path.getsize(dst) > 0 and not os.path.exists(dst + ".part"):
        return dst
    _mk(os.path.dirname(dst))
    part = dst + ".part"
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers,
                              allow_redirects=True) as r:
                if r.status_code == 416:                  # already complete
                    os.replace(part, dst)
                    return dst
                r.raise_for_status()
                if have and r.status_code != 206:          # server ignored Range: restart
                    have = 0
                clen = int(r.headers.get("Content-Length", 0) or 0)
                total = clen + have if clen else 0
                mode = "ab" if have else "wb"
                t0, last, got = time.time(), 0.0, have
                name = os.path.basename(dst)
                print(f"  downloading {name}: "
                      + (f"{total / 1e9:.2f} GB" if total > have else "size not reported")
                      + (f" (resuming at {have / 1e9:.2f} GB)" if have else ""), flush=True)
                with open(part, mode) as fh:
                    for blk in r.iter_content(chunk_size=chunk):
                        fh.write(blk)
                        got += len(blk)
                        now = time.time()
                        if now - last > 30:
                            rate = (got - have) / max(now - t0, 1e-9)
                            eta = (total - got) / max(rate, 1) / 60 if total else 0.0
                            print(f"    {got / 1e9:6.2f} / "
                                  + (f"{total / 1e9:.2f} GB  " if total > have else "? GB  ")
                                  + f"{rate / 1e6:6.1f} MB/s"
                                  + (f"  ETA {eta:5.1f} min" if total > have else ""), flush=True)
                            last = now
            if total and os.path.getsize(part) < total:
                raise IOError(f"incomplete: {os.path.getsize(part)} of {total} bytes")
            os.replace(part, dst)
            print(f"  downloaded {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
            return dst
        except Exception as e:
            print(f"  download interrupted ({type(e).__name__}: {str(e)[:120]}); "
                  f"retry {attempt + 1}/{retries} in 15 s, resuming", flush=True)
            time.sleep(15)
    raise RuntimeError(f"download failed: {url}")


def _zip_tree(zp, inner):
    """Class-folder listing straight from the zip index (no extraction).
    Reproduces _folder_tree's ordering exactly: classes sorted, files sorted
    by name within a class. The manifest hash check guards this equivalence."""
    import zipfile
    by_cls = {}
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            parts = n.split("/")
            if len(parts) != 3 or parts[0] != inner or not parts[2]:
                continue
            if os.path.splitext(parts[2])[1].lower() not in IMG_EXT:
                continue
            by_cls.setdefault(parts[1], []).append(parts[2])
    classes = sorted(by_cls)
    root = os.path.join(CFG.LOCAL_DL, inner)
    files, labels = [], []
    for ci, c in enumerate(classes):
        for f in sorted(by_cls[c]):
            files.append(os.path.join(root, c, f))
            labels.append(ci)
    return classes, files, labels, root


def extract_needed(zp, files):
    """Extract only the members the manifest uses (20k of 100k for NCT)."""
    import zipfile
    missing = [f for f in files if not os.path.exists(f)]
    if not missing:
        return
    base = CFG.LOCAL_DL
    t0 = time.time()
    print(f"  extracting {len(missing)} of the zip's images that the manifest uses ...", flush=True)
    with zipfile.ZipFile(zp) as z:
        for i, f in enumerate(missing):
            z.extract(os.path.relpath(f, base).replace(os.sep, "/"), base)
            if (i + 1) % 5000 == 0:
                print(f"    {i + 1}/{len(missing)}", flush=True)
    print(f"  extracted in {(time.time() - t0) / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# manifests: each builder returns dict(uids, labels, groups|None, classes, reader)
#   reader = ("files", [paths]) | ("torchvision", spec) | ("h5", path, row_index)
# ---------------------------------------------------------------------------

def _bracs():
    root = CFG.BRACS_DIR
    classes = sorted(d for d in os.listdir(os.path.join(root, "train"))
                     if os.path.isdir(os.path.join(root, "train", d)))
    files, labels = [], []
    for split in ("train", "val"):                       # v2 order: train + val
        for ci, c in enumerate(classes):
            d = os.path.join(root, split, c)
            fs = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT)
            files += [os.path.join(d, f) for f in fs]
            labels += [ci] * len(fs)
    pat = re.compile(r"^BRACS_(\d+)_")
    g = [pat.match(os.path.basename(f)) for f in files]
    if sum(m is None for m in g) > 0.01 * len(files):
        print("  [bracs] WARNING: WSI id not parseable from filenames -> ungrouped folds")
        groups = None
    else:
        groups = [("BRACS_" + m.group(1)) if m else os.path.basename(f) for m, f in zip(g, files)]
    uids = [os.path.relpath(f, root) for f in files]
    return dict(uids=uids, labels=labels, groups=groups, classes=classes, reader=("files", files))


def _sicapv2():
    import pandas as pd
    part = os.path.join(CFG.SICAP_DIR, "partition", "Test")
    cols = ["NC", "G3", "G4", "G5"]
    df = pd.concat([pd.read_excel(os.path.join(part, "Train.xlsx")),
                    pd.read_excel(os.path.join(part, "Test.xlsx"))], ignore_index=True)
    names = list(df["image_name"])
    labels = df[cols].values.argmax(1).tolist()
    groups = [n.split("_Block")[0] for n in names]
    files = [os.path.join(CFG.SICAP_DIR, "images", n) for n in names]
    return dict(uids=names, labels=labels, groups=groups, classes=cols, reader=("files", files))


def _mhist():
    import pandas as pd
    csvp = next((os.path.join(CFG.MHIST_DIR, c) for c in CFG.MHIST_CSV_CANDIDATES
                 if os.path.exists(os.path.join(CFG.MHIST_DIR, c))), None)
    if csvp is None:
        raise FileNotFoundError(f"no MHIST annotations csv in {CFG.MHIST_DIR}")
    df = pd.read_csv(csvp)
    classes = sorted(df["Majority Vote Label"].unique())          # ['HP', 'SSA']
    cmap = {c: i for i, c in enumerate(classes)}
    names = list(df["Image Name"])
    labels = df["Majority Vote Label"].map(cmap).tolist()
    files = [os.path.join(CFG.MHIST_DIR, "images", n) for n in names]
    return dict(uids=names, labels=labels, groups=None, classes=classes,
                reader=("files", files), extra={"partition": list(df["Partition"])})


def _folder_tree(root):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    files, labels = [], []
    for ci, c in enumerate(classes):
        fs = sorted(f for f in os.listdir(os.path.join(root, c))
                    if os.path.splitext(f)[1].lower() in IMG_EXT)
        files += [os.path.join(root, c, f) for f in fs]
        labels += [ci] * len(fs)
    return classes, files, labels


def _zenodo_zip(fname, inner):
    zp = download(CFG.ZENODO_NCT.format(fname), os.path.join(CFG.LOCAL_DL, fname))
    classes, files, labels, root = _zip_tree(zp, inner)
    return dict(uids=[os.path.relpath(f, root) for f in files], labels=labels, groups=None,
                classes=classes, reader=("files", files), zip=zp)


def _crc_val_7k():
    return _zenodo_zip("CRC-VAL-HE-7K.zip", "CRC-VAL-HE-7K")


def _nct_crc_100k():
    return _zenodo_zip("NCT-CRC-HE-100K.zip", "NCT-CRC-HE-100K")


def _pcam():
    import gzip
    import pandas as pd
    out = {}
    for part in ("x", "y"):
        gz = download(CFG.ZENODO_PCAM.format(f"camelyonpatch_level_2_split_test_{part}.h5.gz"),
                      os.path.join(CFG.LOCAL_DL, f"pcam_test_{part}.h5.gz"))
        h5 = gz[:-3]
        if not os.path.exists(h5):
            print(f"  gunzip {os.path.basename(gz)} ...")
            with gzip.open(gz, "rb") as fi, open(h5 + ".tmp", "wb") as fo:
                shutil.copyfileobj(fi, fo, length=1 << 24)
            os.replace(h5 + ".tmp", h5)
        out[part] = h5
    meta = download(CFG.ZENODO_PCAM.format("camelyonpatch_level_2_split_test_meta.csv"),
                    os.path.join(CFG.LOCAL_DL, "pcam_test_meta.csv"))
    import h5py
    with h5py.File(out["y"], "r") as f:
        labels = np.asarray(f["y"]).reshape(-1).astype(int).tolist()
    wsi = list(pd.read_csv(meta)["wsi"])
    assert len(wsi) == len(labels), "PCam meta / label length mismatch"
    n = len(labels)
    return dict(uids=[f"pcam_test_{i}" for i in range(n)], labels=labels, groups=wsi,
                classes=["normal", "tumor"], reader=("h5", out["x"], list(range(n))))


class _HFImages:
    """A Hugging Face image dataset exposed like a torchvision dataset
    (__getitem__ -> (PIL image, label)), used as a fallback mirror."""

    def __init__(self, hf_id, split):
        from datasets import load_dataset
        self.ds = load_dataset(hf_id, split=split)
        self._labels = [int(v) for v in self.ds["label"]]
        self.classes = list(self.ds.features["label"].names)
        self.source = f"huggingface:{hf_id}:{split}"

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, i):
        r = self.ds[int(i)]
        return r["image"], int(r["label"])


# Mirrors used only when the official torchvision host fails. A manifest built
# from a mirror records it (reader name "<dataset>_hf"), so every later session
# decodes from the same source and row order.
HF_MIRRORS = {"gtsrb": ("tanganke/gtsrb", "train")}   # 26,640 images, 43 classes (= torchvision train)


def _tv(name):
    """torchvision datasets: returns (dataset_without_transform, labels)."""
    if name.endswith("_hf"):
        hf_id, split = HF_MIRRORS[name[:-3]]
        ds = _HFImages(hf_id, split)
        return ds, ds._labels
    from torchvision import datasets as D
    root = os.path.join(CFG.LOCAL_DL, "tv")
    _mk(root)
    if name == "dtd":
        ds = D.DTD(root, split="train", download=True)
    elif name == "aircraft":
        ds = D.FGVCAircraft(root, split="train", download=True)
    elif name == "eurosat":
        ds = D.EuroSAT(root, download=True)
    elif name == "pets":
        ds = D.OxfordIIITPet(root, split="trainval", download=True)
    elif name == "food101":
        ds = D.Food101(root, split="train", download=True)
    elif name == "cifar100":
        ds = D.CIFAR100(root, train=True, download=True)
    elif name == "gtsrb":
        ds = D.GTSRB(root, split="train", download=True)
    elif name == "country211":
        ds = D.Country211(root, split="train", download=True)
    else:
        raise KeyError(name)
    labels = tv_labels(ds)
    return ds, labels


def tv_labels(ds):
    for attr in ("targets", "_labels"):
        if hasattr(ds, attr):
            return [int(v) for v in getattr(ds, attr)]
    if hasattr(ds, "_samples"):
        return [int(s[1]) for s in ds._samples]
    print("  (labels not exposed; iterating once)")
    return [int(ds[i][1]) for i in range(len(ds))]


def _tv_manifest(name):
    src = name
    try:
        ds, labels = _tv(name)
    except Exception as e:
        if name not in HF_MIRRORS:
            raise
        print(f"  [{name}] official torchvision download failed ({type(e).__name__}: "
              f"{str(e)[:100]}); using mirror {HF_MIRRORS[name][0]}", flush=True)
        src = name + "_hf"
        ds, labels = _tv(src)
    classes = [str(c) for c in getattr(ds, "classes", sorted(set(labels)))]
    return dict(uids=[f"{src}_{i}" for i in range(len(labels))], labels=labels, groups=None,
                classes=classes, reader=("torchvision", src, list(range(len(labels)))),
                extra_source=getattr(ds, "source", f"torchvision:{name}"))


BUILDERS = {
    "bracs": _bracs, "sicapv2": _sicapv2, "mhist": _mhist,
    "crc_val_7k": _crc_val_7k, "nct_crc_100k": _nct_crc_100k, "pcam": _pcam,
    **{k: (lambda k=k: _tv_manifest(k)) for k in
       ("dtd", "aircraft", "eurosat", "pets", "food101", "cifar100", "gtsrb", "country211")},
}


def dataset_cfg(name):
    for dsets in (CFG.PATHOLOGY_DATASETS, CFG.VISION_DATASETS):
        if name in dsets:
            return dsets[name]
    raise KeyError(name)


def build_manifest(name, force=False):
    """Build (or load) the manifest; applies the stratified cap. Saved to Drive."""
    _mk(CFG.MANIFEST_DIR)
    jp = os.path.join(CFG.MANIFEST_DIR, f"{name}.json")
    if os.path.exists(jp) and not force:
        with open(jp) as fh:
            man = json.load(fh)
        return man
    t0 = time.time()
    m = BUILDERS[name]()
    labels = np.asarray(m["labels"], int)
    groups = None if m["groups"] is None else np.asarray(m["groups"])
    cap = dataset_cfg(name)["cap"]
    keep = stratified_cap(labels, cap, groups=groups)
    reader = list(m["reader"])
    # subset the reader's per-row source list (last element)
    reader[-1] = [reader[-1][i] for i in keep]
    if m.get("zip"):
        extract_needed(m["zip"], reader[-1])
    man = dict(
        name=name,
        n=int(len(keep)), n_before_cap=int(len(labels)), cap=cap, seed=CFG.SEED,
        classes=list(m["classes"]),
        uids=[m["uids"][i] for i in keep],
        labels=labels[keep].tolist(),
        groups=None if groups is None else groups[keep].tolist(),
        reader=reader,
        class_counts=np.bincount(labels[keep], minlength=len(m["classes"])).tolist(),
        n_groups=None if groups is None else int(len(np.unique(groups[keep]))),
        built=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    if "extra" in m:
        man["extra"] = {k: [v[i] for i in keep] for k, v in m["extra"].items()}
    if "extra_source" in m:
        man["source"] = m["extra_source"]
    man["hash"] = manifest_hash(man["uids"], man["labels"])
    with open(jp, "w") as fh:
        json.dump(man, fh)
    with open(os.path.join(CFG.MANIFEST_DIR, f"{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "uid", "label", "group"])
        for i, (u, l) in enumerate(zip(man["uids"], man["labels"])):
            w.writerow([i, u, l, "" if man["groups"] is None else man["groups"][i]])
    print(f"  [{name}] manifest n={man['n']} (of {man['n_before_cap']}) "
          f"classes={len(man['classes'])} groups={man['n_groups']} "
          f"({time.time() - t0:.0f}s)")
    return man


# ---------------------------------------------------------------------------
# decoding to the uint8 cache
# ---------------------------------------------------------------------------

class _Decoder:
    """torch Dataset yielding uint8 (224, 224, 3) arrays for a manifest."""

    def __init__(self, man, size=CFG.IMAGE_SIZE):
        self.reader = man["reader"]
        self.size = size
        self._h5 = None
        self._tv = None

    def __len__(self):
        return len(self.reader[-1])

    def _tf(self, img):
        from PIL import Image
        img = img.convert("RGB")
        w, h = img.size
        s = self.size / min(w, h)
        nw, nh = max(self.size, round(w * s)), max(self.size, round(h * s))
        img = img.resize((nw, nh), Image.BILINEAR, reducing_gap=3.0)
        l, t = (nw - self.size) // 2, (nh - self.size) // 2
        return np.asarray(img.crop((l, t, l + self.size, t + self.size)), dtype=np.uint8)

    def __getitem__(self, i):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        kind = self.reader[0]
        if kind == "files":
            img = Image.open(self.reader[1][i])
            try:
                img.draft("RGB", (2 * self.size, 2 * self.size))   # JPEG only; no-op otherwise
            except Exception:
                pass
            return self._tf(img)
        if kind == "h5":
            import h5py
            if self._h5 is None:
                self._h5 = h5py.File(self.reader[1], "r")["x"]
            return self._tf(Image.fromarray(np.asarray(self._h5[self.reader[2][i]])))
        if kind == "torchvision":
            if self._tv is None:
                self._tv, _ = _tv(self.reader[1])
            img, _ = self._tv[self.reader[2][i]]
            return self._tf(img)
        raise ValueError(kind)


def _sources_present(man, n_check=25):
    kind = man["reader"][0]
    if kind == "files":
        paths = man["reader"][1]
        step = max(1, len(paths) // n_check)
        return all(os.path.exists(p) for p in paths[::step]) and os.path.exists(paths[-1])
    if kind == "h5":
        return os.path.exists(man["reader"][1])
    return True          # torchvision: checked by ensure_sources itself


def ensure_sources(name, man):
    """Make sure the raw images behind a saved manifest exist in THIS session.

    Manifests live on Drive and survive a Colab restart; downloaded images live
    in /content and do not. Without this, a resumed run tries to decode files
    that are gone (the CRC-VAL-HE-7K FileNotFoundError). Re-running the builder
    re-downloads the data; the rebuilt manifest must hash-match the saved one,
    so the row order cannot silently change between sessions."""
    kind = man["reader"][0]
    if kind == "torchvision":
        # download ONCE in the main process; otherwise every DataLoader worker
        # would start its own download of the same archive
        _tv(man["reader"][1])
        return
    if _sources_present(man):
        return
    if name in ("bracs", "sicapv2", "mhist"):
        raise FileNotFoundError(f"[{name}] source images missing on Drive: e.g. "
                                f"{man['reader'][1][0]} -- check the PATHS block in config.py")
    print(f"  [{name}] source images not in this session (runtime was reset); "
          f"downloading again ...", flush=True)
    m = BUILDERS[name]()
    labels = np.asarray(m["labels"], int)
    groups = None if m["groups"] is None else np.asarray(m["groups"])
    keep = stratified_cap(labels, dataset_cfg(name)["cap"], groups=groups)
    uids = [m["uids"][i] for i in keep]
    h = manifest_hash(uids, labels[keep].tolist())
    if h != man["hash"]:
        raise RuntimeError(f"[{name}] re-downloaded data does not reproduce the saved manifest "
                           f"(hash {h[:12]} vs {man['hash'][:12]}); refusing to mix row orders")
    src = list(m["reader"])
    src[-1] = [src[-1][i] for i in keep]
    if src[-1] != man["reader"][-1]:
        raise RuntimeError(f"[{name}] source paths changed after re-download")
    if m.get("zip"):
        extract_needed(m["zip"], src[-1])
    if not _sources_present(man):
        raise FileNotFoundError(f"[{name}] sources still missing after re-download")
    print(f"  [{name}] sources restored; manifest hash verified", flush=True)


def _cache_paths(name):
    return (os.path.join(CFG.LOCAL_IMG, f"{name}.npy"),
            os.path.join(CFG.LOCAL_IMG, f"{name}.progress.json"),
            os.path.join(CFG.IMG_DRIVE_DIR, f"{name}.npy"))


def _is_complete(npy, man):
    meta = npy + ".meta.json"
    if not (os.path.exists(npy) and os.path.exists(meta)):
        return False
    with open(meta) as fh:
        m = json.load(fh)
    return m.get("hash") == man["hash"] and m.get("complete") is True


def ensure_images(name, workers=8, chunk=1024):
    """Return a read-only memmap (n, 224, 224, 3) uint8 for dataset `name`.

    Order of preference: complete local cache -> complete Drive cache (copied
    to local) -> decode now (resumable chunk by chunk)."""
    import torch
    from torch.utils.data import DataLoader
    _mk(CFG.LOCAL_IMG, CFG.IMG_DRIVE_DIR)
    man = build_manifest(name)
    local, prog, drive = _cache_paths(name)
    if _is_complete(local, man):
        return np.load(local, mmap_mode="r"), man
    if _is_complete(drive, man):
        print(f"  [{name}] copying image cache from Drive ...")
        shutil.copy(drive, local)
        shutil.copy(drive + ".meta.json", local + ".meta.json")
        return np.load(local, mmap_mode="r"), man

    ensure_sources(name, man)
    n, S = man["n"], CFG.IMAGE_SIZE
    if os.path.exists(local) and os.path.exists(prog):
        arr = np.load(local, mmap_mode="r+")
        with open(prog) as fh:
            done = set(json.load(fh)["done"])
    else:
        arr = np.lib.format.open_memmap(local, mode="w+", dtype=np.uint8, shape=(n, S, S, 3))
        done = set()
    chunks = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]
    todo = [i for i in range(len(chunks)) if i not in done]
    print(f"  [{name}] decoding {n} images: {len(todo)}/{len(chunks)} chunks to go")
    dec = _Decoder(man)
    t0 = time.time()
    for k, ci in enumerate(todo):
        a, b = chunks[ci]
        sub = torch.utils.data.Subset(dec, list(range(a, b)))
        dl = DataLoader(sub, batch_size=64, num_workers=workers, shuffle=False,
                        collate_fn=lambda xs: np.stack(xs))
        pos = a
        for batch in dl:
            arr[pos:pos + len(batch)] = batch
            pos += len(batch)
        assert pos == b
        arr.flush()
        done.add(ci)
        with open(prog, "w") as fh:
            json.dump({"done": sorted(done)}, fh)
        el = time.time() - t0
        eta = el / (k + 1) * (len(todo) - k - 1) / 60
        print(f"    chunk {ci + 1}/{len(chunks)}  {b}/{n}  ETA {eta:5.1f} min", flush=True)
    del arr
    # sanity: no all-zero images (an undecoded row would be all zeros)
    arr = np.load(local, mmap_mode="r")
    zero = [i for i in range(0, n, max(1, n // 500)) if not arr[i].any()]
    if zero:
        raise RuntimeError(f"{len(zero)} sampled rows are all-zero, e.g. {zero[:5]}")
    with open(local + ".meta.json", "w") as fh:
        json.dump({"hash": man["hash"], "complete": True, "n": n}, fh)
    if os.path.exists(prog):
        os.remove(prog)
    if dataset_cfg(name).get("persist") or (PERSIST_DOWNLOADED and name in DOWNLOADED):
        print(f"  [{name}] persisting image cache to Drive ({arr.nbytes / 1e9:.2f} GB) ...")
        shutil.copy(local, drive + ".tmp")
        os.replace(drive + ".tmp", drive)
        shutil.copy(local + ".meta.json", drive + ".meta.json")
    return arr, man


def drop_local_images(name):
    local, prog, _ = _cache_paths(name)
    for p in (local, local + ".meta.json", prog):
        if os.path.exists(p):
            os.remove(p)
