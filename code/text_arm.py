"""Text arm: frozen sentence encoders on public classification sets.

Same downstream pipeline as the image arms (the analysis only ever sees an
(n, d) feature matrix and labels), so this adds independent pools in a
different modality at almost no engineering cost.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import config as CFG
import imgcache as IC          # manifest_hash, stratified_cap
import extract as X            # _paths, is_done


def build_text_manifest(name, force=False):
    os.makedirs(CFG.MANIFEST_DIR, exist_ok=True)
    jp = os.path.join(CFG.MANIFEST_DIR, f"{name}.json")
    if os.path.exists(jp) and not force:
        with open(jp) as fh:
            return json.load(fh)
    from datasets import load_dataset
    hf_id, conf, split, tcol, lcol, cap = CFG.TEXT_DATASETS[name]
    ds = load_dataset(hf_id, conf, split=split) if conf else load_dataset(hf_id, split=split)
    texts = [str(t) for t in ds[tcol]]
    raw = list(ds[lcol])
    classes = sorted(set(raw))
    cmap = {c: i for i, c in enumerate(classes)}
    labels = np.array([cmap[v] for v in raw])
    keep_ok = np.array([len(t.strip()) > 0 for t in texts])
    idx_all = np.where(keep_ok)[0]
    sub = IC.stratified_cap(labels[idx_all], cap)
    keep = idx_all[sub]
    man = dict(name=name, n=int(len(keep)), n_before_cap=int(len(labels)), cap=cap,
               seed=CFG.SEED, classes=[str(c) for c in classes],
               uids=[f"{name}_{i}" for i in keep], labels=labels[keep].tolist(),
               groups=None, texts=[texts[i] for i in keep],
               class_counts=np.bincount(labels[keep], minlength=len(classes)).tolist(),
               source=f"{hf_id}:{conf or 'default'}:{split}", n_groups=None,
               built=time.strftime("%Y-%m-%d %H:%M:%S"))
    man["hash"] = IC.manifest_hash(man["uids"], man["labels"])
    with open(jp, "w") as fh:
        json.dump(man, fh)
    print(f"  [{name}] n={man['n']} classes={len(classes)}")
    return man


def run_text(datasets=None, encoders=None, device="cuda", batch=128):
    from sentence_transformers import SentenceTransformer
    import encoders as E
    datasets = datasets or list(CFG.TEXT_DATASETS)
    encoders = encoders or CFG.TEXT_ENCODERS
    mans = {}
    for ds in datasets:
        try:
            mans[ds] = build_text_manifest(ds)
        except Exception as e:
            print(f"  !! {ds}: manifest failed: {type(e).__name__}: {e}")
    log = []
    for enc_name, spec in encoders.items():          # load each model once
        todo = [ds for ds in mans if not X.is_done(ds, enc_name, mans[ds])]
        if not todo:
            print(f"{enc_name}: complete")
            continue
        t0 = time.time()
        try:
            model = SentenceTransformer(spec["id"], device=device)
        except Exception as e:
            print(f"  !! {enc_name}: load failed: {e}")
            continue
        for ds in todo:
            man = mans[ds]
            texts = [spec.get("prefix", "") + t for t in man["texts"]]
            F = model.encode(texts, batch_size=batch, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=False)
            F = np.asarray(F, dtype=CFG.FEATURE_DTYPE)
            d, npy, meta, _ = X._paths(ds, enc_name)
            os.makedirs(d, exist_ok=True)
            np.save(npy, F)
            stats = E.feature_stats(F)
            with open(meta, "w") as fh:
                json.dump({"dataset": ds, "encoder": enc_name, "manifest_hash": man["hash"],
                           "shape": list(F.shape), "complete": True,
                           "encoder_info": {"name": enc_name, "kind": "st", "id": spec["id"],
                                            "prefix": spec.get("prefix", "")},
                           "stats": stats}, fh, indent=1)
            print(f"  {enc_name:14s} {ds:14s} {F.shape}  eff_rank {stats['eff_rank']:.0f}")
            log.append((ds, enc_name, list(F.shape)))
        del model
        print(f"{enc_name}: done in {(time.time() - t0) / 60:.1f} min")
    return log
