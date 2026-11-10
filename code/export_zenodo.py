"""Package the frozen features and held-out logits for Zenodo, verify them, and
(optionally) upload them to a Zenodo *draft* that you review and publish yourself.

Layout produced (one archive per dataset keeps every file well under Zenodo's
limits and lets users download only what they need):

    features_<dataset>.zip   <dataset>/<encoder>.npy + <dataset>/<encoder>.json
    heldout_logits.zip       pools/<dataset>_logits.npz   (per-fold held-out logits)
    zenodo_manifest.json     SHA-256 of every archive and of every file inside it
    README.md                what the record contains and how to use it

The archives are written with ZIP_STORED: float32 arrays barely compress, and
stored members can be read back without decompression cost. Nothing is ever
modified in place; the source files on Drive are only read.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
import zipfile

import config as CFG

CHUNK = 1 << 22


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- inventory
def inventory(exclude_datasets=(), exclude_encoders=()):
    """Every feature file (with its provenance json) and every logits file."""
    feats = {}
    for npy in sorted(glob.glob(os.path.join(CFG.FEAT_DIR, "*", "*.npy"))):
        ds = os.path.basename(os.path.dirname(npy))
        enc = os.path.splitext(os.path.basename(npy))[0]
        if ds in exclude_datasets or enc in exclude_encoders or enc.startswith("_"):
            continue
        meta = npy[:-4] + ".json"
        feats.setdefault(ds, []).append({"encoder": enc, "npy": npy,
                                         "json": meta if os.path.exists(meta) else None,
                                         "bytes": os.path.getsize(npy)})
    logits = [p for p in sorted(glob.glob(os.path.join(CFG.RESULTS_DIR, "pools", "*_logits.npz")))
              if os.path.basename(p)[:-len("_logits.npz")] not in exclude_datasets]
    return feats, logits


def summary(feats, logits):
    n = sum(len(v) for v in feats.values())
    b = sum(x["bytes"] for v in feats.values() for x in v) + sum(os.path.getsize(p) for p in logits)
    return {"datasets": len(feats), "feature_files": n, "logits_files": len(logits),
            "total_gb": round(b / 1e9, 2),
            "per_dataset": {ds: {"encoders": len(v), "gb": round(sum(x["bytes"] for x in v) / 1e9, 3)}
                            for ds, v in sorted(feats.items())}}


# ---------------------------------------------------------------- build
def build(out_dir, feats, logits, include_logits=True, verbose=True):
    """Write one archive per dataset (+ logits) and the manifest. Resumable: an
    archive that already exists and verifies against its sidecar is skipped."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "archives": {}}
    jobs = [(f"features_{ds}.zip", [(x["npy"], f"{ds}/{x['encoder']}.npy") for x in items]
             + [(x["json"], f"{ds}/{x['encoder']}.json") for x in items if x["json"]])
            for ds, items in sorted(feats.items())]
    if include_logits and logits:
        jobs.append(("heldout_logits.zip", [(p, f"pools/{os.path.basename(p)}") for p in logits]))
    for name, members in jobs:
        path = os.path.join(out_dir, name)
        side = path + ".members.json"
        if os.path.exists(path) and os.path.exists(side):
            rec = json.load(open(side))
            if rec.get("archive_sha256") == sha256_file(path):
                manifest["archives"][name] = rec
                if verbose:
                    print(f"  {name}: exists, verified - skipped", flush=True)
                continue
        t0 = time.time()
        inner = {}
        with zipfile.ZipFile(path + ".part", "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for src, arc in members:
                inner[arc] = {"sha256": sha256_file(src), "bytes": os.path.getsize(src)}
                z.write(src, arc)
        os.replace(path + ".part", path)
        rec = {"archive_sha256": sha256_file(path), "archive_md5": md5_file(path),
               "archive_bytes": os.path.getsize(path), "members": inner}
        with open(side, "w") as fh:
            json.dump(rec, fh, indent=1)
        manifest["archives"][name] = rec
        if verbose:
            print(f"  {name}: {len(inner)} files, {rec['archive_bytes'] / 1e9:.2f} GB "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    with open(os.path.join(out_dir, "zenodo_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def verify(out_dir, verbose=True):
    """Re-open every archive and re-hash every member against the manifest."""
    man = json.load(open(os.path.join(out_dir, "zenodo_manifest.json")))
    bad = []
    for name, rec in man["archives"].items():
        path = os.path.join(out_dir, name)
        if sha256_file(path) != rec["archive_sha256"]:
            bad.append((name, "archive hash"))
            continue
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            for arc, info in rec["members"].items():
                if arc not in names:
                    bad.append((name, f"missing {arc}"))
                    continue
                h = hashlib.sha256()
                with z.open(arc) as fh:
                    for b in iter(lambda: fh.read(CHUNK), b""):
                        h.update(b)
                if h.hexdigest() != info["sha256"]:
                    bad.append((name, f"hash {arc}"))
        if verbose:
            print(f"  {name}: {'OK' if not any(b[0] == name for b in bad) else 'FAILED'}", flush=True)
    return bad


def write_readme(out_dir, title, github_url, paper_title, summary_dict):
    lines = [f"# {title}", "",
             f"Frozen-encoder features and per-fold held-out logits for the study *{paper_title}*.",
             "", f"Code, registration, per-pool records and the verification gate: {github_url}", "",
             "## Contents", "",
             f"- `features_<dataset>.zip`: one archive per dataset ({summary_dict['datasets']} datasets, "
             f"{summary_dict['feature_files']} feature files). Each holds `<dataset>/<encoder>.npy` (float32, "
             "rows in the order of the dataset manifest in the GitHub repository) and `<dataset>/<encoder>.json` "
             "(encoder source, preprocessing, pooling, load report, feature statistics, extraction time).",
             "- `heldout_logits.zip`: `pools/<dataset>_logits.npz`, the per-fold held-out probe logits of every "
             "encoder, with labels and fold masks, as used by the analysis.",
             "- `zenodo_manifest.json`: SHA-256 of every archive and of every file inside it.", "",
             "## Use", "",
             "`python code/fetch_features.py --record <this record's id> --root <dir>` in the GitHub repository "
             "downloads, verifies and unpacks everything into the layout the notebooks expect.", "",
             "## Provenance and licences", "",
             "Features were extracted from publicly available datasets with publicly released encoders, "
             "some of them gated, under their respective licences. Users must comply with the terms of the "
             "original datasets and models; see the GitHub repository for sources.", "",
             "## Per dataset", "", "| dataset | encoders | GB |", "|---|---|---|"]
    lines += [f"| {ds} | {v['encoders']} | {v['gb']} |" for ds, v in summary_dict["per_dataset"].items()]
    with open(os.path.join(out_dir, "README.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- Zenodo
class Zenodo:
    """Minimal client for the Zenodo deposit REST API. Creates and fills a DRAFT;
    it never publishes (publishing on Zenodo is irreversible)."""

    def __init__(self, token, sandbox=False, session=None):
        import requests
        self.base = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"
        self.web = self.base.replace("/api", "")
        self.s = session or requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"

    def create_draft(self):
        r = self.s.post(f"{self.base}/deposit/depositions", json={})
        r.raise_for_status()
        return r.json()

    def get(self, dep_id):
        r = self.s.get(f"{self.base}/deposit/depositions/{dep_id}")
        r.raise_for_status()
        return r.json()

    def uploaded(self, dep):
        return {f["filename"]: f.get("checksum") for f in dep.get("files", [])}

    def upload(self, dep, path, verbose=True):
        """Stream one file into the draft's bucket; Zenodo returns its MD5, which is
        compared with the local file's."""
        name = os.path.basename(path)
        t0 = time.time()
        with open(path, "rb") as fh:
            r = self.s.put(f"{dep['links']['bucket']}/{name}", data=fh,
                           headers={"Content-Type": "application/octet-stream"})
        r.raise_for_status()
        remote = r.json().get("checksum", "")
        ok = remote.replace("md5:", "") == md5_file(path)
        if verbose:
            print(f"  uploaded {name} ({os.path.getsize(path) / 1e9:.2f} GB, "
                  f"{(time.time() - t0) / 60:.1f} min) checksum {'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            raise RuntimeError(f"Zenodo checksum mismatch for {name}")
        return remote

    def set_metadata(self, dep_id, metadata):
        r = self.s.put(f"{self.base}/deposit/depositions/{dep_id}", json={"metadata": metadata})
        if r.status_code >= 400:
            raise RuntimeError(f"metadata rejected ({r.status_code}): {r.text[:500]}")
        return r.json()


def upload_all(out_dir, token, metadata, sandbox=False, dep_id=None, verbose=True):
    """Upload every file in out_dir (except sidecars) to a draft. Re-running with
    the same dep_id skips files already uploaded with a matching checksum."""
    zc = Zenodo(token, sandbox=sandbox)
    dep = zc.get(dep_id) if dep_id else zc.create_draft()
    have = zc.uploaded(dep)
    files = sorted(p for p in glob.glob(os.path.join(out_dir, "*"))
                   if os.path.isfile(p) and not p.endswith((".members.json", ".part")))
    if len(files) > 100:
        raise RuntimeError(f"{len(files)} files: Zenodo allows at most 100 per record")
    for p in files:
        name = os.path.basename(p)
        if name in have and have[name] and have[name].replace("md5:", "") == md5_file(p):
            if verbose:
                print(f"  {name}: already in the draft - skipped", flush=True)
            continue
        zc.upload(dep, p, verbose=verbose)
    zc.set_metadata(dep["id"], metadata)
    url = f"{zc.web}/deposit/{dep['id']}"
    if verbose:
        print(f"\nDRAFT ready (not published): {url}")
    return {"id": dep["id"], "url": url}
