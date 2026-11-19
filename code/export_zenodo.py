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
def build(out_dir, feats, logits, include_logits=True, exclude_logits_of=(), verbose=True):
    """Write one archive per dataset (+ logits) and the manifest. Resumable: an
    archive that already exists and verifies against its sidecar is skipped."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "archives": {},
                "encoders_excluded": sorted(exclude_logits_of)}
    jobs = [(f"features_{ds}.zip", [(x["npy"], f"{ds}/{x['encoder']}.npy") for x in items]
             + [(x["json"], f"{ds}/{x['encoder']}.json") for x in items if x["json"]])
            for ds, items in sorted(feats.items())]
    if include_logits and logits:
        if exclude_logits_of:
            import numpy as np
            tmp = os.path.join(out_dir, "_filtered_logits")
            os.makedirs(tmp, exist_ok=True)
            kept = []
            for p in logits:
                z = np.load(p)
                keep = {k: z[k] for k in z.files if not (k.startswith("Z_") and k[2:] in exclude_logits_of)}
                q = os.path.join(tmp, os.path.basename(p))
                np.savez_compressed(q, **keep)
                kept.append(q)
            logits = kept
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


def add_file(out_dir, path, name=None):
    """Copy a file into the export folder unchanged and record its checksums."""
    import shutil
    dst = os.path.join(out_dir, name or os.path.basename(path))
    shutil.copy2(path, dst)
    mp = os.path.join(out_dir, "zenodo_manifest.json")
    man = json.load(open(mp))
    man.setdefault("other_files", {})[os.path.basename(dst)] = {
        "sha256": sha256_file(dst), "md5": md5_file(dst), "bytes": os.path.getsize(dst)}
    with open(mp, "w") as fh:
        json.dump(man, fh, indent=1)
    return dst


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


def write_readme(out_dir, title, github_url, paper_title, summary_dict, excluded=(), code_snapshot=None):
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
             "- `zenodo_manifest.json`: SHA-256 of every archive and of every file inside it.", ""]
    if code_snapshot:
        lines[-1:-1] = [f"- `{code_snapshot}`: snapshot of the code repository at the time of deposit."]
    if excluded:
        lines += ["## Encoders not included", "",
                  "As a precaution, features and held-out logits of the following encoders are not redistributed: "
                  "they are distributed under gated licences that may restrict redistributing their outputs: " + ", ".join(f"`{e}`" for e in excluded)
                  + ". They are regenerated from the released code after obtaining each model from its authors "
                  "(notebook 01).", ""]
    lines += [
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
def detect_site(token):
    """Return 'zenodo' or 'sandbox' for the site that accepts this token, or None."""
    import requests
    for site, base in (("zenodo", "https://zenodo.org/api"), ("sandbox", "https://sandbox.zenodo.org/api")):
        r = requests.get(f"{base}/deposit/depositions", params={"size": 1},
                         headers={"Authorization": f"Bearer {token}"}, timeout=60)
        if r.status_code == 200:
            return site
    return None


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
        r = self.s.post(f"{self.base}/deposit/depositions", json={"metadata": {"prereserve_doi": True}})
        r.raise_for_status()
        return r.json()

    def get(self, dep_id):
        r = self.s.get(f"{self.base}/deposit/depositions/{dep_id}")
        r.raise_for_status()
        return r.json()

    def uploaded(self, dep):
        return {f["filename"]: f.get("checksum") for f in dep.get("files", [])}

    def upload(self, dep, path, verbose=True, attempts=3, wait=20, local_md5=None):
        """Stream one file into the draft's bucket; Zenodo returns its MD5, which is
        compared with the local file's. A network error, a 5xx answer (e.g. a 504
        gateway time-out) or a checksum mismatch is retried up to `attempts` times;
        re-sending a file to the same name replaces it."""
        name = os.path.basename(path)
        local = local_md5 or md5_file(path)
        last = None
        for attempt in range(1, attempts + 1):
            t0 = time.time()
            try:
                with open(path, "rb") as fh:
                    r = self.s.put(f"{dep['links']['bucket']}/{name}", data=fh,
                                   headers={"Content-Type": "application/octet-stream"}, timeout=(60, None))
                if r.status_code >= 500:
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                remote = r.json().get("checksum", "")
                if remote.replace("md5:", "") != local:
                    raise RuntimeError("checksum MISMATCH")
                mins = (time.time() - t0) / 60
                size = os.path.getsize(path)
                if verbose:
                    print(f"  uploaded {name} ({size / 1e9:.2f} GB, {mins:.1f} min, "
                          f"{size / 1e6 / max(mins * 60, 1e-9):.2f} MB/s) checksum OK", flush=True)
                return remote
            except Exception as e:      # noqa: BLE001 - every failure is retried, then re-raised
                last = e
                if verbose:
                    print(f"  {name}: attempt {attempt}/{attempts} failed ({str(e)[:120]})", flush=True)
                if attempt < attempts:
                    time.sleep(wait * attempt)
        raise RuntimeError(f"{name} did not upload after {attempts} attempts: {last}")

    def open_drafts(self):
        r = self.s.get(f"{self.base}/deposit/depositions", params={"size": 50})
        r.raise_for_status()
        return [d for d in r.json() if not d.get("submitted")]

    def set_metadata(self, dep_id, metadata):
        r = self.s.put(f"{self.base}/deposit/depositions/{dep_id}", json={"metadata": metadata})
        if r.status_code >= 400:
            raise RuntimeError(f"metadata rejected ({r.status_code}): {r.text[:500]}")
        return r.json()


def mirror(src_dir, dst_dir, verbose=True):
    """Copy the export folder (archives, manifest, sidecars, README, code snapshot)
    to a second location, e.g. Google Drive, so that a runtime reset does not
    force a rebuild. Each copy is checked against the MD5 of its source; a file
    already present with the same size and MD5 is skipped."""
    import shutil
    os.makedirs(dst_dir, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(src_dir, "*"))):
        if not os.path.isfile(p) or p.endswith(".part"):
            continue
        q = os.path.join(dst_dir, os.path.basename(p))
        want = md5_file(p)
        if os.path.exists(q) and os.path.getsize(q) == os.path.getsize(p) and md5_file(q) == want:
            if verbose:
                print(f"  {os.path.basename(p)}: already on the mirror - skipped", flush=True)
            continue
        t0 = time.time()
        shutil.copyfile(p, q + ".part")
        os.replace(q + ".part", q)
        if md5_file(q) != want:
            raise RuntimeError(f"copy of {os.path.basename(p)} to {dst_dir} is corrupt")
        if verbose:
            print(f"  {os.path.basename(p)}: copied ({os.path.getsize(q) / 1e9:.2f} GB, "
                  f"{(time.time() - t0) / 60:.1f} min)", flush=True)


def _upload_files(out_dir):
    files = sorted(p for p in glob.glob(os.path.join(out_dir, "*"))
                   if os.path.isfile(p) and not p.endswith((".members.json", ".part", ".state.json")))
    if len(files) > 100:
        raise RuntimeError(f"{len(files)} files: Zenodo allows at most 100 per record")
    return files


def _doi(dep):
    return ((dep.get("metadata") or {}).get("prereserve_doi") or {}).get("doi")


def upload_all(out_dir, token, metadata, sandbox=False, dep_id=None, verbose=True,
               state_path=None, n_parallel=1, new_draft=False):
    """Upload every file in out_dir (except sidecars) to a Zenodo DRAFT and set its
    metadata. Never publishes.

    Which draft: `dep_id` if given; otherwise the draft recorded in `state_path`
    (written as soon as a draft is created, so a disconnect never loses it);
    otherwise an open draft with the same title. If none of these exists but the
    account has other open drafts, it stops and lists them rather than create a
    second draft with a second DOI, unless `new_draft=True`.

    Re-running skips files already in the draft with a matching checksum. Each
    file is retried on failure (Zenodo.upload); with n_parallel > 1 several files
    upload at once, each over its own connection."""
    zc = Zenodo(token, sandbox=sandbox)
    md = dict(metadata); md["prereserve_doi"] = True
    dep = None
    if dep_id:
        dep = zc.get(dep_id)
    elif state_path and os.path.exists(state_path):
        st = json.load(open(state_path))
        if st.get("sandbox") == bool(sandbox):
            d = zc.get(st["id"])
            if not d.get("submitted"):
                dep = d
                if verbose:
                    print(f"reusing draft {dep['id']} recorded in {state_path}", flush=True)
    if dep is None and not new_draft:
        drafts = zc.open_drafts()
        same = [d for d in drafts if (d.get("title") or (d.get("metadata") or {}).get("title")) == md.get("title")]
        if same:
            dep = zc.get(same[0]["id"])
            if verbose:
                print(f"reusing open draft {dep['id']} with the same title", flush=True)
        elif drafts:
            lines = "\n".join(f"  {d['id']}  {d.get('title') or '(no title yet)'}  "
                              f"{len(d.get('files', []))} files  created {d.get('created', '')[:16]}" for d in drafts)
            raise RuntimeError("This account already has open drafts:\n" + lines +
                               "\nSet DEPOSITION_ID to the one to continue, or NEW_DRAFT = True to start another.")
    if dep is None:
        dep = zc.create_draft()
    if dep.get("submitted"):
        raise RuntimeError(f"deposition {dep['id']} is already published; its files cannot change")
    dep = zc.set_metadata(dep["id"], md)            # title + reserved DOI visible from the start
    doi = _doi(dep)
    url = f"{zc.web}/deposit/{dep['id']}"
    if state_path:
        os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
        with open(state_path, "w") as fh:
            json.dump({"id": dep["id"], "doi": doi, "url": url, "sandbox": bool(sandbox),
                       "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, fh, indent=1)
    if verbose:
        print(f"draft {dep['id']}  reserved DOI {doi}\n  {url}" + (f"\n  saved to {state_path}" if state_path else ""),
              flush=True)

    have = zc.uploaded(dep)
    todo, md5s = [], {}
    for p in _upload_files(out_dir):
        name = os.path.basename(p)
        local = md5s[name] = md5_file(p)
        if have.get(name) and have[name].replace("md5:", "") == local:
            if verbose:
                print(f"  {name}: already in the draft - skipped", flush=True)
        else:
            todo.append((p, local))
    gb = sum(os.path.getsize(p) for p, _ in todo) / 1e9
    if verbose:
        print(f"{len(todo)} files to upload ({gb:.2f} GB), {max(1, n_parallel)} at a time", flush=True)
    t0, failed = time.time(), []
    if n_parallel <= 1:
        for p, local in todo:
            try:
                zc.upload(dep, p, verbose=verbose, local_md5=local)
            except Exception as e:      # noqa: BLE001
                failed.append(os.path.basename(p)); print(f"  FAILED {os.path.basename(p)}: {e}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=n_parallel) as ex:
            futs = {ex.submit(Zenodo(token, sandbox=sandbox).upload, dep, p, verbose, 3, 20, local): p
                    for p, local in todo}
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception as e:  # noqa: BLE001
                    failed.append(os.path.basename(futs[f])); print(f"  FAILED {os.path.basename(futs[f])}: {e}", flush=True)
    mins = (time.time() - t0) / 60
    if failed:
        raise RuntimeError(f"{len(failed)} file(s) failed: {failed}. Re-run: finished files are skipped "
                           f"and the same draft ({dep['id']}) is reused.")
    final = zc.uploaded(zc.get(dep["id"]))
    missing = [n for n, m in md5s.items() if (final.get(n) or "").replace("md5:", "") != m]
    if missing:
        raise RuntimeError(f"not in the draft with a matching checksum: {missing}; re-run")
    if verbose:
        print(f"\nall {len(final)} files in the draft, checksums match "
              f"({gb * 1000 / max(mins * 60, 1e-9):.2f} MB/s over {mins:.1f} min for this run)")
        print(f"DRAFT ready (not published): {url}\nReserved DOI: {doi}")
    return {"id": dep["id"], "url": url, "doi": doi}
