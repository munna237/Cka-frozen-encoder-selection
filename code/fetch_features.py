"""Download the features and held-out logits from Zenodo, verify every checksum,
and set up a project folder in which notebooks 04-08 run unchanged.

    python code/fetch_features.py --record 1234567 --root ~/cka_study
    export V3_DRIVE=~/cka_study          # the notebooks' config reads this
    python code/verify_v3.py --root .    # still works from the repository alone

Resulting layout (the one the code expects under <root>/TMLR_v3):

    code/                       copied from this repository (hash-checked by the registration)
    prereg.json                 the registration
    manifests/                  row order, labels and groups of every dataset
    features/<dataset>/<encoder>.npy, .json
    results/pools/<dataset>.json, <dataset>_logits.npz
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

CHUNK = 1 << 22
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def record_files(record, sandbox=False, session=None):
    import requests
    s = session or requests.Session()
    base = "https://sandbox.zenodo.org" if sandbox else "https://zenodo.org"
    r = s.get(f"{base}/api/records/{record}", timeout=60)
    r.raise_for_status()
    out = []
    for f in r.json().get("files", []):
        name = f.get("key") or f.get("filename")
        link = (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download")
        out.append({"name": name, "url": link, "size": f.get("size"), "checksum": f.get("checksum")})
    return out


def download(url, dst, session=None, retries=5):
    """Resumable download (HTTP Range); keeps <dst>.part until complete."""
    import requests
    s = session or requests.Session()
    part = dst + ".part"
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            with s.get(url, stream=True, timeout=120,
                       headers={"Range": f"bytes={have}-"} if have else {}) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(part, mode) as fh:
                    for b in r.iter_content(CHUNK):
                        fh.write(b)
            break
        except Exception as e:
            print(f"  retry {attempt + 1}/{retries} for {os.path.basename(dst)}: {e}")
            time.sleep(10)
    os.replace(part, dst)
    return dst


def setup_project(root):
    """Copy what the notebooks need from this repository into <root>/TMLR_v3."""
    proj = os.path.join(root, "TMLR_v3")
    for d in ("code", "manifests", os.path.join("results", "pools"), "features", "logs"):
        os.makedirs(os.path.join(proj, d), exist_ok=True)
    for p in glob.glob(os.path.join(REPO, "code", "*.py")):
        shutil.copy2(p, os.path.join(proj, "code"))
    for p in glob.glob(os.path.join(REPO, "manifests", "*")):
        shutil.copy2(p, os.path.join(proj, "manifests"))
    for p in glob.glob(os.path.join(REPO, "results", "pools", "*.json")):
        shutil.copy2(p, os.path.join(proj, "results", "pools"))
    for p in ("battery.json", "paper_tables.json"):
        src = os.path.join(REPO, "results", p)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(proj, "results"))
    shutil.copy2(os.path.join(REPO, "registration", "prereg.json"), os.path.join(proj, "prereg.json"))
    return proj


def unpack(zip_path, proj, members_sha):
    """Extract an archive, re-hashing every member against the manifest."""
    with zipfile.ZipFile(zip_path) as z:
        for arc in z.namelist():
            if arc.startswith("pools/"):
                dst = os.path.join(proj, "results", arc)
            else:
                dst = os.path.join(proj, "features", arc)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(arc) as src, open(dst + ".part", "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            os.replace(dst + ".part", dst)
            want = members_sha.get(arc)
            if want and sha256_file(dst) != want:
                raise RuntimeError(f"checksum mismatch after unpacking {arc}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True, help="Zenodo record id")
    ap.add_argument("--root", required=True, help="folder to create the project in")
    ap.add_argument("--only", nargs="*", default=None,
                    help="dataset names to fetch (default: all); logits are always fetched")
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--keep-archives", action="store_true")
    a = ap.parse_args(argv)
    root = os.path.abspath(os.path.expanduser(a.root))
    dl = os.path.join(root, "zenodo_download")
    os.makedirs(dl, exist_ok=True)
    proj = setup_project(root)
    files = {f["name"]: f for f in record_files(a.record, a.sandbox)}
    if "zenodo_manifest.json" not in files:
        sys.exit("record has no zenodo_manifest.json")
    man_path = download(files["zenodo_manifest.json"]["url"], os.path.join(dl, "zenodo_manifest.json"))
    man = json.load(open(man_path))
    wanted = [n for n in man["archives"]
              if n == "heldout_logits.zip" or a.only is None
              or n[len("features_"):-len(".zip")] in set(a.only)]
    for name in wanted:
        dst = os.path.join(dl, name)
        rec = man["archives"][name]
        if not (os.path.exists(dst) and sha256_file(dst) == rec["archive_sha256"]):
            print(f"downloading {name} ({rec['archive_bytes'] / 1e9:.2f} GB) ...", flush=True)
            download(files[name]["url"], dst)
        if sha256_file(dst) != rec["archive_sha256"]:
            sys.exit(f"{name}: archive checksum mismatch")
        unpack(dst, proj, {k: v["sha256"] for k, v in rec["members"].items()})
        print(f"  {name}: verified and unpacked ({len(rec['members'])} files)", flush=True)
        if not a.keep_archives:
            os.remove(dst)
    print(f"\nproject ready: {proj}\nnext: export V3_DRIVE={root}  (then run notebooks 04-08 or code/ directly)")


if __name__ == "__main__":
    main()
