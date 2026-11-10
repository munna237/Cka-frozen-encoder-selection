"""Pre-registration of the analysis.

register() writes prereg.json with the PREREG parameters, the dataset/encoder
lists and a SHA-256 of every file that can change a reported number. Upload
that file to OSF (or commit and tag it) BEFORE running notebook 04: the
timestamp from a third party is what makes "pre-registered" checkable.

check() is called by notebook 04. It refuses to proceed if any frozen file or
parameter changed, unless a deviation is declared, in which case the deviation
text is written into every results file.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os

import config as CFG

FROZEN_FILES = ["config.py", "cka.py", "folds.py", "fusion.py", "pairs.py", "inference.py",
                "analysis.py", "run_pipeline.py", "analyse_all.py"]


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def snapshot():
    files = {f: _sha(os.path.join(CFG.CODE_DIR, f)) for f in FROZEN_FILES
             if os.path.exists(os.path.join(CFG.CODE_DIR, f))}
    return {
        "params": CFG.PREREG,
        "datasets": {"pathology": list(CFG.PATHOLOGY_DATASETS),
                     "vision": list(CFG.VISION_DATASETS),
                     "text": list(CFG.TEXT_DATASETS)},
        "encoders": {"pathology": list(CFG.PATHOLOGY_ENCODERS),
                     "vision": list(CFG.VISION_ENCODERS),
                     "text": list(CFG.TEXT_ENCODERS)},
        "file_sha256": files,
    }


def register(overwrite=False):
    if os.path.exists(CFG.PREREG_PATH) and not overwrite:
        with open(CFG.PREREG_PATH) as fh:
            old = json.load(fh)
        print(f"already registered at {old['registered_utc']} -- not overwriting.")
        return old
    snap = snapshot()
    snap["registered_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    blob = json.dumps({k: snap[k] for k in ("params", "datasets", "encoders", "file_sha256")},
                      sort_keys=True).encode()
    snap["registration_hash"] = hashlib.sha256(blob).hexdigest()
    os.makedirs(os.path.dirname(CFG.PREREG_PATH), exist_ok=True)
    with open(CFG.PREREG_PATH, "w") as fh:
        json.dump(snap, fh, indent=1)
    print(f"registered: {CFG.PREREG_PATH}")
    print(f"registration hash: {snap['registration_hash']}")
    return snap


def check(deviation_note: str | None = None):
    if not os.path.exists(CFG.PREREG_PATH):
        raise RuntimeError("no prereg.json -- run notebook 00 (register) first")
    with open(CFG.PREREG_PATH) as fh:
        reg = json.load(fh)
    now = snapshot()
    diffs = []
    for f, h in reg["file_sha256"].items():
        if now["file_sha256"].get(f) != h:
            diffs.append(f"file changed: {f}")
    if json.dumps(reg["params"], sort_keys=True) != json.dumps(now["params"], sort_keys=True):
        diffs.append("PREREG parameters changed")
    for k in ("datasets", "encoders"):
        if reg[k] != now[k]:
            diffs.append(f"{k} list changed")
    status = {"registered_utc": reg["registered_utc"],
              "registration_hash": reg["registration_hash"],
              "deviations": diffs, "deviation_note": deviation_note}
    if diffs and not deviation_note:
        raise RuntimeError("analysis differs from the registration:\n  - " + "\n  - ".join(diffs)
                           + "\nEither restore the registered files, or set DEVIATION_NOTE "
                             "in notebook 04 to a sentence explaining the change (it is "
                             "written into every results file).")
    print("pre-registration check:", "PASSED (no deviations)" if not diffs else
          f"{len(diffs)} DECLARED deviation(s)")
    return status
