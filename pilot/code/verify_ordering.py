#!/usr/bin/env python3
"""
verify_ordering.py -- assert that the shipped label vectors match the raw datasets.

The fold indices in splits/ are POSITIONAL. If a user reconstructs a dataset in a
different row order from ours, every fold silently changes membership and every
downstream number moves without any error being raised. This script is the check
against that: it rebuilds each label vector from the raw dataset in the exact
order documented in README.md, and asserts an elementwise match against the
labels shipped in logits/<dataset>/labels.npy.

The raw datasets are not redistributable, so this script is not run by
verify_claims.py. Point it at your own copies:

    python verify_ordering.py --bracs   /path/to/BRACS_RoI \
                              --sicapv2 /path/to/SICAPv2 \
                              --mhist   /path/to/MHIST

Any subset of the three flags may be given; datasets not supplied are skipped.
Exit code 0 means every supplied dataset reproduces the shipped order exactly.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BRACS_CLASSES = ["N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC"]
SICAP_CLASSES = ["NC", "G3", "G4", "G5"]


def shipped(ds):
    return np.load(os.path.join(ROOT, "logits", ds, "labels.npy"))


def report(ds, rebuilt, ref):
    if rebuilt is None:
        print(f"[ skip ] {ds:8s} not supplied")
        return None
    if len(rebuilt) != len(ref):
        print(f"[ FAIL ] {ds:8s} length {len(rebuilt)} != shipped {len(ref)}")
        return False
    bad = int((rebuilt != ref).sum())
    if bad:
        first = int(np.flatnonzero(rebuilt != ref)[0])
        print(f"[ FAIL ] {ds:8s} {bad} of {len(ref)} labels differ; "
              f"first at index {first} "
              f"(rebuilt {rebuilt[first]}, shipped {ref[first]})")
        return False
    print(f"[  ok  ] {ds:8s} {len(ref)} labels match exactly")
    return True


def build_bracs(root):
    """train + val of the RoI release, in that order, class order as above."""
    ys = []
    for split in ["train", "val"]:
        for ci, cname in enumerate(BRACS_CLASSES):
            d = os.path.join(root, split, cname)
            if not os.path.isdir(d):
                d = os.path.join(root, split, f"{ci}_{cname}")
            if not os.path.isdir(d):
                raise FileNotFoundError(f"expected {d}")
            ys += [ci] * len([f for f in sorted(os.listdir(d))
                              if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    return np.array(ys)


def build_sicapv2(root):
    """Train.xlsx followed by Test.xlsx from partition/Test/, in that order."""
    import pandas as pd
    part = os.path.join(root, "partition", "Test")
    frames = [pd.read_excel(os.path.join(part, f)) for f in ["Train.xlsx", "Test.xlsx"]]
    df = pd.concat(frames, ignore_index=True)
    onehot = df[SICAP_CLASSES].values
    return onehot.argmax(1).astype(int)


def build_mhist(root):
    """Row order of the released annotations.csv."""
    import pandas as pd
    df = pd.read_csv(os.path.join(root, "annotations.csv"))
    col = "Majority Vote Label" if "Majority Vote Label" in df.columns else df.columns[1]
    # SSA is the positive class in our encoding; HP is 0.
    return (df[col].astype(str).str.upper().str.strip() == "SSA").astype(int).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bracs")
    ap.add_argument("--sicapv2")
    ap.add_argument("--mhist")
    a = ap.parse_args()

    print("=" * 72)
    print("ORDERING GATE -- shipped labels vs. rebuilt from raw datasets")
    print("=" * 72)

    results = []
    for ds, path, fn in [("bracs", a.bracs, build_bracs),
                         ("sicapv2", a.sicapv2, build_sicapv2),
                         ("mhist", a.mhist, build_mhist)]:
        if path is None:
            results.append(report(ds, None, None))
            continue
        try:
            rebuilt = fn(path)
        except Exception as exc:
            print(f"[ FAIL ] {ds:8s} could not rebuild: "
                  f"{exc.__class__.__name__}: {exc}")
            results.append(False)
            continue
        results.append(report(ds, rebuilt, shipped(ds)))

    checked = [r for r in results if r is not None]
    print("=" * 72)
    if not checked:
        print("Nothing checked. Supply at least one dataset path.")
        return 1
    if all(checked):
        print(f"Ordering verified for {len(checked)} dataset(s).")
        return 0
    print("ORDERING MISMATCH -- the shipped fold indices do not apply to your "
          "reconstruction. Do not compare numbers until this is resolved.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
