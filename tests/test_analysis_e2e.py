"""End-to-end analysis test on synthetic pools (CPU, ~2-4 min).

Writes synthetic features + manifests exactly where notebooks 01-03 would, then
runs the real notebook-04 path: prereg.register -> prereg.check -> run_all ->
battery -> write_all -> verify_determinism. Also checks that the prereg guard
blocks an undeclared change and accepts a declared one.
"""

import json
import os
import shutil
import sys

import numpy as np

ROOT = "/tmp/v3_analysis"
shutil.rmtree(ROOT, ignore_errors=True)
os.environ["V3_DRIVE"] = os.path.join(ROOT, "MyDrive")
os.environ["V3_LOCAL"] = os.path.join(ROOT, "local")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "code")
sys.path.insert(0, SRC)

import config as CFG  # noqa: E402

os.makedirs(CFG.CODE_DIR, exist_ok=True)
for f in os.listdir(SRC):
    if f.endswith(".py"):
        shutil.copy(os.path.join(SRC, f), CFG.CODE_DIR)

import imgcache as IC    # noqa: E402
import synth_pool as SP  # noqa: E402

# two synthetic "domains", 4 pools each, 7 members per pool
MEMBERS = [f"synth_{i:02d}" for i in range(7)]
CFG.PATHOLOGY_DATASETS = {f"sp{i}": dict(cap=None, persist=False) for i in range(4)}
CFG.VISION_DATASETS = {f"sv{i}": dict(cap=None, persist=False) for i in range(4)}
CFG.TEXT_DATASETS = {}
enc = {m: dict(kind="timm", id="synthetic") for m in MEMBERS}
CFG.PATHOLOGY_ENCODERS = enc
CFG.VISION_ENCODERS = enc
CFG.TEXT_ENCODERS = {}
CFG.DOMAINS = {"pathology": (CFG.PATHOLOGY_DATASETS, enc), "vision": (CFG.VISION_DATASETS, enc),
               "text": ({}, {})}
CFG.PREREG["n_cap_cka"] = 1500

import extract as X  # noqa: E402

for j, ds in enumerate(list(CFG.PATHOLOGY_DATASETS) + list(CFG.VISION_DATASETS)):
    pool = SP.make_synthetic_pool(n_members=7, n_classes=4, n_per_class=150, d_total=64,
                                  seed=100 + j)
    y = pool.y
    groups = [f"g{i % 30}" for i in range(len(y))] if j % 2 == 0 else None
    man = dict(name=ds, n=len(y), labels=y.tolist(), uids=[f"{ds}_{i}" for i in range(len(y))],
               groups=groups, classes=[str(c) for c in range(4)])
    man["hash"] = IC.manifest_hash(man["uids"], man["labels"])
    os.makedirs(CFG.MANIFEST_DIR, exist_ok=True)
    json.dump(man, open(os.path.join(CFG.MANIFEST_DIR, f"{ds}.json"), "w"))
    for m in MEMBERS:
        d, npy, meta, _ = X._paths(ds, m)
        os.makedirs(d, exist_ok=True)
        np.save(npy, pool.X[m].astype(np.float32))
        json.dump({"manifest_hash": man["hash"], "complete": True, "saved": "t"}, open(meta, "w"))

import analyse_all as A  # noqa: E402
import prereg            # noqa: E402

reg = prereg.register()
status = prereg.check()
assert status["deviations"] == []

# guard: undeclared change must be refused, declared change accepted
p = os.path.join(CFG.CODE_DIR, "pairs.py")
orig = open(p).read()
open(p, "a").write("\n# edit\n")
try:
    prereg.check()
    raise AssertionError("prereg guard failed to block an undeclared change")
except RuntimeError:
    pass
st2 = prereg.check(deviation_note="test deviation")
assert st2["deviations"] == ["file changed: pairs.py"]
open(p, "w").write(orig)
print("PASS  prereg: registers, blocks undeclared change, records declared deviation")

CFG.PREREG["probe_backend"] = "torch"
records, skipped = A.run_all(status, domains=("pathology", "vision"))
assert len(records) == 8 and not skipped, skipped
assert any(r["grouped_folds"] for r in records) and not all(r["grouped_folds"] for r in records)
bat = A.battery(records, status)
for sub in ("all", "pathology", "vision", "image"):
    assert "error" not in bat[sub], (sub, bat[sub])
    assert bat[sub]["D_selection_regret"]["n_pools_scored"] > 0

# Regression test for the Colab failure: a subset in which only ONE pool has
# matched pairs must report A as not estimable, not crash the whole subset.
import run_pipeline as RP  # noqa: E402
one = [dict(r) for r in records]
keep = next(i for i, r in enumerate(one) if r["matched_pairs"])
for i, r in enumerate(one):
    if i != keep:
        r["matched_pairs"] = []
res = RP.analyse_pools(one)
assert "not_estimable" in res["A_matched_null_primary_only"], res["A_matched_null_primary_only"]
assert res["D_selection_regret"]["n_pools_scored"] == len(one)
none = [dict(r, matched_pairs=[]) for r in records]
res0 = RP.analyse_pools(none)
assert "not_estimable" in res0["A_matched_null_all_pairs"]
md_one = A.summary_md(one, {"prereg": status, "all": res}, [])
assert "Not estimable" in md_one
print("PASS  estimands fail independently (1 pool / 0 pools with matched pairs)")
md = A.write_all(records, bat, skipped)
assert os.path.exists(os.path.join(CFG.RESULTS_DIR, "figures", "dose_response.png"))
assert A.verify_determinism(records, bat)
print(md[:1800])

# cached re-run must reuse every pool
records2, _ = A.run_all(status, domains=("pathology", "vision"))
assert [r["pool"] for r in records2] == [r["pool"] for r in records]
print("PASS  analysis: 8 pools, all subsets, summary, figures, determinism, cache reuse")
print("\nall analysis checks passed")
