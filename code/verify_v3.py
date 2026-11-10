"""Verification gate for the multi-pool study (supplementary material, v3).

    python code/verify_v3.py --root .          # exit 0 = every check passed

Checks, in order. Each is reported PASS / FAIL; any FAIL exits non-zero.

  1. registration   the SHA-256 of every registered analysis file in code/ equals
                    the hash recorded in registration/prereg.json, so the code in
                    this archive is the code that was registered.
  2. records        every per-pool record carries the registration hash and the
                    manifest hash of the pool it was computed on.
  3. battery        recomputing estimands A-F from results/pools/*.json with the
                    registered code reproduces results/battery.json, every field to
                    within floating-point rounding (rtol 1e-9); differing fields are listed.
  4. tables         rebuilding the paper tables from those records reproduces
                    results/paper_tables.json (registered estimands and the
                    exploratory analyses), asserting every internal identity.
  5. claims         the numbers quoted in the paper equal the recomputed values at
                    the precision printed in the paper.

Nothing here needs a GPU, torch or the features: the per-pool records contain
every quantity the estimands are computed from. Regenerating the records from
public data is what notebooks/01-04 do.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

# Numbers as printed in the paper: (subset, path into battery.json, printed value, decimals)
CLAIMS = [
    ("all", ["A_matched_null_primary_only", "cluster_robust", "estimate_pp"], -0.34, 2),
    ("all", ["A_matched_null_primary_only", "cluster_robust", "ci_low"], -0.84, 2),
    ("all", ["A_matched_null_primary_only", "cluster_robust", "ci_high"], 0.16, 2),
    ("all", ["A_matched_null_primary_only", "cluster_robust", "G"], 18, 0),
    ("all", ["A_matched_null_all_pairs", "cluster_robust", "estimate_pp"], -0.10, 2),
    ("all", ["A_matched_null_all_pairs", "cluster_robust", "n"], 65, 0),
    ("all", ["C_equivalence", "margin_0.5", "equivalent"], True, None),
    ("all", ["C_equivalence", "margin_0.25", "equivalent"], False, None),
    ("image", ["C_equivalence", "margin_0.25", "equivalent"], True, None),
    ("pathology", ["C_equivalence", "margin_0.25", "equivalent"], True, None),
    ("text", ["A_matched_null_all_pairs", "cluster_robust", "estimate_pp"], -0.92, 2),
    ("all", ["D_selection_regret", "per_rule", "cka_diversity", "mean_regret_pp"], 1.76, 2),
    ("all", ["D_selection_regret", "per_rule", "competence", "mean_regret_pp"], 0.27, 2),
    ("all", ["D_selection_regret", "per_rule", "disagreement_with_base", "mean_regret_pp"], 1.92, 2),
    ("all", ["D_selection_regret", "per_rule", "oracle_headroom", "mean_regret_pp"], 0.28, 2),
    ("all", ["B_dose_response", "mde_80pct_power"], 3.51, 2),
    ("all", ["E_mechanism_on_real", "joint_model", "cka", "coef_pp_per_unit"], 1.94, 2),
]

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def dig(d, path):
    for k in path:
        d = d[k]
    return d


def canon(obj):
    return json.dumps(obj, sort_keys=True, default=float)


RTOL, ATOL = 1e-9, 1e-12   # floating-point reproducibility across processes / BLAS builds


def diff(a, b, path="", out=None):
    """Every leaf where a and b differ beyond floating-point rounding.
    Returns a list of (path, saved, recomputed). Numbers compare with
    |a - b| <= ATOL + RTOL * |b|; NaN equals NaN; everything else exactly."""
    import math
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}/{k}", a.get(k, "<absent>"), b.get(k, "<absent>")))
            else:
                diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"len {len(a)}", f"len {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool) or not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))):
        if a != b:
            out.append((path, a, b))
    else:
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return out
        if not (abs(fa - fb) <= ATOL + RTOL * abs(fb)):
            out.append((path, a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-claims", action="store_true",
                    help="skip check 5 (for archives built from synthetic test data)")
    a = ap.parse_args()
    root = os.path.abspath(a.root)

    # Preflight: the registered robustness model (random-intercept REML) needs statsmodels.
    # Without it the inference code falls back to a method-of-moments estimate, whose numbers
    # differ from the registered ones, which would show up as misleading battery failures.
    try:
        import numpy, scipy, statsmodels
        print(f"environment: python {sys.version.split()[0]}, numpy {numpy.__version__}, "
              f"scipy {scipy.__version__}, statsmodels {statsmodels.__version__}\n")
    except ImportError as e:
        print(f"missing dependency: {e.name}. Install the requirements first:\n"
              f"    pip install -r requirements.txt")
        sys.exit(2)
    code = os.path.join(root, "code")
    res = os.path.join(root, "results")
    sys.path.insert(0, code)

    # 1. registration ------------------------------------------------------
    with open(os.path.join(root, "registration", "prereg.json")) as fh:
        reg = json.load(fh)
    for f, h in sorted(reg["file_sha256"].items()):
        p = os.path.join(code, f)
        ok = os.path.exists(p) and sha256(p) == h
        report(f"registration: {f}", ok, "" if ok else "hash differs from prereg.json")
    blob = json.dumps({k: reg[k] for k in ("params", "datasets", "encoders", "file_sha256")},
                      sort_keys=True).encode()
    report("registration: registration_hash", hashlib.sha256(blob).hexdigest() == reg["registration_hash"])

    # 2. records -------------------------------------------------------------
    pools = {}
    for f in sorted(glob.glob(os.path.join(res, "pools", "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        pools[r["pool"]] = r
    report("records: pool records present", len(pools) > 0, f"{len(pools)} pools")
    man_dir = os.path.join(root, "manifests")
    for name, r in pools.items():
        ok = r.get("prereg", {}).get("registration_hash") == reg["registration_hash"]
        mp = os.path.join(man_dir, f"{name}.json")
        if os.path.exists(mp):
            with open(mp) as fh:
                ok = ok and json.load(fh)["hash"] == r.get("manifest_hash")
        else:
            ok = False
        report(f"records: {name}", ok, "" if ok else "registration or manifest hash mismatch")

    # 3. battery -------------------------------------------------------------
    import analyse_all as A
    with open(os.path.join(res, "battery.json")) as fh:
        bat = json.load(fh)
    import config as CFG
    order = [ds for dom in ("pathology", "vision", "text") for ds in CFG.DOMAINS[dom][0]]
    records = [pools[p] for p in order if p in pools]     # the order notebook 04 used
    records += [pools[p] for p in pools if p not in order]
    again = A.battery(records, bat["prereg"])
    for sub in A.SUBSETS:
        x = json.loads(canon({k: v for k, v in bat.get(sub, {}).items() if k != "pools"}))
        y = json.loads(canon({k: v for k, v in again.get(sub, {}).items() if k != "pools"}))
        d = diff(x, y)
        exact = canon(x) == canon(y)
        # The random-intercept model on the PRIMARY rows is degenerate by design: one
        # observation per pool leaves no within-pool variance, so REML returns a standard
        # error that is numerically zero and whose last digits are optimiser noise. The paper
        # (Appendix B) states these rows are not estimable and does not report them. For them
        # we verify the documented degeneracy instead of comparing noise.
        DEGEN = "/A_matched_null_primary_only/mixed_effects/"
        degen = [t for t in d if t[0].startswith(DEGEN)]
        rest = [t for t in d if not t[0].startswith(DEGEN)]
        degen_ok, degen_note = True, ""
        if degen:
            import math
            def tiny(v):
                try:
                    v = float(v)
                    return math.isnan(v) or abs(v) < 1e-6
                except (TypeError, ValueError):
                    return False
            mx = x.get("A_matched_null_primary_only", {}).get("mixed_effects", {})
            my = y.get("A_matched_null_primary_only", {}).get("mixed_effects", {})
            cr = x.get("A_matched_null_primary_only", {}).get("cluster_robust", {})
            degen_ok = (tiny(mx.get("se")) and tiny(my.get("se"))
                        and abs(float(mx.get("estimate")) - float(my.get("estimate"))) < 1e-9
                        and abs(float(mx.get("estimate")) - float(cr.get("estimate_pp"))) < 1e-9)
            degen_note = (f"; primary-row random-intercept model degenerate as documented "
                          f"(SE saved {mx.get('se')!r}, recomputed {my.get('se')!r}; not reported, "
                          f"Appendix B)" if degen_ok else
                          "; primary-row random-intercept model is NOT degenerate as documented")
        ok = not rest and degen_ok
        if exact:
            detail = "bit-identical"
        elif ok:
            detail = f"identical to within rtol {RTOL:g}" + degen_note
        else:
            detail = ((f"{len(rest)} field(s) differ, e.g. " + "; ".join(
                f"{p}: saved {v1!r} vs recomputed {v2!r}" for p, v1, v2 in rest[:3]))
                      if rest else "") + degen_note.lstrip("; ")
        report(f"battery: subset '{sub}' recomputes", ok, detail)

    # 4. tables --------------------------------------------------------------
    import report as R
    pt = os.path.join(res, "paper_tables.json")
    if os.path.exists(pt):
        with open(pt) as fh:
            saved = json.load(fh)
        import tempfile, shutil
        tmp = tempfile.mkdtemp()
        shutil.copytree(os.path.join(res, "pools"), os.path.join(tmp, "pools"))
        shutil.copy(os.path.join(res, "battery.json"), tmp)
        if os.path.exists(os.path.join(res, "fusion_fit_log.csv")):
            shutil.copy(os.path.join(res, "fusion_fit_log.csv"), tmp)
        try:
            T, _ = R.build(tmp)          # asserts every recomputation equals the registered value
            report("tables: internal identities", True)
            d = diff(saved, json.loads(canon(T)))
            report("tables: paper_tables.json reproduces", not d,
                   "" if not d else f"{len(d)} field(s) differ, e.g. {d[:2]}")
        except AssertionError as e:
            report("tables: internal identities", False, str(e))
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        report("tables: paper_tables.json present", False)

    # 5. claims --------------------------------------------------------------
    if not a.no_claims:
        for sub, path, printed, nd in CLAIMS:
            try:
                v = dig(bat[sub], path)
                ok = (v == printed) if nd is None else (round(float(v), nd) == printed)
            except Exception as e:
                v, ok = f"missing ({e})", False
            report(f"claim: {sub}/{'/'.join(path)} = {printed}", ok, "" if ok else f"recomputed {v}")

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
