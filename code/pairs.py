"""Pair construction: the design decisions that determine whether the study has power.

The rejected version's fatal structural weakness was that it produced exactly
THREE matched comparisons, one of which (MHIST, CKA gap 0.012) had no separation
at all. Everything here is built to produce as many usable matched pairs as the
pool allows, and to record *why* each excluded pair was excluded, so the selection
is auditable rather than looking hand-picked.

Definitions (unchanged from v2 where it was sound):
  anchor   : reference encoder, default = highest individual held-out accuracy.
  base     : anchor + the median-CKA pool member, fixed BEFORE any pair is chosen
             (the base member is therefore not available as a candidate).
  marginal utility of candidate c, per fold f:
             U(c, f) = acc( fuse(base_f + c_f) ) - acc( fuse(base_f) )
  matched pair (a, b): |indiv_acc(a) - indiv_acc(b)| <= comp_tol
                       and |CKA(a, anchor) - CKA(b, anchor)| >= cka_min
  primary per-pair estimand:
             D(pair) = mean_f [ U(higher-CKA member, f) - U(lower-CKA member, f) ]
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class PairSpec:
    pool: str
    anchor: str
    base: tuple
    high: str          # member with the higher CKA to the anchor
    low: str           # member with the lower CKA to the anchor
    cka_high: float
    cka_low: float
    cka_gap: float
    acc_high: float
    acc_low: float
    comp_gap: float
    category: str = "matched"

    def to_dict(self):
        d = asdict(self)
        d["base"] = list(self.base)
        return d


def choose_anchor(cka_with: dict[str, float], indiv_acc: dict[str, float],
                  rule: str = "best") -> str:
    """rule: 'best' = highest individual accuracy (default, matches v2),
             'median_cka' = the pool member with median CKA to the rest."""
    if rule == "best":
        return max(indiv_acc, key=lambda k: indiv_acc[k])
    if rule == "median_cka":
        names = sorted(cka_with)
        order = sorted(names, key=lambda k: cka_with[k])
        return order[len(order) // 2]
    raise ValueError(f"unknown anchor rule {rule!r}")


def fix_base(anchor: str, cka_with: dict[str, float]) -> tuple[str, str]:
    """Anchor plus the median-CKA pool member, chosen before any pair is formed.

    v3 FIX: the previous version picked others[len//2] in ALPHABETICAL order when
    the number of non-anchor members was odd, so the "median-CKA member" was in
    fact the alphabetical median. The base now always uses the CKA ordering
    (lower median for an even count; ties broken by name for determinism).
    """
    others = [k for k in cka_with if k != anchor]
    if not others:
        raise ValueError("pool must contain at least one non-anchor member")
    order = sorted(others, key=lambda k: (cka_with[k], k))
    mid = order[(len(order) - 1) // 2]
    return (anchor, mid)


def enumerate_pairs(pool: str, cka_with: dict[str, float], indiv_acc: dict[str, float],
                    comp_tol: float = 1.5, cka_min: float = 0.15,
                    anchor: str | None = None, cka_tol: float | None = None) -> list[PairSpec]:
    """List every candidate pair in the pool with its gaps and category.

    comp_tol : max competence gap (accuracy points) for a pair to be 'matched'.
               v2 used 5 points on general vision and reported 0.63-1.36 in
               pathology; 1.5 is the tighter default now that pools are large.
    cka_min  : min CKA gap for a pair to be informative at all. Pairs below it
               are 'no_separation' and excluded -- this is what MHIST was.
    cka_tol  : if given, the CKA gap must also exceed this many subsampling SDs
               of the CKA estimate. STRONGLY recommended: it is the check whose
               absence let a 0.012 gap be reported alongside a 0.259 gap.
    """
    if anchor is None:
        anchor = choose_anchor(cka_with, indiv_acc)
    base = fix_base(anchor, cka_with)
    cands = [k for k in cka_with if k not in base]

    specs: list[PairSpec] = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            hi, lo = (a, b) if cka_with[a] >= cka_with[b] else (b, a)
            cka_gap = abs(cka_with[a] - cka_with[b])
            comp_gap = abs(indiv_acc[a] - indiv_acc[b])

            if cka_tol is not None and cka_gap < cka_tol:
                cat = "no_separation"
            elif cka_gap < cka_min:
                cat = "cka_gap_too_small"
            elif comp_gap <= comp_tol:
                cat = "matched"
            else:
                cat = "unmatched"

            specs.append(PairSpec(
                pool=pool, anchor=anchor, base=base, high=hi, low=lo,
                cka_high=cka_with[hi], cka_low=cka_with[lo], cka_gap=cka_gap,
                acc_high=indiv_acc[hi], acc_low=indiv_acc[lo], comp_gap=comp_gap,
                category=cat,
            ))
    return specs


def matched_pairs(pool: str, cka_with, indiv_acc, **kw) -> list[PairSpec]:
    return [p for p in enumerate_pairs(pool, cka_with, indiv_acc, **kw)
            if p.category == "matched"]


def primary_pair(specs: list[PairSpec]) -> PairSpec | None:
    """The reported pair per pool: matched, maximising the CKA gap.

    Declaring this rule in advance is what makes Table 9-style auditing possible
    and defeats the 'chosen for its result' objection.
    """
    m = [p for p in specs if p.category == "matched"]
    return max(m, key=lambda p: p.cka_gap) if m else None


def summarise_pool(specs: list[PairSpec]) -> dict:
    """Auditable counts of why pairs were kept or dropped, per pool."""
    out: dict[str, int] = {}
    for p in specs:
        out[p.category] = out.get(p.category, 0) + 1
    out["total"] = len(specs)
    m = [p for p in specs if p.category == "matched"]
    out["matched_cka_gap_min"] = float(min((p.cka_gap for p in m), default=np.nan))
    out["matched_cka_gap_max"] = float(max((p.cka_gap for p in m), default=np.nan))
    out["matched_cka_gap_median"] = float(np.median([p.cka_gap for p in m])) if m else np.nan
    return out
