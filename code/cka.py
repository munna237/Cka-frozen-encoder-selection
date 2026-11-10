"""Linear Centered Kernel Alignment (Kornblith et al., 2019).

Memory/scaling notes for Colab Pro (2 vCPU, ~13-25 GB RAM):
  The d x d Gram formulation below costs O(n * di * dj) and O(di * dj) memory,
  which is independent of n for storage. For a 20k-sample pool of 24 encoders
  (276 pairs) this is seconds, not hours.

  The paper's released subsets reproduce full-sample CKA to within 0.013
  (half-precision quantisation) with a subsampling SD of 0.003-0.006. We
  therefore compute the full pairwise matrix on a fixed n_cap subsample and
  report the subsampling SD explicitly, rather than pretending to exactness.
"""

from __future__ import annotations

import numpy as np


def center(F: np.ndarray) -> np.ndarray:
    """Column-center a feature matrix (n x d)."""
    F = np.asarray(F, dtype=np.float64)
    return F - F.mean(axis=0, keepdims=True)


def linear_cka(Fi: np.ndarray, Fj: np.ndarray) -> float:
    """Linear CKA between two feature matrices on the *same* inputs, same order.

        CKA = ||Fj^T Fi||_F^2 / (||Fi^T Fi||_F * ||Fj^T Fj||_F)

    Both matrices are column-centred first. Returns a float in [0, 1].
    """
    Fi = center(Fi)
    Fj = center(Fj)
    if Fi.shape[0] != Fj.shape[0]:
        raise ValueError(
            f"CKA requires the same number of inputs, got {Fi.shape[0]} and {Fj.shape[0]}"
        )
    G = Fj.T @ Fi                      # (dj, di) -- the only n-dependent cost
    num = float(np.sum(G * G))         # ||G||_F^2
    di = float(np.sum((Fi.T @ Fi) ** 2))
    dj = float(np.sum((Fj.T @ Fj) ** 2))
    if di <= 0.0 or dj <= 0.0:
        raise ValueError("degenerate feature matrix (zero self-alignment)")
    return num / (np.sqrt(di) * np.sqrt(dj))


def pairwise_cka_matrix(feats: dict[str, np.ndarray], n_cap: int | None = None,
                        seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """Full pairwise CKA matrix over a dict {name: features}.

    All feature matrices must be row-aligned (same inputs, same order). If
    n_cap is given, a single fixed subsample of rows is used for every pair so
    that all entries are mutually comparable.
    """
    names = sorted(feats)
    mats = {k: np.asarray(feats[k], dtype=np.float64) for k in names}
    ns = {m.shape[0] for m in mats.values()}
    if len(ns) != 1:
        raise ValueError(f"feature matrices are not row-aligned: row counts {ns}")
    n = ns.pop()

    if n_cap is not None and n_cap < n:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=n_cap, replace=False))
        mats = {k: v[idx] for k, v in mats.items()}

    K = len(names)
    C = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            v = linear_cka(mats[names[i]], mats[names[j]])
            C[i, j] = C[j, i] = v
    return C, names


def cka_subsampling_sd(Fi: np.ndarray, Fj: np.ndarray, n_draws: int = 50,
                       n_sub: int = 2000, seed: int = 0) -> float:
    """SD of the CKA estimate across independent subsamples of n_sub inputs.

    This is the uncertainty the paper reports (0.003-0.006). Report it next to
    every CKA gap you interpret: a gap that does not exceed ~3 SD carries no
    information, which is exactly the failure mode of the MHIST comparison
    (gap 0.012) in the rejected version.
    """
    Fi = np.asarray(Fi); Fj = np.asarray(Fj)
    n = Fi.shape[0]
    n_sub = min(n_sub, n)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_draws)
    for d in range(n_draws):
        idx = rng.choice(n, size=n_sub, replace=False)
        vals[d] = linear_cka(Fi[idx], Fj[idx])
    return float(vals.std(ddof=1))
