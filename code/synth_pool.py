"""Synthetic pool generator with EXPLICIT control over CKA and competence.

Two purposes, and the second saves Colab hours:
  1. Scientific: populate the (CKA x competence) plane by construction, which no
     public checkpoint pool does -- the structural reason v2's evidence was thin.
  2. Engineering: a stand-in pool with known ground truth, so the whole analysis
     pipeline is validated end-to-end BEFORE any GPU time is spent.

DESIGN NOTE -- why this differs from v2 Appendix L.
v2 controlled CKA via shared-block *width* (ds) with a common realisation S and
private blocks P_k. Reimplementing that here produced a CKA range of only
0.259-0.352 across a 12-member pool: with fixed total width d and independently
drawn nuisance vectors, the nuisance cross-terms dominate and the ds contrast is
washed out. Since v2 reported reaching CKA 0.08-0.95, its generator must have
scaled the blocks differently than a literal reading of the description implies
(the full parameterisation is in its Appendix L, which is not in the released PDF).

Rather than guess, this version parameterises the quantity that actually controls
linear CKA directly. For centred features F_k = s_k * S + p_k * P_k with S shared
(identical realisation across members) and P_k independent:

    CKA(F_i, F_j) = corr_i * corr_j,   corr_k = s_k^2 / (s_k^2 + p_k^2)

which is exact in the large-n limit when S and P are independent and isotropic.
So specifying corr_k specifies member k's CKA to a fully-shared reference, and
pairs multiply. Competence is controlled independently by the class-signal energy
in the private block. The two knobs are orthogonal by construction, which is what
the design needs -- and `calibrate()` below verifies it numerically rather than
asserting it.

Kept from v2 (correctly): R_k is a random orthogonal matrix applied per member.
Linear CKA is orthogonally invariant, so rotation CANNOT vary CKA -- which is why
v2 rightly rejected the reviewer's suggestion to control CKA by rotation -- and it
is applied only so that no member's informative directions are axis-aligned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cka import linear_cka


@dataclass
class SyntheticPool:
    n_classes: int
    n_per_class: int
    d_total: int
    members: list
    X: dict          # name -> (n, d) features
    y: np.ndarray
    corr: dict       # name -> shared-energy fraction actually realised
    sig: dict        # name -> private class-signal strength actually realised
    phi: float       # fraction of class-signal energy in the shared block
    truth: dict


def _class_signal(y: np.ndarray, n_classes: int, dim: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Per-input class prototype signal in `dim` dimensions, unit total energy."""
    P = rng.normal(size=(n_classes, dim))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P[y] / np.sqrt(max(dim, 1))


def make_synthetic_pool(n_members: int = 24, n_classes: int = 7, n_per_class: int = 400,
                        d_total: int = 384, phi: float = 0.5, seed: int = 0,
                        corr_range=(0.15, 0.95), sig_range=(0.3, 2.5),
                        shared_dim_frac: float = 0.5,
                        cka_dependent_utility: bool = False) -> SyntheticPool:
    """Build a pool spanning CKA and competence near-independently.

    corr_range : range of the shared-energy fraction s^2/(s^2+p^2) per member.
                 Determines each member's CKA to the shared reference; pairwise
                 CKA is the product of the two members' corr values.
    sig_range  : private class-signal strength per member. Determines competence.
    phi        : fraction of total class-signal energy placed in the shared block.
                 This is the mediator v2 found: it sets how correlated members'
                 errors are, and it flips the sign of the CKA-disagreement
                 relation. It is unobservable to a practitioner with frozen features.
    cka_dependent_utility : PIPELINE VALIDATION SWITCH (see tests).
    """
    rng = np.random.default_rng(seed)
    n = n_classes * n_per_class
    y = np.repeat(np.arange(n_classes), n_per_class)
    rng.shuffle(y)

    d_shared = max(8, int(round(d_total * shared_dim_frac)))
    d_priv = max(8, d_total - d_shared)

    # one shared realisation, identical across members -> contributes maximally to CKA
    S_nuis = rng.normal(size=(n, d_shared))
    S_sig = _class_signal(y, n_classes, d_shared, rng)

    corr_target = rng.uniform(corr_range[0], corr_range[1], size=n_members)
    sig_target = rng.uniform(sig_range[0], sig_range[1], size=n_members)

    names = [f"synth_{i:02d}" for i in range(n_members)]
    X, corr, sig = {}, {}, {}
    for i, name in enumerate(names):
        c = float(corr_target[i])
        a = float(sig_target[i])
        # split total energy between shared and private blocks per corr
        s_w = np.sqrt(c)
        p_w = np.sqrt(max(1.0 - c, 1e-6))

        P_nuis = rng.normal(size=(n, d_priv))
        P_sig = _class_signal(y, n_classes, d_priv, rng)

        shared = s_w * (S_nuis + phi * np.sqrt(d_shared) * S_sig)
        private = p_w * (P_nuis + (1.0 - phi) * a * np.sqrt(d_priv) * P_sig)
        F = np.hstack([shared, private])

        # random orthogonal mixing: CKA-invariant, breaks axis alignment
        Q, _ = np.linalg.qr(rng.normal(size=(F.shape[1], F.shape[1])))
        X[name] = F @ Q
        corr[name], sig[name] = c, a

    truth = {
        "corr": corr, "sig": sig, "phi": phi,
        "d_shared": d_shared, "d_priv": d_priv,
        "cka_dependent_utility": bool(cka_dependent_utility),
        "expected_pairwise_cka": "corr_i * corr_j (large-n limit)",
        "NOTE": ("utility was made CKA-dependent by construction: the pipeline MUST "
                 "recover a positive dose-response slope" if cka_dependent_utility
                 else "utility carries no CKA signal: the pipeline MUST return a null"),
    }
    return SyntheticPool(n_classes=n_classes, n_per_class=n_per_class, d_total=d_total,
                         members=names, X=X, y=y, corr=corr, sig=sig, phi=phi, truth=truth)


def calibrate(n_members: int = 16, seed: int = 0, **kw) -> dict:
    """Verify the construction actually delivers the claimed control BEFORE using
    it for science. Returns the realised CKA range, the realised CKA vs. the
    predicted corr_i*corr_j, and the CKA-competence correlation.

    If realised CKA does not track corr_i*corr_j, or if CKA and competence come
    out collinear, the pool is confounded in the same way the real pools were and
    must not be used as a 'controlled' arm.
    """
    pool = make_synthetic_pool(n_members=n_members, seed=seed, **kw)
    names = pool.members
    realised, predicted = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            realised.append(linear_cka(pool.X[names[i]], pool.X[names[j]]))
            predicted.append(pool.corr[names[i]] * pool.corr[names[j]])
    realised, predicted = np.array(realised), np.array(predicted)
    from inference import spearman
    return {
        "n_pairs": len(realised),
        "cka_range": [float(realised.min()), float(realised.max())],
        "corr_realised_vs_predicted": float(np.corrcoef(realised, predicted)[0, 1]),
        "rmse": float(np.sqrt(np.mean((realised - predicted) ** 2))),
        "spearman": spearman(realised, predicted),
        "VERDICT": ("construction delivers the claimed CKA control"
                    if realised.max() - realised.min() > 0.4
                    and np.corrcoef(realised, predicted)[0, 1] > 0.95
                    else "CONSTRUCTION FAILED: CKA not controlled -- do not use"),
    }


def inject_utility_signal(utilities: np.ndarray, cka_gaps: np.ndarray,
                          beta: float = 6.0, seed: int = 0) -> np.ndarray:
    """TEST-ONLY: add a known CKA-dependent component so the analysis has
    something to find. Used by the positive control in test_pipeline_synthetic.py."""
    rng = np.random.default_rng(seed)
    return np.asarray(utilities, float) + beta * np.asarray(cka_gaps, float) \
        + rng.normal(0, 0.05, size=len(utilities))
