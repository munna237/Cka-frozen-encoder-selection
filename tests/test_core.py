"""Unit tests for the analysis core. Run: python tests/test_core.py"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))

import folds as F          # noqa: E402
import fusion              # noqa: E402
import inference as I      # noqa: E402
import pairs as P          # noqa: E402


def test_fix_base_uses_cka_median_not_alphabetical():
    # odd number of non-anchor members: alphabetical median would be "c"
    cka = {"anchor": 0.0, "a": 0.9, "b": 0.1, "c": 0.8, "d": 0.2, "e": 0.5}
    base = P.fix_base("anchor", cka)
    assert base == ("anchor", "e"), base            # CKA order b,d,e,c,a -> median e
    cka2 = {"anchor": 0.0, "a": 0.9, "b": 0.1, "c": 0.8, "d": 0.2}
    assert P.fix_base("anchor", cka2) == ("anchor", "d")   # even: lower median


def test_torch_probe_reaches_sklearn_optimum():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    for C in (7, 2):
        n, d = 2000, 128
        y = rng.integers(0, C, n)
        X = rng.normal(size=(C, d))[y] * 0.2 + rng.normal(size=(n, d))
        m = F.make_folds(y, k=5)
        Zt = F.per_fold_logits(X, y, m, backend="torch")
        # tightly converged sklearn reference on the first fold
        sc = StandardScaler().fit(X[~m[0]])
        ref = LogisticRegression(C=1.0, max_iter=20000, tol=1e-10).fit(sc.transform(X[~m[0]]), y[~m[0]])
        zr = ref.decision_function(sc.transform(X[m[0]]))
        if zr.ndim == 1:
            zr = np.stack([-zr / 2, zr / 2], 1)
        agree = (zr.argmax(1) == Zt[m[0]].argmax(1)).mean()
        assert agree > 0.995, (C, agree)
        assert np.corrcoef(zr.ravel(), Zt[m[0]].ravel())[0, 1] > 0.999


def test_fusion_identifiability_cases():
    """The corrected claim: no global scaling symmetry; non-identified when C < K
    or at the boundary where a useless teacher's tau -> inf."""
    rng = np.random.default_rng(1)
    m = fusion.ClassConditionalFusion(lam=0.0)
    n, K, C = 200, 3, 5
    Zs = [rng.normal(size=(n, C)) for _ in range(K)]
    theta = rng.normal(size=C * K + K + C) * 0.3
    base = m._fused(theta, Zs, K, C)
    t2 = theta.copy()
    t2[C * K:C * K + K] += np.log(2.0)            # rescale tau alone
    assert np.abs(m._fused(t2, Zs, K, C) - base).max() > 1e-3, "tau rescaling must change the fit"
    # C < K: a direction in (alpha, tau) space leaves v = alpha/tau unchanged
    C2 = 2
    W = rng.normal(size=(C2, K)); tau = np.exp(rng.normal(size=K))
    alpha = np.exp(W) / np.exp(W).sum(1, keepdims=True)
    v = alpha / tau
    # find tau' != tau with sum_k v_ck tau'_k = 1 for both classes (2 equations, 3 unknowns)
    A = v                                        # (C2, K)
    null = np.linalg.svd(A)[2][-1]               # null-space direction of A
    tau_p = tau + 0.1 * null * np.abs(tau).min() / np.abs(null).max()
    assert np.all(tau_p > 0)
    alpha_p = v * tau_p
    assert np.allclose(alpha_p.sum(1), 1.0, atol=1e-9)
    assert np.allclose(alpha_p / tau_p, v) and not np.allclose(tau_p, tau)


def test_identified_fusion_param_count():
    rng = np.random.default_rng(2)
    n, K, C = 300, 3, 7
    y = rng.integers(0, C, n)
    Zs = [np.eye(C)[y] * 2 + rng.normal(size=(n, C)) for _ in range(K)]
    f = fusion.IdentifiedClassConditionalFusion().fit(Zs, y)
    assert f.n_params == C * K + C == 28


def test_mixed_effects_close_to_cluster_robust():
    rng = np.random.default_rng(3)
    cl = np.repeat(np.arange(15), 6)
    y = 0.2 + np.repeat(rng.normal(0, 0.3, 15), 6) + rng.normal(0, 1, 90)
    me = I.mixed_effects_random_intercept(y, cl)
    cr = I.cluster_robust_ols(y, cl)
    assert abs(me["estimate"] - cr["estimate"][0]) < 0.05
    assert 0.5 < me["se"] / cr["se"][0] < 2.0, (me["se"], cr["se"][0])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall core checks passed")
