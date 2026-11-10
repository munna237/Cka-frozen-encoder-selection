"""Inference utilities sized for the actual design.

The rejected version's primary interval, +0.25 [ -1.00, +1.50 ], was wide purely
because G = 3 clusters. Every function here takes a cluster label so the same
code gives you a tight interval at G = 16 without any change in analysis plan.
Implementations are deliberately hand-rolled (numpy/scipy only) so they run in
CI and on a Colab CPU cell.

CR1 small-sample correction and a t(G-1) reference are used throughout, matching
what the paper already adopted and what reviewers accepted as conservative.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _design(x: np.ndarray | None, n: int) -> np.ndarray:
    if x is None:
        return np.ones((n, 1))
    x = np.asarray(x, dtype=np.float64).reshape(n, -1)
    return np.hstack([np.ones((n, 1)), x])


def cluster_robust_ols(y: np.ndarray, clusters: np.ndarray,
                       x: np.ndarray | None = None,
                       weights: np.ndarray | None = None) -> dict:
    """OLS with a cluster-robust (CR1) covariance and a t(G-1) reference.

    y       : outcome per observation (e.g. per-fold utility difference)
    clusters: cluster id per observation (e.g. dataset or pool)
    x       : optional regressor(s); None => intercept-only (the primary estimand)
    weights : optional per-observation weights (e.g. inverse-variance)

    Returns dict with estimate(s), SEs, t, df, p, and 95% CIs.
    """
    y = np.asarray(y, dtype=np.float64)
    clusters = np.asarray(clusters)
    n = y.shape[0]
    X = _design(x, n)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)

    Xw = X * w[:, None]
    XtX_inv = np.linalg.pinv(Xw.T @ X)
    beta = XtX_inv @ (Xw.T @ y)
    resid = y - X @ beta

    G = len(np.unique(clusters))
    if G < 2:
        raise ValueError("need at least 2 clusters for a cluster-robust SE")
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(clusters):
        m = clusters == g
        s = (Xw[m].T @ resid[m])
        meat += np.outer(s, s)
    scale = G / (G - 1) * (n - 1) / max(n - X.shape[1], 1)   # CR1
    V = scale * XtX_inv @ meat @ XtX_inv

    se = np.sqrt(np.diag(V))
    df = G - 1
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    tcrit = stats.t.ppf(0.975, df)
    return {
        "estimate": beta, "se": se, "t": t, "df": df, "p": p,
        "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se,
        "n": n, "G": G, "crit": tcrit,
    }


def tost_equivalence(y: np.ndarray, clusters: np.ndarray, margin: float,
                     x: np.ndarray | None = None, param_index: int = 1) -> dict:
    """Two-one-sided-tests equivalence against +/- margin, cluster-robust.

    With G = 3 this failed (p = 0.061) in the rejected version. At G >= 12 the
    same point estimate should clear it, which upgrades the claim from
    "we failed to reject a difference" to "we establish equivalence within
    +/- margin" -- the difference between a weak null and a real result.
    """
    r = cluster_robust_ols(y, clusters, x=x)
    b = r["estimate"][param_index] if r["estimate"].size > 1 else r["estimate"][0]
    se = r["se"][param_index] if r["se"].size > 1 else r["se"][0]
    df = r["df"]
    t_lo = (b - (-margin)) / se
    t_hi = (b - (+margin)) / se
    p_lo = stats.t.sf(t_lo, df)     # H0: b <= -margin
    p_hi = stats.t.cdf(t_hi, df)    # H0: b >= +margin
    tcrit = stats.t.ppf(0.95, df)   # one-sided 5%
    return {
        "estimate": b, "se": se, "df": df, "margin": margin,
        "p_lower": float(p_lo), "p_upper": float(p_hi),
        "p_tost": float(max(p_lo, p_hi)),
        "equivalent": bool(max(p_lo, p_hi) < 0.05),
        "ci90_low": b - tcrit * se, "ci90_high": b + tcrit * se,
        "G": r["G"], "n": r["n"],
    }


def mixed_effects_random_intercept(y: np.ndarray, clusters: np.ndarray,
                                   x: np.ndarray | None = None) -> dict:
    """Intercept-only random-intercept model, reported as a robustness check.

    Uses statsmodels MixedLM (REML) when available, with a t(G-1) reference to
    match the primary analysis. Falls back to a method-of-moments estimate
    whose SE is Var(grand mean) = (sigma2_between + sigma2_within / n_bar) / G.
    (v3 fix: the previous fallback divided the within variance by n*G, which
    understated the SE.)
    """
    y = np.asarray(y, dtype=np.float64)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    G, n = len(uniq), y.shape[0]
    df = G - 1
    tcrit = stats.t.ppf(0.975, df) if df > 0 else np.nan
    try:
        import warnings
        import statsmodels.api as sm
        codes = np.searchsorted(uniq, clusters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sm.MixedLM(y, np.ones((n, 1)), groups=codes).fit(reml=True)
        est, se = float(r.params[0]), float(r.bse[0])
        s2u = float(np.asarray(r.cov_re).ravel()[0])
        s2w = float(r.scale)
        method = "statsmodels MixedLM (REML)"
    except Exception as e:                                   # fallback
        means = np.array([y[clusters == g].mean() for g in uniq])
        sizes = np.array([np.sum(clusters == g) for g in uniq], dtype=float)
        est = float(np.average(means, weights=sizes))
        ss_between = float(np.sum(sizes * (means - est) ** 2))
        ss_within = float(sum(((y[clusters == g] - y[clusters == g].mean()) ** 2).sum()
                              for g in uniq))
        s2w = ss_within / max(n - G, 1)
        ms_between = ss_between / max(G - 1, 1)
        n0 = (n - np.sum(sizes ** 2) / n) / max(G - 1, 1)
        s2u = max((ms_between - s2w) / max(n0, 1e-12), 0.0)
        se = float(np.sqrt((s2u + s2w / sizes.mean()) / G))
        method = f"method of moments (statsmodels unavailable: {type(e).__name__})"
    t = est / se if se > 0 else np.nan
    return {
        "estimate": est, "se": se, "t": t, "df": df,
        "p": float(2 * stats.t.sf(abs(t), df)) if se > 0 and df > 0 else np.nan,
        "ci_low": est - tcrit * se, "ci_high": est + tcrit * se,
        "sigma2_between": s2u, "sigma2_within": s2w,
        "icc": float(s2u / (s2u + s2w)) if (s2u + s2w) > 0 else np.nan,
        "G": G, "n": n, "method": method,
    }


def mannwhitney_u(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Mann-Whitney U, used for matched-band comparisons."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return {"U": np.nan, "p": np.nan, "n_a": len(a), "n_b": len(b)}
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(U), "p": float(p), "n_a": len(a), "n_b": len(b)}


def spearman(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return {"rho": np.nan, "p": np.nan, "n": len(a)}
    rho, p = stats.spearmanr(a, b)
    return {"rho": float(rho), "p": float(p), "n": len(a)}


def paired_fold_test(d: np.ndarray) -> dict:
    """Naive fold-independent paired tests. Reported but NOT primary -- folds
    within a dataset share training data, which is why the cluster-robust figure
    is the one to quote. Keep it in the table for comparability with v2."""
    d = np.asarray(d, float)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1) if n > 1 else np.nan
    t = m / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan
    p_t = float(2 * stats.t.sf(abs(t), n - 1)) if np.isfinite(t) else np.nan
    try:
        _, p_w = stats.wilcoxon(d)
        p_w = float(p_w)
    except ValueError:
        p_w = np.nan
    return {"mean": float(m), "sd": float(sd), "n": n, "t": float(t),
            "p_t": p_t, "p_wilcoxon": p_w,
            "wins": int(np.sum(d > 0)), "losses": int(np.sum(d < 0)),
            "ties": int(np.sum(d == 0))}
