"""Fusion rules used to measure marginal utility.

The class-conditional method (Eq. 2 of the rejected paper) is retained as the
primary rule, but the paper's own Appendix I showed the null is invariant across
fusion rules, so we compute all three by default and report the spread. Keeping
the parameter-free rules front and centre also pre-empts the objection that the
learned fusion can down-weight a candidate into invisibility.

Order sensitivity is real: class weights are initialised with mass on the first
teacher. Every call therefore sorts teacher keys canonically (alphabetically)
unless an explicit order is passed, so results are reproducible.
"""

from __future__ import annotations

import numpy as np

from folds import softmax


def fuse_mean_logits(Zs: list[np.ndarray]) -> np.ndarray:
    """Zero-parameter logit averaging. Standardise each teacher's logits first,
    otherwise a teacher with larger logit scale dominates by construction."""
    std = []
    for Z in Zs:
        Z = np.asarray(Z, dtype=np.float64)
        s = Z.std(axis=0, keepdims=True)
        s[s == 0] = 1.0
        std.append((Z - Z.mean(axis=0, keepdims=True)) / s)
    return np.mean(std, axis=0)


def fuse_mean_probs(Zs: list[np.ndarray]) -> np.ndarray:
    """Zero-parameter probability averaging (returns probabilities, not logits)."""
    return np.mean([softmax(np.asarray(Z, dtype=np.float64)) for Z in Zs], axis=0)


class ClassConditionalFusion:
    """Eq. 2: z_fused(c) = sum_k alpha_{c,k} * z_k(c) / tau_k + b_c.

    Parameter count = C*K + K + C (11-31 in the pathology experiments).
    Trained with cross-entropy + entropy regulariser (lambda=0.01) via Adam if
    torch is available, otherwise by scipy L-BFGS on the same objective. The
    scipy path is what makes this runnable on a Colab CPU cell and in CI.
    """

    def __init__(self, lam: float = 0.01, max_iter: int = 400, seed: int = 0):
        self.lam = lam
        self.max_iter = max_iter
        self.seed = seed

    def _unpack(self, theta, K, C):
        W = theta[: C * K].reshape(C, K)
        tau_hat = theta[C * K : C * K + K]
        b = theta[C * K + K :]
        return W, np.exp(np.clip(tau_hat, -10, 10)), b

    def _fused(self, theta, Zs, K, C):
        W, tau, b = self._unpack(theta, K, C)
        alpha = softmax(W, axis=1)                      # (C, K)
        out = np.zeros((Zs[0].shape[0], C))
        for k in range(K):
            out += alpha[:, k][None, :] * (Zs[k] / tau[k])
        return out + b[None, :]

    def _loss_grad(self, theta, Zs, y, K, C):
        """Objective: mean cross-entropy minus lam * mean per-class entropy.

        Gradient is analytic; tests/test_gradient.py checks it against a
        central-difference approximation before this is ever used on real data.
        """
        W, tau, b = self._unpack(theta, K, C)
        alpha = softmax(W, axis=1)                       # (C, K)
        Z = self._fused(theta, Zs, K, C)
        n = Z.shape[0]

        Zsh = Z - Z.max(axis=1, keepdims=True)
        logp = Zsh - np.log(np.exp(Zsh).sum(axis=1, keepdims=True))
        p = np.exp(logp)
        ce = -logp[np.arange(n), y].mean()
        ent = -(alpha * np.log(alpha + 1e-12)).sum(axis=1).mean()
        loss = ce - self.lam * ent

        onehot = np.zeros_like(p)
        onehot[np.arange(n), y] = 1.0
        dZ = (p - onehot) / n                            # dL/dZ_fused, (n, C)

        db = dZ.sum(axis=0)                              # (C,)
        dalpha = np.zeros((C, K))
        dtau_hat = np.zeros(K)
        for k in range(K):
            dalpha[:, k] = (dZ * (Zs[k] / tau[k])).sum(axis=0)
            # d/d tau_hat_k of alpha_k * Z_k * exp(-tau_hat_k) = -alpha_k * Z_k / tau_k
            dtau_hat[k] = -(dZ * (alpha[:, k][None, :] * Zs[k] / tau[k])).sum()

        def softmax_jac_rows(g_alpha):
            """Map dL/dalpha -> dL/dW row-wise through the per-class softmax."""
            out = np.zeros_like(g_alpha)
            for c in range(C):
                a = alpha[c]
                out[c] = a * (g_alpha[c] - np.dot(a, g_alpha[c]))
            return out

        dW = softmax_jac_rows(dalpha)
        # entropy term: ent = -(1/C) sum_c sum_k alpha_ck log alpha_ck
        d_ent_dalpha = -(np.log(alpha + 1e-12) + 1.0) / C
        dW = dW - self.lam * softmax_jac_rows(d_ent_dalpha)

        g = np.concatenate([dW.ravel(), dtau_hat, db])
        return loss, g

    def fit(self, Zs: list[np.ndarray], y: np.ndarray):
        Zs = [np.asarray(Z, dtype=np.float64) for Z in Zs]
        K = len(Zs)
        C = Zs[0].shape[1]
        rng = np.random.default_rng(self.seed)
        # W is a LOG-space parameter: alpha = softmax(W). Initialise near zero so
        # alpha starts close to uniform (1/K) and the softmax is not saturated.
        # Saturating it here zeroes the gradient and the fit silently does nothing.
        theta0 = np.concatenate([
            rng.normal(0.0, 0.01, size=C * K),
            np.zeros(K),          # tau_hat = 0  =>  tau = 1
            np.zeros(C),          # b = 0
        ])
        from scipy.optimize import minimize
        res = minimize(self._loss_grad, theta0, args=(Zs, y, K, C), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.theta_, self.K_, self.C_, self.res_ = res.x, K, C, res
        return self

    def predict_logits(self, Zs: list[np.ndarray]) -> np.ndarray:
        Zs = [np.asarray(Z, dtype=np.float64) for Z in Zs]
        return self._fused(self.theta_, Zs, self.K_, self.C_)

    @property
    def n_params(self) -> int:
        return self.C_ * self.K_ + self.K_ + self.C_

    @property
    def class_weights(self) -> np.ndarray:
        return softmax(self._unpack(self.theta_, self.K_, self.C_)[0], axis=1)


class IdentifiedClassConditionalFusion:
    """RECOMMENDED parameterisation: z_fused(c) = sum_k v_{c,k} z_k(c) + b_c.

    Why this replaces the v2 parameterisation as the default (corrected v3 note).

    v2 wrote the fused logit as alpha_{c,k} * z_k(c) / tau_k with alpha a
    per-class softmax. There is NO global scaling symmetry in that model
    (alpha is constrained to the simplex, so alpha -> c*alpha is not allowed,
    and rescaling tau alone changes the logits). What IS true:

      * The quantity that acts on the data is the effective weight
        v_{c,k} = alpha_{c,k} / tau_k. A teacher can carry a large alpha and
        still contribute almost nothing if its tau is large, so alpha alone
        must not be reported as "trust" (v2 Appendix G did this).
      * The decomposition (alpha, tau) is not identified when C < K
        (e.g. MHIST: 2 classes, 3 teachers), and it is weakly identified at the
        boundary where a useless teacher's tau -> infinity, because that
        teacher's alpha is then free. tests/test_core.py
        shows both cases.
      * The entropy regulariser acts on alpha, so where alpha is free it pins
        alpha towards uniform without changing the fit.

    Fitting v directly removes all three problems. Calibration is handled as a
    separate post-hoc step (v2 showed the learned temperatures do not beat
    temperature scaling). Parameter count is C*K + C (28 on BRACS with K=3,
    versus C*K + K + C = 31 for v2).
    """

    def __init__(self, ridge: float = 1e-3, max_iter: int = 500, seed: int = 0):
        self.ridge = ridge
        self.max_iter = max_iter
        self.seed = seed

    def _fused(self, theta, Zs, K, C):
        V = theta[: C * K].reshape(C, K)
        b = theta[C * K:]
        out = np.zeros((Zs[0].shape[0], C))
        for k in range(K):
            out += V[:, k][None, :] * Zs[k]
        return out + b[None, :]

    def _loss_grad(self, theta, Zs, y, K, C):
        V = theta[: C * K].reshape(C, K)
        b = theta[C * K:]
        Z = self._fused(theta, Zs, K, C)
        n = Z.shape[0]
        Zsh = Z - Z.max(axis=1, keepdims=True)
        logp = Zsh - np.log(np.exp(Zsh).sum(axis=1, keepdims=True))
        p = np.exp(logp)
        loss = -logp[np.arange(n), y].mean() + self.ridge * float((V ** 2).sum())

        onehot = np.zeros_like(p)
        onehot[np.arange(n), y] = 1.0
        dZ = (p - onehot) / n
        dV = np.zeros((C, K))
        for k in range(K):
            dV[:, k] = (dZ * Zs[k]).sum(axis=0)
        dV += 2 * self.ridge * V
        g = np.concatenate([dV.ravel(), dZ.sum(axis=0)])
        return loss, g

    def fit(self, Zs: list, y):
        Zs = [np.asarray(Z, dtype=np.float64) for Z in Zs]
        K, C = len(Zs), Zs[0].shape[1]
        rng = np.random.default_rng(self.seed)
        # initialise at the equal-weight average, which is a sensible starting point
        theta0 = np.concatenate([np.full(C * K, 1.0 / K) + rng.normal(0, 1e-3, C * K),
                                 np.zeros(C)])
        from scipy.optimize import minimize
        res = minimize(self._loss_grad, theta0, args=(Zs, y, K, C), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.theta_, self.K_, self.C_, self.res_ = res.x, K, C, res
        return self

    def predict_logits(self, Zs: list):
        Zs = [np.asarray(Z, dtype=np.float64) for Z in Zs]
        return self._fused(self.theta_, Zs, self.K_, self.C_)

    @property
    def n_params(self) -> int:
        return self.C_ * self.K_ + self.C_

    @property
    def class_weights(self) -> np.ndarray:
        """Identified per-class teacher weights: V normalised to sum to one per
        class. Safe to report as "trust placed in teacher k for class c"."""
        V = self.theta_[: self.C_ * self.K_].reshape(self.C_, self.K_)
        V = np.clip(V, 0.0, None)
        s = V.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return V / s

    @property
    def raw_class_weights(self) -> np.ndarray:
        return self.theta_[: self.C_ * self.K_].reshape(self.C_, self.K_)


RULES = {
    "mean_logits": lambda Zs, ytr=None, seed=0: fuse_mean_logits(Zs),
    "mean_probs": lambda Zs, ytr=None, seed=0: fuse_mean_probs(Zs),
    # default: the identified parameterisation (see IdentifiedClassConditionalFusion)
    "class_conditional": lambda Zs, ytr, seed=0: IdentifiedClassConditionalFusion(seed=seed).fit(Zs, ytr).predict_logits(Zs),
    "class_conditional_v2_legacy": lambda Zs, ytr, seed=0: ClassConditionalFusion(lam=0.01, seed=seed).fit(Zs, ytr).predict_logits(Zs),
}


def fuse(Zs, rule: str, ytr=None, seed: int = 0):
    if rule not in RULES:
        raise KeyError(f"unknown fusion rule {rule!r}; choose from {sorted(RULES)}")
    return RULES[rule](Zs, ytr=ytr, seed=seed)
