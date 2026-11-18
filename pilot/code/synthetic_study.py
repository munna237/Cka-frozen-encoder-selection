"""
Core library for the controlled synthetic study (Reviewer 7oVd, W1.3).

Design goal: vary representational similarity (CKA) between two teachers and
their individual competence INDEPENDENTLY, which is impossible in the real
four-encoder pool (W1.1).

Generative model
----------------
Each teacher's feature matrix is built from two blocks of a fixed total width d:

    F_k = [  S  |  P_k  ]           (then a random orthogonal rotation)

  * S    (d_s dims) is the SHARED block. Its realisation is byte-identical
         across teachers, so it contributes maximally to CKA.
  * P_k  (d_p = d - d_s dims) is the PRIVATE block, drawn independently per
         teacher, so it contributes ~0 to CKA.

Both blocks carry class information. Prototype norms are scaled by
1/sqrt(dim) so that the *total* class-signal energy of a block does not
change as its width changes. This is what decouples the two axes:

  * d_s / d          controls CKA                (representational overlap)
  * s_k (private SNR) controls teacher k accuracy (individual competence)

Complementarity is NOT built in by construction. Errors driven by the shared
block are perfectly correlated across teachers; errors driven by the private
blocks are independent. So diversity genuinely *can* buy fusion utility here --
the experiment is free to conclude either way.

A random orthogonal rotation is applied to each teacher's features. Note that
linear CKA is invariant to orthogonal transformation, so rotation alone cannot
be used to control CKA (relevant to the reviewer's suggested implementation).
It is applied only so that no teacher's informative directions are
axis-aligned.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------
# Representational similarity
# ----------------------------------------------------------------------------
def linear_cka(X, Y):
    """Linear CKA on column-centred features (Eq. 1 of the manuscript)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 0 else 0.0


# ----------------------------------------------------------------------------
# Synthetic teacher pool
# ----------------------------------------------------------------------------
def make_pool(n=3000, d=64, n_classes=4, d_shared=32,
              s_shared=1.0, s_private=(1.0, 1.0), noise=1.0,
              rotate=True, seed=0):
    """Generate labels and one feature matrix per teacher.

    Parameters
    ----------
    d_shared   : width of the shared block -> controls CKA
    s_shared   : class-signal strength of the shared block (common to all)
    s_private  : per-teacher class-signal strength of the private block
                 -> controls individual competence
    """
    rng = np.random.default_rng(seed)
    K = len(s_private)
    d_priv = d - d_shared
    y = rng.integers(0, n_classes, size=n)

    # --- shared block: identical realisation for every teacher --------------
    if d_shared > 0:
        M_s = rng.normal(size=(n_classes, d_shared)) / np.sqrt(d_shared)
        S = s_shared * M_s[y] + noise * rng.normal(size=(n, d_shared))
    else:
        S = np.zeros((n, 0))

    feats = []
    for k in range(K):
        if d_priv > 0:
            M_p = rng.normal(size=(n_classes, d_priv)) / np.sqrt(d_priv)
            P = s_private[k] * M_p[y] + noise * rng.normal(size=(n, d_priv))
        else:
            P = np.zeros((n, 0))
        F = np.hstack([S, P])
        if rotate and F.shape[1] > 0:
            Q, _ = np.linalg.qr(rng.normal(size=(F.shape[1], F.shape[1])))
            F = F @ Q
        feats.append(F)
    return y, feats


# ----------------------------------------------------------------------------
# Per-fold logit protocol (Section 3.2 / Appendix B.3 of the manuscript)
# ----------------------------------------------------------------------------
def perfold_logits(F, y, n_splits=5, seed=0):
    """Out-of-fold logits. No input contributes to the probe producing its own
    logit; scaler statistics come from the training partition only."""
    Z = np.zeros((len(y), len(np.unique(y))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(F, y):
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
        clf.fit(sc.transform(F[tr]), y[tr])
        Z[te] = clf.decision_function(sc.transform(F[te]))
    return Z


# ----------------------------------------------------------------------------
# Class-conditional logit fusion (Eq. 3), numpy + Adam
# ----------------------------------------------------------------------------
def _forward(Z, W, tau_hat, b):
    """Z: (N, K, C) -> fused logits (N, C)."""
    alpha = _softmax(W, axis=1)              # (C, K)
    tau = np.exp(tau_hat)                    # (K,)
    scaled = Z / tau[None, :, None]
    fused = np.einsum("ck,nkc->nc", alpha, scaled) + b[None, :]
    return fused, alpha, tau, scaled


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def fit_fusion(Ztr, ytr, Zva, yva, n_classes, lam=0.01, lr=0.05,
               steps=800, patience=100, eps=1e-8, seed=0):
    """Train the class-conditional fusion. Early stopping is monitored on an
    explicit validation split that is disjoint from the evaluation fold."""
    rng = np.random.default_rng(seed)
    K = Ztr.shape[1]
    C = n_classes
    W = rng.normal(scale=0.01, size=(C, K))
    tau_hat = np.zeros(K)
    b = np.zeros(C)
    params = [W, tau_hat, b]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2 = 0.9, 0.999
    best = (np.inf, None, 0)
    Y = np.eye(C)[ytr]
    N = len(ytr)

    for t in range(1, steps + 1):
        fused, alpha, tau, scaled = _forward(Ztr, W, tau_hat, b)
        P = _softmax(fused, axis=1)
        G = (P - Y) / N                                   # dL/dfused (N, C)

        dA = np.einsum("nc,nkc->ck", G, scaled)           # dL/dalpha (C, K)
        # entropy regulariser: loss includes -lam * H, H = -(1/C) sum a log a
        dA += (lam / C) * (np.log(alpha + eps) + 1.0)
        # backprop through the per-class softmax over teachers
        dW = alpha * (dA - (dA * alpha).sum(axis=1, keepdims=True))
        dtau_hat = -np.einsum("nc,ck,nkc->k", G, alpha, scaled)
        db = G.sum(axis=0)

        grads = [dW, dtau_hat, db]
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = b1 * m[i] + (1 - b1) * g
            v[i] = b2 * v[i] + (1 - b2) * g * g
            mh = m[i] / (1 - b1 ** t)
            vh = v[i] / (1 - b2 ** t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        W, tau_hat, b = params

        if t % 10 == 0:
            fv, _, _, _ = _forward(Zva, W, tau_hat, b)
            Pv = _softmax(fv, axis=1)
            l = -np.log(Pv[np.arange(len(yva)), yva] + 1e-12).mean()
            if l < best[0] - 1e-5:
                best = (l, (W.copy(), tau_hat.copy(), b.copy()), t)
            elif t - best[2] > patience:
                break
    return best[1] if best[1] is not None else (W, tau_hat, b)


def Z_over_tau(Z, tau):
    return Z / tau[None, :, None]


def predict_fusion(Z, params):
    W, tau_hat, b = params
    fused, _, _, _ = _forward(Z, W, tau_hat, b)
    return fused


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
def ece(probs, y, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def fusion_gain(Zs, y, n_classes, seed=0, n_splits=5):
    """Nested protocol: outer folds define the evaluation partition; the
    fusion parameters are fit on the outer-training logits with an inner
    split for early stopping. Returns fused accuracy, best-single accuracy,
    fused ECE and best-single ECE."""
    Z = np.stack(Zs, axis=1)                      # (N, K, C)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fused_pred = np.zeros(len(y), dtype=int)
    fused_prob = np.zeros((len(y), n_classes))
    for tr, te in skf.split(Z[:, 0, :], y):
        cut = int(0.8 * len(tr))
        itr, iva = tr[:cut], tr[cut:]
        params = fit_fusion(Z[itr], y[itr], Z[iva], y[iva], n_classes, seed=seed)
        f = predict_fusion(Z[te], params)
        fused_pred[te] = f.argmax(1)
        fused_prob[te] = _softmax(f, axis=1)
    fused_acc = float((fused_pred == y).mean())
    singles = [float((z.argmax(1) == y).mean()) for z in Zs]
    best_i = int(np.argmax(singles))
    best_prob = _softmax(Zs[best_i], axis=1)
    return dict(
        fused_acc=fused_acc,
        single_accs=singles,
        best_single=singles[best_i],
        gain=fused_acc - singles[best_i],
        fused_ece=ece(fused_prob, y),
        best_single_ece=ece(best_prob, y),
    )
