"""Per-fold logit protocol.

The rejected version got this right and it should not change: a probe fit on all
data and scored on the same data inflates apparent teacher accuracy by up to 10
points, and that inflation propagates into every oracle bound, unique-contribution
count and fusion gain. Everything downstream depends on these being held-out.

New in this version: optional *group* awareness. If the dataset has a group
variable (patient, slide, institution), folds MUST be grouped by it. The
SICAPv2 patient-level result in Appendix K showed that patch-level folds leak:
patches from one biopsy appear in both train and validation, which inflates
in-distribution numbers and hides the shift failure. Group from the start.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def make_folds(y: np.ndarray, groups: np.ndarray | None = None, k: int = 5,
               seed: int = 0) -> list[np.ndarray]:
    """Return a list of k boolean holdout masks over the inputs."""
    y = np.asarray(y)
    n = y.shape[0]
    masks: list[np.ndarray] = []
    if groups is not None:
        groups = np.asarray(groups)
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(n), y, groups)
    else:
        # stratify when feasible, else plain KFold (e.g. very few samples/class)
        _, counts = np.unique(y, return_counts=True)
        if counts.min() >= k:
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        else:
            splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros(n), y)
    for _, va in iterator:
        m = np.zeros(n, dtype=bool)
        m[va] = True
        masks.append(m)
    # every input is held out exactly once
    assert np.sum(np.stack(masks).astype(int), axis=0).min() == 1
    if groups is not None:
        for m in masks:
            assert len(np.intersect1d(groups[m], groups[~m])) == 0, "group leakage"
    return masks


def _torch_probe(Xtr, ytr, Xva, n_classes, C=1.0, max_iter=500, device=None):
    """Same objective as sklearn's LogisticRegression(C, lbfgs) solved with
    torch L-BFGS, so large pools (20k x 2560 x 211 classes) fit in seconds on a
    GPU instead of minutes on CPU.

    sklearn minimises  C * sum_i CE_i + 0.5 * ||W||^2  (bias unpenalised).
    Dividing by C*n gives  mean CE + ||W||^2 / (2 C n), which is what we use.
    Binary problems use a single weight vector, exactly as sklearn does, because
    a two-column multinomial with the same L2 penalty is a different estimator.
    tests/test_core.py checks agreement with sklearn.
    """
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32
    X = torch.as_tensor(Xtr, dtype=dt, device=dev)
    V = torch.as_tensor(Xva, dtype=dt, device=dev)
    y = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    n, d = X.shape
    lam = 1.0 / (2.0 * C * n)
    binary = n_classes == 2
    out_dim = 1 if binary else n_classes
    W = torch.zeros(d, out_dim, dtype=dt, device=dev, requires_grad=True)
    b = torch.zeros(out_dim, dtype=dt, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-6, tolerance_change=1e-9,
                            line_search_fn="strong_wolfe")
    yf = y.to(dt)

    def closure():
        opt.zero_grad()
        z = X @ W + b
        if binary:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z[:, 0], yf)
        else:
            loss = torch.nn.functional.cross_entropy(z, y)
        loss = loss + lam * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        z = (V @ W + b).double().cpu().numpy()
    if binary:
        z = np.concatenate([-z / 2.0, z / 2.0], axis=1)
    return z


def per_fold_logits(X: np.ndarray, y: np.ndarray, masks: list[np.ndarray],
                    C: float = 1.0, max_iter: int = 5000,
                    backend: str = "sklearn") -> np.ndarray:
    """Held-out logits for every input under the per-fold protocol.

    For each fold: fit StandardScaler on the training partition only, fit a
    logistic probe on the training partition only, predict logits for the
    held-out partition only. Concatenate into one (n, C) matrix aligned to the
    original input order.

    backend: "sklearn" (reference) or "torch" (same objective, L-BFGS on GPU).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, _ = X.shape
    classes = np.unique(y)
    Z = np.full((n, len(classes)), np.nan)
    cls_to_col = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_to_col[c] for c in y])
    for m in masks:
        sc = StandardScaler().fit(X[~m])
        Xtr, Xva = sc.transform(X[~m]), sc.transform(X[m])
        if backend == "torch":
            present = np.unique(y_idx[~m])
            if len(present) != len(classes):
                raise RuntimeError("a class is missing from a training fold; "
                                   "use more data or fewer folds")
            Z[m] = _torch_probe(Xtr, y_idx[~m], Xva, len(classes), C=C,
                                max_iter=min(max_iter, 1000))
            continue
        probe = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        probe.fit(Xtr, y[~m])
        lg = probe.decision_function(Xva)
        if lg.ndim == 1:                       # binary: decision_function gives one column
            lg = np.stack([-lg / 2.0, lg / 2.0], axis=1)
        cols = [cls_to_col[c] for c in probe.classes_]
        Z[np.ix_(np.where(m)[0], cols)] = lg
    if np.isnan(Z).any():
        raise RuntimeError("some inputs never received held-out logits")
    return Z


def per_fold_probs(X, y, masks, **kw) -> np.ndarray:
    """Held-out class probabilities (for probability-averaging baselines)."""
    Z = per_fold_logits(X, y, masks, **kw)
    return Z  # caller uses softmax(Z); kept separate for clarity in the pipeline


def softmax(Z: np.ndarray, axis: int = -1) -> np.ndarray:
    Z = Z - Z.max(axis=axis, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=axis, keepdims=True)


def preds_from_logits(Z: np.ndarray) -> np.ndarray:
    return np.asarray(Z).argmax(axis=1)


def accuracy(Z: np.ndarray, y: np.ndarray) -> float:
    return float((preds_from_logits(Z) == np.asarray(y)).mean())
