"""Core functions: per-fold logit generation, class-conditional fusion, CKA, ECE."""
import numpy as np, json, os, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform

TEACHERS = ['uni','phikon_v2','ctranspath','resnet50_imagenet']
DS_CFG = {
    'bracs':  {'C':7,'pfx':'trainval'},
    'sicapv2':{'C':4,'pfx':'all'},
    'mhist':  {'C':2,'pfx':'all'},
}

def load_splits(split_dir, ds):
    for fn in ['5fold_cv_splits.json','5fold.json']:
        p = os.path.join(split_dir, fn)
        if os.path.exists(p):
            with open(p) as f: data = json.load(f)
            sp = []
            for i in range(5):
                for k in [f'fold_{i}', f'fold{i}', str(i)]:
                    if k in data:
                        fd = data[k]
                        tr = np.array(fd.get('train_indices', fd.get('train', [])))
                        va = np.array(fd.get('val_indices', fd.get('val', [])))
                        sp.append((tr, va)); break
            if len(sp) == 5: return sp
    raise FileNotFoundError(f"No splits for {ds}")

def canonical_ece(logits, labels, n_bins=15):
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    confs, preds = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for b in range(n_bins):
        m = (confs > edges[b]) & (confs <= edges[b + 1])
        if m.sum() == 0: continue
        ece += m.sum()/len(labels) * abs((preds[m]==labels[m]).mean() - confs[m].mean())
    return ece

def linear_cka(F1, F2, sub=2000):
    """Linear CKA.

    NOTE two behaviours that matter if you reuse this outside the harness:
    (1) inputs with more than `sub` rows are subsampled with a fixed seed, so
        this returns the subsample estimate, not the full-sample value the
        paper reports. The released feature subsets are already 2,000 rows, so
        the branch never fires here; compute_cka.py computes full-sample values
        when the full arrays are available.
    (2) the centring below is in place and mutates the caller's arrays. Pass a
        copy if you need the originals afterwards.
    """
    n = F1.shape[0]
    if n > sub:
        idx = np.random.RandomState(42).choice(n, sub, replace=False)
        F1, F2 = F1[idx], F2[idx]
    F1 -= F1.mean(0); F2 -= F2.mean(0)
    num = np.linalg.norm(F2.T @ F1, 'fro') ** 2
    den = np.linalg.norm(F1.T @ F1, 'fro') * np.linalg.norm(F2.T @ F2, 'fro')
    return num / den if den > 0 else 0

def rbf_cka(F1, F2, sub=1500):
    n = F1.shape[0]
    if n > sub:
        idx = np.random.RandomState(42).choice(n, sub, replace=False)
        F1, F2 = F1[idx], F2[idx]
    F1 -= F1.mean(0); F2 -= F2.mean(0); n = F1.shape[0]
    def kern(X):
        d = pdist(X, 'sqeuclidean'); s = np.median(d)
        if s == 0: s = 1.0
        K = squareform(np.exp(-d / (2*s))); np.fill_diagonal(K, 1.0); return K
    def ctr(K):
        nn = K.shape[0]; H = np.eye(nn) - np.ones((nn,nn))/nn; return H @ K @ H
    K1c = ctr(kern(F1)); K2c = ctr(kern(F2))
    h12 = np.sum(K1c*K2c)/((n-1)**2)
    h11 = np.sum(K1c*K1c)/((n-1)**2)
    h22 = np.sum(K2c*K2c)/((n-1)**2)
    d = np.sqrt(h11*h22)
    return h12/d if d > 0 else 0

class ClassCondFusion(nn.Module):
    def __init__(self, nc, nt):
        super().__init__()
        w = torch.zeros(nc, nt); w[:, 0] = 0.5
        self.class_weights = nn.Parameter(w)
        self.log_temps = nn.Parameter(torch.zeros(nt))
        self.bias = nn.Parameter(torch.zeros(nc))
    def forward(self, tl):
        temps = self.log_temps.exp().clamp(0.1, 10.0)
        scaled = tl / temps[None, :, None]
        alpha = F.softmax(self.class_weights, dim=1)
        return (scaled.permute(0,2,1) * alpha[None]).sum(2) + self.bias[None]

def run_fusion_cv(stacked, labels, nc, nt, splits, seed=42, device='cpu'):
    fa = []; al = np.zeros((len(labels), nc))
    for tr, va in splits:
        torch.manual_seed(seed); np.random.seed(seed)
        Xtr = torch.FloatTensor(stacked[tr]).to(device)
        ytr = torch.LongTensor(labels[tr]).to(device)
        Xva = torch.FloatTensor(stacked[va]).to(device)
        m = ClassCondFusion(nc, nt).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=0.05, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=800)
        crit = nn.CrossEntropyLoss()
        bl, bs, pc = 1e9, None, 0
        for ep in range(800):
            m.train(); out = m(Xtr)
            a = F.softmax(m.class_weights, dim=1)
            ent = -(a*(a+1e-8).log()).sum(1).mean()
            loss = crit(out, ytr) - 0.01*ent
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            # Stopping criterion is the TRAINING objective, not a held-out
            # slice. Held-out data is never touched during fitting, so there is
            # no leakage; but note this is a best-training-loss checkpoint
            # rather than early stopping in the usual sense. The Appendix F
            # baseline comparison instead stops on a held-out slice of the
            # training partition, which is why that table reports 78.53.
            with torch.no_grad(): train_loss = crit(m(Xtr), ytr).item()
            if train_loss < bl: bl=train_loss; bs={k:v.clone() for k,v in m.state_dict().items()}; pc=0
            else:
                pc += 1
                if pc >= 100: break
        m.load_state_dict(bs); m.eval()
        with torch.no_grad(): vlog = m(Xva).cpu().numpy()
        al[va] = vlog; fa.append(accuracy_score(labels[va], vlog.argmax(1))*100)
    return fa, al

import sklearn as _sk
_SK = tuple(int(x) for x in _sk.__version__.split(".")[:2])
_MC = {} if _SK >= (1, 7) else {"multi_class": "multinomial"}


def generate_logits(features, labels, splits, C=1.0, max_iter=5000):
    """Per-fold logits over frozen features.

    The multinomial setting is required to reproduce the released logits. It
    was deprecated in scikit-learn 1.5 and removed in 1.7; on the binary
    benchmark (MHIST) the multinomial and standard binary parameterisations
    differ, so a later version produces different logits there. Pin
    scikit-learn==1.6.1 -- see requirements.txt.
    """
    if _SK >= (1, 7):
        import warnings
        warnings.warn(
            "scikit-learn >= 1.7 cannot reproduce the released MHIST logits; "
            "install scikit-learn==1.6.1", RuntimeWarning)
    nc = len(np.unique(labels))
    logits = np.zeros((len(labels), nc), dtype=np.float32)
    for tr, va in splits:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(features[tr])
        Xva = scaler.transform(features[va])
        probe = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs',
                                   **_MC, random_state=42)
        probe.fit(Xtr, labels[tr])
        tl = probe.decision_function(Xva)
        if nc == 2 and tl.ndim == 1: tl = np.column_stack([-tl, tl])
        logits[va] = tl
    return logits
