"""
JOB A -- CHECKPOINTED EXTRACTION.

The previous run was correct but had two practical problems: no visible
progress, and nothing written until the very end, so a Colab disconnect after
hours of work would lose everything. BRACS ROIs are 139-173 megapixels and
stream over Drive's FUSE mount, so full decode dominates runtime.

This version:
  * saves a shard every CHUNK images and resumes from where it stopped
  * uses PIL draft mode, which decodes JPEGs directly at reduced scale
  * raises worker count, since the bottleneck is I/O and decode, not GPU
  * prints real progress with an ETA

Safe to re-run after a disconnect: it picks up from the last shard.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None

try:
    DRIVE
except NameError:
    DRIVE = next((d for d in ["/content/drive/MyDrive", "/content/drive/My Drive"]
                  if os.path.isdir(d)), "/content/drive/MyDrive")
HET = os.path.join(DRIVE, "project_root")
CKPT = os.path.join(HET, "model_weights/ctranspath.pth")
SHARDS = "/content/ctp_shards"          # local scratch: fast, survives cell reruns
OUT = os.path.join(HET, "expanded/bracs/ctranspath_fixed")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 256
BATCH = 32
WORKERS = 8

os.makedirs(SHARDS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- model
class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True):
        super().__init__()
        assert patch_size == 4 and embed_dim % 8 == 0
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        stem, in_dim, out_dim = [], in_chans, embed_dim // 8
        for _ in range(2):
            stem += [nn.Conv2d(in_dim, out_dim, 3, 2, 1, bias=False),
                     nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
            in_dim, out_dim = out_dim, out_dim * 2
        stem.append(nn.Conv2d(in_dim, embed_dim, 1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


def build_and_load():
    import timm
    m = timm.create_model("swin_tiny_patch4_window7_224",
                          pretrained=False, num_classes=0)
    ed = getattr(m, "embed_dim", 96)
    if isinstance(ed, (list, tuple)):
        ed = ed[0]
    m.patch_embed = ConvStem(embed_dim=ed, norm_layer=nn.LayerNorm)
    sd = torch.load(CKPT, map_location="cpu")
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    missing, unexpected = m.load_state_dict(state, strict=False)
    print(f"  timm {timm.__version__}  missing={len(missing)} "
          f"unexpected={len(unexpected)}")
    if unexpected or len(missing) > 2:
        raise RuntimeError("Incomplete load -- do not extract.")
    print("  LOAD VERIFIED")
    return m.eval().to(DEV)


# ---------------------------------------------------------------- data
class FastImages(Dataset):
    """draft() lets PIL decode JPEGs at reduced scale, avoiding a full
    150-megapixel decode just to produce a 224x224 crop."""

    def __init__(self, files):
        self.files = files
        self.tf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i])
        try:
            img.draft("RGB", (448, 448))     # no-op for PNG, big win for JPEG
        except Exception:
            pass
        return self.tf(img.convert("RGB")), 0


@torch.no_grad()
def run():
    files = np.load("/content/bracs_file_order.npy", allow_pickle=True)
    y = np.load(os.path.join(HET,
                "cached_features_bracs/uni/trainval_labels.npy"))
    assert len(files) == len(y)
    n = len(files)
    chunks = [(s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]

    done = {int(f.split("_")[1].split(".")[0]) for f in os.listdir(SHARDS)
            if f.startswith("shard_")}
    todo = [(i, c) for i, c in enumerate(chunks) if i not in done]
    print(f"{n} images, {len(chunks)} shards of {CHUNK}; "
          f"{len(done)} already done, {len(todo)} remaining\n")
    if not todo:
        print("all shards present")
    else:
        model = build_and_load()
        t0 = time.time()
        for k, (idx, (a, b)) in enumerate(todo):
            dl = DataLoader(FastImages(files[a:b]), batch_size=BATCH,
                            shuffle=False, num_workers=WORKERS,
                            pin_memory=True)
            out = []
            for x, _ in dl:
                f = model(x.to(DEV, non_blocking=True))
                if f.ndim == 3:
                    f = f.mean(1)
                out.append(f.float().cpu().numpy())
            np.save(os.path.join(SHARDS, f"shard_{idx}.npy"),
                    np.concatenate(out).astype(np.float32))
            el = time.time() - t0
            rate = (k + 1) / el
            eta = (len(todo) - k - 1) / rate / 60 if rate > 0 else 0
            print(f"  shard {idx+1}/{len(chunks)}  "
                  f"({(k+1)/len(todo)*100:5.1f}%)  "
                  f"elapsed {el/60:5.1f}m  ETA {eta:5.1f}m", flush=True)

    # ------------------------------------------------------------ assemble
    parts = [np.load(os.path.join(SHARDS, f"shard_{i}.npy"))
             for i in range(len(chunks))]
    F = np.concatenate(parts)
    assert len(F) == n, f"{len(F)} != {n}"
    np.save(os.path.join(OUT, "trainval_features.npy"), F)
    np.save(os.path.join(OUT, "trainval_labels.npy"), y)
    print(f"\nsaved {F.shape} -> {OUT}")
    return F, y


def linear_cka(X, Y):
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    d = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(np.linalg.norm(Y.T @ X, "fro") ** 2 / d) if d else 0.0


def diagnose(F, y, name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import balanced_accuracy_score
    F = np.asarray(F, np.float64)
    Xc = F - F.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / s.sum()
    eff = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    pred = cross_val_predict(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        F, y, cv=5)
    acc = (pred == y).mean() * 100
    bal = balanced_accuracy_score(y, pred) * 100
    print(f"  [{name}] dim={F.shape[1]} eff_rank={eff:.1f} "
          f"({eff/F.shape[1]*100:.0f}%) norm={np.linalg.norm(F,axis=1).mean():.2f}")
    print(f"           acc={acc:.2f}%  bal_acc={bal:.2f}%")
    return acc, bal


if __name__ == "__main__":
    F_new, y = run()
    F_old = np.load(os.path.join(HET,
                    "cached_features_bracs/ctranspath/trainval_features.npy"))
    uni = np.load(os.path.join(HET,
                  "cached_features_bracs/uni/trainval_features.npy"))
    print("\n" + "=" * 66)
    a_old, _ = diagnose(F_old, y, "original (broken patch embed)")
    a_new, _ = diagnose(F_new, y, "refit (ConvStem loaded)")
    print(f"\n  majority baseline: {np.bincount(y).max()/len(y)*100:.2f}%")
    print(f"  CKA with UNI: original={linear_cka(uni, F_old):.4f} "
          f"(paper 0.246)  refit={linear_cka(uni, F_new):.4f}")
    print(f"\n  accuracy {a_old:.2f}% -> {a_new:.2f}%  ({a_new-a_old:+.2f})")
    print("=" * 66)
