"""CTransPath feature extraction, run as a SEPARATE PROCESS under timm 0.5.4.

Why a separate process: CTransPath's released checkpoint matches the Swin
implementation of timm 0.5.4, which is the setup v2 verified (all 200 entries,
missing=0, unexpected=0, 16 ConvStem tensors). The other encoders need a modern
timm. Two timm versions cannot coexist in one Python process, so this script
runs with PYTHONPATH pointing at an isolated timm 0.5.4 install.

Hard guard: the script refuses to extract unless the checkpoint load reports
zero missing and zero unexpected keys AND the ConvStem tensors were loaded.
That is the exact check whose absence produced the v1 bug.

Usage (called by extract.extract_ctranspath):
    python ctranspath_worker.py --images X.npy --ckpt ctranspath.pth --shards DIR
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn


class ConvStem(nn.Module):
    """Convolutional patch embedding of CTransPath (Wang et al., 2022)."""

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
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


def build(ckpt):
    import timm
    if not timm.__version__.startswith("0.5."):
        raise RuntimeError(f"CTransPath worker must run under timm 0.5.x, got {timm.__version__}")
    m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
    ed = getattr(m, "embed_dim", 96)
    if isinstance(ed, (list, tuple)):
        ed = ed[0]
    m.patch_embed = ConvStem(embed_dim=ed, norm_layer=nn.LayerNorm)
    # weights_only=False: torch>=2.6 defaults to True, which rejects this
    # (trusted, your own) checkpoint format.
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    res = m.load_state_dict(state, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    stem_keys = [k for k in state if k.startswith("patch_embed")]
    report = {"timm": timm.__version__, "torch": torch.__version__,
              "ckpt_entries": len(state), "missing": missing, "unexpected": unexpected,
              "patch_embed_tensors_in_ckpt": len(stem_keys)}
    print(f"  [ctranspath] timm {timm.__version__}  entries={len(state)}  "
          f"missing={len(missing)}  unexpected={len(unexpected)}  "
          f"ConvStem tensors={len(stem_keys)}", flush=True)
    if missing or unexpected or not stem_keys:
        raise SystemExit("INCOMPLETE CTransPath LOAD - refusing to extract "
                         f"(missing {missing[:5]}, unexpected {unexpected[:5]})")
    print("  [ctranspath] LOAD VERIFIED", flush=True)
    return m.eval(), report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--shard_rows", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    os.makedirs(a.shards, exist_ok=True)
    model, report = build(a.ckpt)
    model.to(dev)
    with open(os.path.join(a.shards, "load_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    imgs = np.load(a.images, mmap_mode="r")
    n = imgs.shape[0]
    mean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1)
    ranges = [(s, min(s + a.shard_rows, n)) for s in range(0, n, a.shard_rows)]
    todo = [i for i in range(len(ranges))
            if not os.path.exists(os.path.join(a.shards, f"shard_{i:04d}.npy"))]
    print(f"  [ctranspath] {n} images, {len(todo)}/{len(ranges)} shards to go", flush=True)
    t0 = time.time()
    with torch.no_grad():                       # fp32, as in the v2 extraction
        for k, i in enumerate(todo):
            lo, hi = ranges[i]
            out = []
            for s in range(lo, hi, a.batch):
                x = torch.from_numpy(np.ascontiguousarray(imgs[s:min(s + a.batch, hi)]))
                x = x.to(dev).permute(0, 3, 1, 2).float().div_(255.0)
                f = model((x - mean) / std)
                if f.ndim == 3:
                    f = f.mean(1)
                out.append(f.float().cpu().numpy())
            np.save(os.path.join(a.shards, f"shard_{i:04d}.npy"),
                    np.concatenate(out).astype(np.float32))
            el = time.time() - t0
            print(f"      shard {i + 1}/{len(ranges)}  {hi}/{n}  "
                  f"ETA {el / (k + 1) * (len(todo) - k - 1) / 60:5.1f} min", flush=True)


if __name__ == "__main__":
    main()
