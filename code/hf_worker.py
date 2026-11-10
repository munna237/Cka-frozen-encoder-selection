"""Feature extraction for a Hugging Face model whose remote code needs an OLDER
transformers than Colab ships (Hibou-B: its custom DINOv2 code imports helpers
removed in transformers 5).

Run as a separate process with PYTHONPATH pointing at an isolated install of
transformers 4.46 (see extract.ensure_old_transformers). torch, numpy and the
GPU come from the normal Colab environment. Output format is identical to the
in-process path: shard_XXXX.npy files plus a load report.

Usage (called by extract.extract_hf_isolated):
    python hf_worker.py --model histai/hibou-b --pool pooler --images X.npy --shards DIR
"""

import argparse
import json
import os
import time

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default="pooler", choices=["pooler", "cls"])
    ap.add_argument("--images", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--shard_rows", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trust_remote_code", action="store_true")
    a = ap.parse_args()

    import transformers
    from transformers import AutoImageProcessor, AutoModel
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    os.makedirs(a.shards, exist_ok=True)

    model = AutoModel.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    model.eval().to(dev)
    try:
        proc = AutoImageProcessor.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
        mean, std = list(proc.image_mean), list(proc.image_std)
    except Exception:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    n_params = int(sum(p.numel() for p in model.parameters()))
    report = {"transformers": transformers.__version__, "torch": torch.__version__,
              "model": a.model, "class": type(model).__name__, "n_params": n_params,
              "mean": mean, "std": std, "pool": a.pool}
    print(f"  [{a.model}] transformers {transformers.__version__}  {type(model).__name__}  "
          f"{n_params / 1e6:.0f}M params  mean={mean} std={std}", flush=True)
    with open(os.path.join(a.shards, "load_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    imgs = np.load(a.images, mmap_mode="r")
    n = imgs.shape[0]
    m_t = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    s_t = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    ranges = [(s, min(s + a.shard_rows, n)) for s in range(0, n, a.shard_rows)]
    todo = [i for i in range(len(ranges))
            if not os.path.exists(os.path.join(a.shards, f"shard_{i:04d}.npy"))]
    print(f"  [{a.model}] {n} images, {len(todo)}/{len(ranges)} shards to go", flush=True)
    t0 = time.time()
    use_amp = dev.startswith("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        for k, i in enumerate(todo):
            lo, hi = ranges[i]
            out = []
            for s in range(lo, hi, a.batch):
                x = torch.from_numpy(np.ascontiguousarray(imgs[s:min(s + a.batch, hi)]))
                x = x.to(dev).permute(0, 3, 1, 2).float().div_(255.0)
                o = model(pixel_values=(x - m_t) / s_t)
                if a.pool == "pooler" and getattr(o, "pooler_output", None) is not None:
                    f = o.pooler_output
                else:
                    f = o.last_hidden_state[:, 0]
                out.append(f.float().cpu().numpy())
            np.save(os.path.join(a.shards, f"shard_{i:04d}.npy"),
                    np.concatenate(out).astype(np.float32))
            el = time.time() - t0
            print(f"      shard {i + 1}/{len(ranges)}  {hi}/{n}  "
                  f"ETA {el / (k + 1) * (len(todo) - k - 1) / 60:5.1f} min", flush=True)


if __name__ == "__main__":
    main()
