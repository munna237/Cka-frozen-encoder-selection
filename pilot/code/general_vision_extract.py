# ==========================================================================
# JOB 1, STEP 1 -- regenerate the general-vision feature cache.
#
# The cache lived on local disk and did not survive the session. This
# rebuilds it and copies it to Drive, so Job 1 (and any rerun) can use it
# without extracting again.
#
# Needs a GPU runtime. Roughly 45-60 minutes: six encoders over three
# benchmarks, dominated by EuroSAT's 27,000 images and the model downloads.
# Resumable -- already-cached (dataset, encoder) pairs are skipped.
# ==========================================================================
import os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL = "/content/w3/cache"
DRIVE_CACHE = "./gv_cache"
os.makedirs(LOCAL, exist_ok=True)
os.makedirs(DRIVE_CACHE, exist_ok=True)
BATCH, WORKERS = 64, 4

if DEV != "cuda":
    print("*** No GPU detected. Runtime > Change runtime type > GPU, then rerun. ***")

ENCODERS = {
    "dinov2_l":   "vit_large_patch14_dinov2.lvd142m",
    "eva02_clip": "eva02_large_patch14_clip_224.merged2b",
    "swin_l":     "swin_large_patch4_window7_224.ms_in22k",
    "convnext_l": "convnext_large.fb_in22k",
    "resnet50":   "resnet50.a1_in1k",
    "vit_mae":    "vit_base_patch16_224.mae",
}
DATASETS = {
    "dtd":      lambda tf: datasets.DTD("/content/data", split="train",
                                        transform=tf, download=True),
    "aircraft": lambda tf: datasets.FGVCAircraft("/content/data", split="train",
                                                 transform=tf, download=True),
    "eurosat":  lambda tf: datasets.EuroSAT("/content/data", transform=tf,
                                            download=True),
}


def cached(ds, key):
    """Return a path if this pair is already available locally or on Drive."""
    for root in (LOCAL, DRIVE_CACHE):
        p = os.path.join(root, f"{ds}_{key}.npz")
        if os.path.exists(p):
            return p
    return None


@torch.no_grad()
def extract(ds, key):
    import timm
    m = timm.create_model(ENCODERS[key], pretrained=True,
                          num_classes=0).eval().to(DEV)
    cfg = timm.data.resolve_data_config({}, model=m)
    tf = timm.data.create_transform(**cfg)
    dl = DataLoader(DATASETS[ds](tf), batch_size=BATCH, shuffle=False,
                    num_workers=WORKERS, pin_memory=True)
    F, Y = [], []
    for x, y in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            f = m(x.to(DEV, non_blocking=True))
        if f.ndim == 3:
            f = f.mean(1)
        F.append(f.float().cpu().numpy())
        Y.append(np.asarray(y))
    del m
    torch.cuda.empty_cache()
    return np.concatenate(F).astype(np.float32), np.concatenate(Y)


todo = [(ds, k) for ds in DATASETS for k in ENCODERS if cached(ds, k) is None]
have = 18 - len(todo)
print(f"{have}/18 pairs already cached, {len(todo)} to extract\n")

t0 = time.time()
for i, (ds, key) in enumerate(todo):
    t1 = time.time()
    F, y = extract(ds, key)
    local_p = os.path.join(LOCAL, f"{ds}_{key}.npz")
    np.savez_compressed(local_p, F=F, y=y)
    try:                              # persist immediately, not at the end
        import shutil
        shutil.copy(local_p, os.path.join(DRIVE_CACHE, f"{ds}_{key}.npz"))
        saved = "-> Drive"
    except Exception as e:
        saved = f"(Drive copy failed: {e})"
    el = time.time() - t0
    eta = (len(todo) - i - 1) * el / (i + 1) / 60
    print(f"  [{i+1:2d}/{len(todo)}] {ds:9s} {key:12s} {str(F.shape):14s} "
          f"{time.time()-t1:5.0f}s  ETA {eta:5.1f}m  {saved}", flush=True)

# pull anything that only exists on Drive into the local cache for Job 1
import shutil
for f in os.listdir(DRIVE_CACHE):
    if f.endswith(".npz") and not os.path.exists(os.path.join(LOCAL, f)):
        shutil.copy(os.path.join(DRIVE_CACHE, f), os.path.join(LOCAL, f))

n = len([f for f in os.listdir(LOCAL) if f.endswith(".npz")])
tot = sum(os.path.getsize(os.path.join(LOCAL, f)) for f in os.listdir(LOCAL))
print(f"\ncache complete: {n}/18 pairs, {tot/1e6:.0f} MB at {LOCAL}")
print(f"mirrored to {DRIVE_CACHE}")
print("\nNow run job1_fixed_base.py.")
