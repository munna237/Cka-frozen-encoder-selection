"""Single source of truth for paths, datasets, encoders and the pre-registered
analysis parameters of v3.

Everything here is hashed into prereg.json by notebook 00. Notebook 04 refuses
to analyse if this file (or any analysis module) changed after registration,
unless you record a deviation explicitly. Edit the PATHS block if your Drive
layout differs; do not edit PREREG after registering.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# PATHS  (taken from your existing Drive layout; edit if anything moved)
# ---------------------------------------------------------------------------
DRIVE = os.environ.get("V3_DRIVE") or next(
    (d for d in ("/content/drive/MyDrive", "/content/drive/My Drive") if os.path.isdir(d)),
    "/content/drive/MyDrive")
HET = os.path.join(DRIVE, "HetFuseKD")

PROJECT = os.path.join(DRIVE, "TMLR_v3")          # everything v3 writes lives here
CODE_DIR = os.path.join(PROJECT, "code")
FEAT_DIR = os.path.join(PROJECT, "features")      # features/<dataset>/<encoder>.npy
MANIFEST_DIR = os.path.join(PROJECT, "manifests")  # manifests/<dataset>.csv + .json
IMG_DRIVE_DIR = os.path.join(PROJECT, "images224")  # persisted image caches (Drive-sourced sets only)
RESULTS_DIR = os.path.join(PROJECT, "results")
LOG_DIR = os.path.join(PROJECT, "logs")
PREREG_PATH = os.path.join(PROJECT, "prereg.json")

LOCAL = os.environ.get("V3_LOCAL", "/content/v3_local")  # fast scratch; lost on disconnect
LOCAL_IMG = os.path.join(LOCAL, "images224")
LOCAL_DL = os.path.join(LOCAL, "downloads")
LOCAL_SHARDS = os.path.join(LOCAL, "shards")

# your datasets already on Drive
BRACS_DIR = os.path.join(HET, "data", "BRACS_RoI", "latest_version")   # train/ val/ test/
SICAP_DIR = os.path.join(DRIVE, "SICAPv2")                             # images/ partition/Test/
MHIST_DIR = os.path.join(DRIVE, "MHIST")                               # images/ + annotations csv
MHIST_CSV_CANDIDATES = ("annotations (1).csv", "annotations.csv")
CTRANSPATH_CKPT = os.path.join(HET, "model_weights", "ctranspath.pth")

# v2 caches, used ONLY by the consistency check in notebook 01
V2_CACHE = {
    "bracs":   os.path.join(HET, "cached_features_bracs"),
    "sicapv2": os.path.join(HET, "cached_features_sicapv2"),
    "mhist":   os.path.join(HET, "cached_features_mhist"),
}
V2_CTRANSPATH_FIXED = {ds: os.path.join(HET, "expanded", ds, "ctranspath_fixed")
                       for ds in ("bracs", "sicapv2", "mhist")}
V2_NAME = {"uni": "uni", "phikon_v2": "phikon_v2", "ctranspath": "ctranspath",
           "resnet50_in1k": "resnet50_imagenet"}

# public downloads (all verified reachable on 2026-10-21)
ZENODO_NCT = "https://zenodo.org/api/records/1214456/files/{}/content"
ZENODO_PCAM = "https://zenodo.org/api/records/2546921/files/{}/content"

# ---------------------------------------------------------------------------
# DATASETS
#   cap      : max rows kept (stratified, seeded); None = all
#   persist  : also save the 224px uint8 image cache to Drive (only worth it for
#              sets that are slow to re-read, i.e. those stored on Drive)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
SEED = 0

PATHOLOGY_DATASETS = {
    "bracs":      dict(cap=None,   persist=True,  note="train+val RoIs (3,969); grouped by WSI id"),
    "sicapv2":    dict(cap=None,   persist=True,  note="partition/Test Train+Test (12,081); grouped by biopsy"),
    "mhist":      dict(cap=None,   persist=True,  note="all 3,152; no patient ids released -> ungrouped"),
    "crc_val_7k": dict(cap=None,   persist=False, note="CRC-VAL-HE-7K, Zenodo 1214456; ungrouped"),
    "nct_crc_100k": dict(cap=20000, persist=False, note="NCT-CRC-HE-100K (11.7 GB download); ungrouped"),
    "pcam":       dict(cap=20000,  persist=False, note="PCam test split (32,768 -> 20k); grouped by WSI"),
}

VISION_DATASETS = {
    "dtd":        dict(cap=None,  persist=False, note="torchvision split=train (1,880), as v2"),
    "aircraft":   dict(cap=None,  persist=False, note="torchvision split=train (3,334), as v2"),
    "eurosat":    dict(cap=20000, persist=False, note="27,000 -> 20k (v2 used all)"),
    "pets":       dict(cap=None,  persist=False, note="OxfordIIITPet trainval (3,680)"),
    "food101":    dict(cap=20000, persist=False, note="train 75,750 -> 20k"),
    "cifar100":   dict(cap=20000, persist=False, note="train 50,000 -> 20k; 32px upsampled"),
    "gtsrb":      dict(cap=20000, persist=False, note="train 26,640 -> 20k"),
    "country211": dict(cap=20000, persist=False, note="train 31,650 -> 20k"),
}

TEXT_DATASETS = {
    # name: (hf_id, config, split, text_col, label_col, cap)
    "emotion":    ("SetFit/emotion", None, "train", "text", "label", 20000),
    "ag_news":    ("fancyzhx/ag_news", None, "train", "text", "label", 20000),
    "banking77":  ("mteb/banking77", None, "train", "text", "label", 20000),
    "trec_coarse": ("SetFit/TREC-QC", None, "train", "text", "label_coarse", 20000),
    "newsgroups20": ("SetFit/20_newsgroups", None, "train", "text", "label", 20000),
    "sst5":       ("SetFit/sst5", None, "train", "text", "label", 20000),
}

# ---------------------------------------------------------------------------
# ENCODERS
#   kind   : timm | hf | ctranspath | st (sentence-transformers)
#   pool   : default (model's own pooled output) | cls+meanpatch (Virchow-style)
#   gated  : needs an accepted licence on Hugging Face + HF_TOKEN in Colab secrets
# Every repo id below was checked to exist on 2026-10-21.
# ---------------------------------------------------------------------------
PATHOLOGY_ENCODERS = {
    # the four v2 encoders
    "uni":           dict(kind="timm", id="hf-hub:MahmoodLab/uni", gated=True,
                          kw=dict(init_values=1e-5, dynamic_img_size=True)),
    "phikon_v2":     dict(kind="hf", id="owkin/phikon-v2", pool="cls"),
    "ctranspath":    dict(kind="ctranspath"),
    "resnet50_in1k": dict(kind="timm", id="resnet50.tv_in1k"),
    # new public pathology encoders
    "phikon":        dict(kind="hf", id="owkin/phikon", pool="cls"),
    "uni2_h":        dict(kind="timm", id="hf-hub:MahmoodLab/UNI2-h", gated=True, kw="UNI2_H"),
    "virchow2":      dict(kind="timm", id="hf-hub:paige-ai/Virchow2", gated=True, kw="SWIGLU",
                          pool="cls+meanpatch"),
    "h_optimus_0":   dict(kind="timm", id="hf-hub:bioptimus/H-optimus-0", gated=True,
                          kw=dict(init_values=1e-5, dynamic_img_size=False)),
    "prov_gigapath": dict(kind="timm", id="hf-hub:prov-gigapath/prov-gigapath", gated=True),
    "hibou_b":       dict(kind="hf", id="histai/hibou-b", gated=True, pool="pooler",
                          trust_remote_code=True),
    "lunit_vits16":  dict(kind="timm", id="hf-hub:1aurent/vit_small_patch16_224.lunit_dino"),
    "kaiko_vitb16":  dict(kind="timm",
                          id="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
                          kw=dict(dynamic_img_size=True), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

VISION_ENCODERS = {
    # the six v2 encoders (same timm tags as the v2 w3 cache)
    "dinov2_l":     dict(kind="timm", id="vit_large_patch14_dinov2.lvd142m"),
    "eva02_clip":   dict(kind="timm", id="eva02_large_patch14_clip_224.merged2b"),
    "swin_l":       dict(kind="timm", id="swin_large_patch4_window7_224.ms_in22k"),
    "convnext_l":   dict(kind="timm", id="convnext_large.fb_in22k"),
    "resnet50":     dict(kind="timm", id="resnet50.a1_in1k"),
    "vit_mae":      dict(kind="timm", id="vit_base_patch16_224.mae"),
    # new
    "clip_b32":     dict(kind="timm", id="vit_base_patch32_clip_224.openai"),
    "siglip_b16":   dict(kind="timm", id="vit_base_patch16_siglip_224.webli"),
    "dinov2_b":     dict(kind="timm", id="vit_base_patch14_dinov2.lvd142m"),
    "openclip_h14": dict(kind="timm", id="vit_huge_patch14_clip_224.laion2b"),
}

TEXT_ENCODERS = {
    "minilm_l6":   dict(kind="st", id="sentence-transformers/all-MiniLM-L6-v2"),
    "mpnet_base":  dict(kind="st", id="sentence-transformers/all-mpnet-base-v2"),
    "distilroberta": dict(kind="st", id="sentence-transformers/all-distilroberta-v1"),
    "bge_small":   dict(kind="st", id="BAAI/bge-small-en-v1.5"),
    "bge_base":    dict(kind="st", id="BAAI/bge-base-en-v1.5"),
    "gte_base":    dict(kind="st", id="thenlper/gte-base"),
    "e5_base":     dict(kind="st", id="intfloat/e5-base-v2", prefix="query: "),
    "minilm_l3":   dict(kind="st", id="sentence-transformers/paraphrase-MiniLM-L3-v2"),
    "glove_avg":   dict(kind="st", id="sentence-transformers/average_word_embeddings_glove.6B.300d"),
}

DOMAINS = {
    "pathology": (PATHOLOGY_DATASETS, PATHOLOGY_ENCODERS),
    "vision": (VISION_DATASETS, VISION_ENCODERS),
    "text": (TEXT_DATASETS, TEXT_ENCODERS),
}

FEATURE_DTYPE = "float32"    # float16 halves Drive use but adds ~0.01 CKA error (v2 App. B.4)
EXTRACT_BATCH = 64
SHARD_ROWS = 2048            # features are flushed to Drive every SHARD_ROWS rows

# ---------------------------------------------------------------------------
# PRE-REGISTERED ANALYSIS PARAMETERS  (do not edit after notebook 00 registers)
# ---------------------------------------------------------------------------
PREREG = dict(
    unit_of_inference="pool (= one dataset with all available encoders of its domain)",
    k_folds=5,
    seed=SEED,
    grouped_folds_where_groups_exist=True,
    probe="StandardScaler (train fold) + L2 logistic regression, C=1.0",
    probe_backend="torch",        # exact optimum; sklearn default tol stops early (tests/)
    anchor_rule="highest individual held-out accuracy",
    base_rule="anchor + median-CKA member (CKA ordering), fixed before pairing",
    comp_tol_pp=1.5,
    cka_min_gap=0.15,
    cka_sd_floor=3.0,
    n_cap_cka=10000,
    primary_fusion_rule="mean_logits",
    secondary_fusion_rules=["mean_probs", "class_conditional"],
    primary_estimand="A: cluster-robust (by pool) mean of D over the ONE primary matched "
                     "pair per pool (max CKA gap), t(G-1) reference",
    equivalence_margins_pp=[1.0, 0.5, 0.25],
    dose_response="D on CKA gap over all matched pairs, cluster-robust by pool; "
                  "interpret slope only if MDE(80%) <= 0.5 pp per unit CKA",
    regret_criteria=["cka_diversity", "disagreement_with_base", "competence", "oracle_headroom"],
    regret_note="only cka_diversity is label-free; disagreement uses probe predictions",
    mechanism="utility ~ disagreement + acc ; utility ~ cka + acc ; joint; cluster-robust",
    analyses=["all pools", "pathology", "vision", "text", "image pools (pathology+vision)"],
    min_encoders_per_pool=5,
)
