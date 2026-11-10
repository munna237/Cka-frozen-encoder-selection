"""Encoder construction and frozen-feature extraction.

Every encoder is wrapped so that it takes a uint8 batch (B, 224, 224, 3) and
returns float32 features (B, d). Each wrapper records what it actually did
(input size, mean/std, pooling, parameter count, weight checksum) so the paper
can state it and a reviewer can check it.

Load safety. The v1 CTransPath failure was a checkpoint silently loaded into the
wrong module. For hub models we rely on the library's strict load (timm and
transformers both raise on missing weights unless told otherwise) and we add a
second guard after extraction: feature statistics (effective rank, dead
dimensions, finiteness) and a quick probe accuracy per encoder are tabulated so
an encoder that "works" but is degenerate stands out. CTransPath, whose
checkpoint we load ourselves, runs in a separate process with the original
timm 0.5.4 and an explicit full-load assertion (ctranspath_worker.py).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

import config as CFG

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def hf_login():
    """Use HF_TOKEN from Colab secrets (key icon in the left bar) if present."""
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
        except Exception:
            tok = None
    if tok:
        os.environ["HF_TOKEN"] = tok
        try:
            from huggingface_hub import login
            login(token=tok, add_to_git_credential=False)
        except Exception as e:
            print("  HF login failed:", e)
        return True
    print("  No HF_TOKEN found: gated encoders (UNI, UNI2-h, Virchow2, H-optimus-0, "
          "Prov-GigaPath, Hibou-B) will be skipped.")
    return False


def _timm_kwargs(spec):
    import torch
    kw = spec.get("kw") or {}
    if kw == "UNI2_H":
        import timm
        return dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
                    embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                    mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                    reg_tokens=8, dynamic_img_size=True)
    if kw == "SWIGLU":
        import timm
        return dict(mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
    return dict(kw)


def _shim_transformers_onnx():
    """transformers >= 5 removed `transformers.onnx`. Hibou-B's remote
    configuration file still imports `OnnxConfig` from it, only to define an
    ONNX-export helper class that feature extraction never uses. Register a
    stand-in module so that import succeeds. Does nothing if the real module
    exists. This changes no weights and no forward computation."""
    import importlib
    import sys
    import types
    try:
        importlib.import_module("transformers.onnx")
        return
    except Exception:
        pass
    mod = types.ModuleType("transformers.onnx")

    class OnnxConfig:                                   # never instantiated for inference
        default_onnx_opset = 14

        def __init__(self, *a, **k):
            raise RuntimeError("ONNX export is not available (stand-in module)")

    mod.OnnxConfig = OnnxConfig
    mod.OnnxConfigWithPast = OnnxConfig
    mod.OnnxSeq2SeqConfigWithPast = OnnxConfig
    sys.modules["transformers.onnx"] = mod
    try:
        import transformers
        transformers.onnx = mod
    except Exception:
        pass


class Encoder:
    def __init__(self, name, spec, device):
        self.name, self.spec, self.device = name, spec, device
        self.kind = spec["kind"]
        self.info = {"name": name, "kind": self.kind, "id": spec.get("id")}
        if self.kind == "timm":
            self._build_timm()
        elif self.kind == "hf":
            self._build_hf()
        else:
            raise ValueError(f"{self.kind} is not built in-process")
        self.model.eval().to(device)
        p = sum(t.numel() for t in self.model.parameters())
        self.info["n_params"] = int(p)
        self.info["weight_checksum"] = self._checksum()

    # -------------------------------------------------------------- builders
    def _build_timm(self):
        import timm
        kw = _timm_kwargs(self.spec)
        kw.setdefault("num_classes", 0)
        m = timm.create_model(self.spec["id"], pretrained=True, **kw)
        cfg = timm.data.resolve_model_data_config(m)
        native = tuple(cfg.get("input_size", (3, 224, 224)))[-1]
        if native != CFG.IMAGE_SIZE and "dynamic_img_size" not in kw and "img_size" not in kw:
            # e.g. DINOv2 defaults to 518. Rebuild at 224; timm resamples the
            # position embedding from the checkpoint. Recorded in self.info.
            # CNNs (ResNet, ConvNeXt) take any input size and reject these ViT-only
            # arguments; they are simply run at 224 as built.
            try:
                m = timm.create_model(self.spec["id"], pretrained=True,
                                      img_size=CFG.IMAGE_SIZE, **kw)
                self.info["resized_from"] = native
            except TypeError:
                try:
                    m = timm.create_model(self.spec["id"], pretrained=True,
                                          dynamic_img_size=True, **kw)
                    self.info["dynamic_from"] = native
                except TypeError:
                    self.info["run_at_224_native"] = native
        self.model = m
        self.mean = tuple(self.spec.get("mean") or cfg.get("mean") or IMAGENET_MEAN)
        self.std = tuple(self.spec.get("std") or cfg.get("std") or IMAGENET_STD)
        self.pool = self.spec.get("pool", "default")
        self.info.update(mean=self.mean, std=self.std, pool=self.pool,
                         timm_version=timm.__version__,
                         num_prefix_tokens=int(getattr(m, "num_prefix_tokens", 0) or 0))

    def _build_hf(self):
        import transformers
        from transformers import AutoImageProcessor, AutoModel
        trc = bool(self.spec.get("trust_remote_code", False))
        if trc:
            _shim_transformers_onnx()
        self.model = AutoModel.from_pretrained(self.spec["id"], trust_remote_code=trc)
        try:
            proc = AutoImageProcessor.from_pretrained(self.spec["id"], trust_remote_code=trc)
            self.mean = tuple(proc.image_mean)
            self.std = tuple(proc.image_std)
        except Exception:
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        self.pool = self.spec.get("pool", "cls")
        self.info.update(mean=self.mean, std=self.std, pool=self.pool,
                         transformers_version=transformers.__version__)

    def _checksum(self):
        h = hashlib.sha256()
        for i, t in enumerate(self.model.state_dict().values()):
            if i % 7 == 0 and t.numel():
                h.update(np.ascontiguousarray(t.detach().float().flatten()[:64].cpu().numpy()).tobytes())
        return h.hexdigest()[:16]

    # -------------------------------------------------------------- forward
    def __call__(self, x_uint8):
        """x_uint8: torch uint8 tensor (B, H, W, 3) on any device."""
        import torch
        x = x_uint8.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor(self.mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        use_amp = self.device.startswith("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            if self.kind == "timm":
                if self.pool == "cls+meanpatch":
                    tok = self.model.forward_features(x)
                    npt = self.info["num_prefix_tokens"] or 1
                    f = torch.cat([tok[:, 0], tok[:, npt:].mean(1)], dim=-1)
                else:
                    f = self.model(x)
                    if f.ndim == 3:
                        f = f.mean(1)
                    elif f.ndim == 4:
                        f = f.mean((2, 3))
            else:
                out = self.model(pixel_values=x)
                if self.pool == "pooler" and getattr(out, "pooler_output", None) is not None:
                    f = out.pooler_output
                else:
                    f = out.last_hidden_state[:, 0]
        return f.float().cpu().numpy()


def build(name, spec, device):
    return Encoder(name, spec, device)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def feature_stats(F):
    F = np.asarray(F, np.float64)
    Fc = F - F.mean(0, keepdims=True)
    s = np.linalg.svd(Fc[: min(len(Fc), 4000)], compute_uv=False)
    p = s ** 2 / max((s ** 2).sum(), 1e-12)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    return {"n": int(F.shape[0]), "dim": int(F.shape[1]),
            "finite": bool(np.isfinite(F).all()),
            "mean_norm": float(np.linalg.norm(F, axis=1).mean()),
            "eff_rank": eff_rank, "eff_rank_frac": eff_rank / F.shape[1],
            "dead_dims_frac": float((F.std(0) < 1e-8).mean())}
