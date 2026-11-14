"""
app.py — Flask backend for the RetinaCare diabetic-retinopathy screening app
=============================================================================

Serves the existing `index.html` and exposes a JSON API that runs the real,
trained `DRClassificationModel` (Swin-Tiny backbone + multi-scale residual
adapters + cross-scale attention + CORAL ordinal head) from
`visionx-classification2.ipynb` on an uploaded fundus image.

WHAT THIS FILE DOES
--------------------
1. Loads `dr_model_config.pth` — a plain config dict saved at the end of the
   notebook (backbone name, input size, class names, normalisation stats,
   referable threshold, Grad-CAM target layer, etc).
2. Rebuilds `DRClassificationModel` **exactly** as defined in the notebook
   (cell 16) and loads `dr_swin_deployment.pth` into it with `strict=True`.
   This checkpoint is a plain `model.state_dict()` of that exact class — no
   extra submodules, so nothing is guessed or invented here.
3. Reproduces the notebook's `preprocess_image()` (Ben Graham enhancement +
   CLAHE) and `val_transform` (ToTensor + Normalize) byte-for-byte, so the
   model sees the same kind of input it was trained/validated on.
4. Runs a single forward+backward pass per request that produces, all from
   real model internals:
     - DR grade (0-4) and referable-DR flag, decoded from the CORAL
       cumulative logits exactly as in the notebook's evaluation loop.
     - A per-class probability distribution derived from those same CORAL
       cumulative probabilities (there is no separate 5-way softmax head in
       this architecture — the notebook itself decodes grade this way).
     - A Grad-CAM heatmap over `backbone.layers_3.blocks[-1]` — the same
       target layer the notebook uses for its own Grad-CAM demo.
     - A second, genuinely different overlay ("scale-attentive evidence
       map"): the model's own `ScaleSpecificEvidencePooling` attention maps
       from all four backbone stages, combined using the *learned* weights
       of `OrdinalScaleAttentiveHead.scale_weights`. This is real, trained
       model state — not a fabricated or randomly-initialised module — and
       is sent to the frontend under the `ggsa_overlay_base64` key so the
       existing `index.html` (which already renders Grad-CAM + a second
       attention overlay as "Visual Evidence", not as a segmentation mask)
       works unchanged. See `render_scale_evidence_overlay()` below for the
       exact math — it is documented as multi-scale attention, not claimed
       to be a lesion segmentation mask.
5. A lightweight, non-clinical image-quality heuristic (blur / exposure /
   field-of-view) that is NOT part of the trained model.

NOTE ON `dr_swin_explainability.pth`
-------------------------------------
Per the notebook's own save cell, `dr_swin_deployment.pth` and
`dr_swin_explainability.pth` are saved from the exact same
`model.state_dict()` — they are byte-for-byte the same weights under two
filenames. Only one needs to be loaded at runtime; this file loads
`dr_swin_deployment.pth` by default. Keeping `dr_swin_explainability.pth`
alongside it on disk is optional and not required for the app to run.

DEPLOYMENT LAYOUT
------------------
    app.py
    requirements.txt
    index.html
    model_files/
        dr_model_config.pth
        dr_swin_deployment.pth
        dr_swin_explainability.pth   (optional — identical weights, not loaded)

Place your `dr_swin_deployment_files` folder's contents into `model_files/`
next to this file (or point `DR_MODEL_DIR` at wherever you put it).

RUNNING LOCALLY
----------------
    pip install -r requirements.txt
    python app.py
    # -> http://localhost:5000

RUNNING IN PRODUCTION (AWS EC2 / behind nginx or an ALB)
-----------------------------------------------------------
    gunicorn -w 2 -k gthread --threads 4 -b 0.0.0.0:8000 --timeout 120 app:app

  - `-w 2`: 2 worker processes. Each loads its own copy of the model
    (~110MB of weights), so size the instance accordingly (a t3.large /
    2 vCPU, 8GB RAM instance comfortably runs 2 workers on CPU).
  - `--timeout 120`: CPU inference on a Swin-Tiny backbone can take a few
    seconds per image; the default 30s gunicorn timeout is too tight.
  - Put nginx or an ALB in front for TLS termination; this app itself only
    speaks plain HTTP. The frontend already calls relative URLs
    (`/api/screen`, `/api/health`), so no frontend changes are needed once
    both are served from the same host.
  - Each worker process serialises its own requests through a lock (see
    `INFERENCE_LOCK` below) so concurrent uploads on `--threads 4` can't
    corrupt the shared Grad-CAM/attention hook state; they simply queue.
"""

import base64
import logging
import os
import re
import threading
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ============================================================================
# CONFIG (env-overridable — all have sane defaults for local/dev use)
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("DR_MODEL_DIR", os.path.join(BASE_DIR, "model_files"))
WEIGHTS_FILENAME = os.environ.get("DR_MODEL_WEIGHTS", "dr_swin_deployment.pth")
CONFIG_FILENAME = os.environ.get("DR_MODEL_CONFIG", "dr_model_config.pth")

DEVICE_PREF = os.environ.get("DR_DEVICE", "auto")  # "auto" | "cpu" | "cuda"
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "12"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")  # tighten this in production
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retinacare")


def resolve_device() -> torch.device:
    if DEVICE_PREF == "cpu":
        return torch.device("cpu")
    if DEVICE_PREF == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = resolve_device()

# ============================================================================
# MODEL ARCHITECTURE — exact copy of visionx-classification2.ipynb (cell 16)
# ============================================================================


class BottleneckResidualAdapter(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(8, hidden)
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, hidden)
        self.up = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.down(x)))
        x = self.act(self.norm2(self.conv(x)))
        x = self.norm3(self.up(x))
        return self.act(x + residual)


class ScaleSpecificEvidencePooling(nn.Module):
    def __init__(self, channels, embed_dim=256):
        super().__init__()
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)
        self.projection = nn.Linear(channels, embed_dim)

    def forward(self, x):
        attention = F.softmax(self.attention(x).flatten(2), dim=-1)
        tokens = x.flatten(2).transpose(1, 2)
        pooled = torch.bmm(attention, tokens).squeeze(1)
        return self.projection(pooled)


class CrossScaleMutualAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_layers=3, num_scales=4, tokens_per_scale=144):
        super().__init__()
        self.num_scales = num_scales
        self.tokens_per_scale = tokens_per_scale
        self.scale_embeddings = nn.Parameter(torch.randn(1, num_scales, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens, scale_tokens):
        batch_size = patch_tokens.size(0)
        scale_embeddings = self.scale_embeddings.repeat_interleave(self.tokens_per_scale, dim=1)
        patch_tokens = patch_tokens + scale_embeddings.expand(batch_size, -1, -1)
        scale_tokens = scale_tokens + self.scale_embeddings.expand(batch_size, -1, -1)
        tokens = torch.cat([patch_tokens, scale_tokens], dim=1)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens[:, -self.num_scales:, :]


class OrdinalScaleAttentiveHead(nn.Module):
    def __init__(self, embed_dim=256, num_thresholds=4, num_scales=4):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.num_scales = num_scales
        self.scale_weights = nn.Parameter(torch.zeros(num_thresholds, num_scales))
        self.classifiers = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_thresholds)])

    def forward(self, scale_tokens):
        logits = []
        attributions = []
        for k in range(self.num_thresholds):
            a_k = F.softmax(self.scale_weights[k], dim=0)
            u_k = torch.sum(scale_tokens * a_k.view(1, self.num_scales, 1), dim=1)
            logit_k = self.classifiers[k](u_k).squeeze(-1)
            logits.append(logit_k)
            attributions.append(a_k)
        return torch.stack(logits, dim=1), torch.stack(attributions, dim=0)


class DRClassificationModel(nn.Module):
    def __init__(self, input_size=384, embed_dim=256, pretrained_backbone: bool = False):
        super().__init__()
        # pretrained=False: we load our own fine-tuned weights right after
        # construction, so there's no need to fetch ImageNet weights from the
        # timm hub at deploy time (avoids an extra network dependency on the
        # server). The training notebook used pretrained=True as a starting
        # point before fine-tuning — irrelevant once we load the checkpoint.
        self.channels = [96, 192, 384, 768]
        self.embed_dim = embed_dim
        self.num_scales = 4
        self.tokens_per_scale = 144
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained_backbone,
            features_only=True,
            img_size=input_size,
            out_indices=(0, 1, 2, 3),
        )
        self.adapters = nn.ModuleList([BottleneckResidualAdapter(c) for c in self.channels])
        self.evidence_pooling = nn.ModuleList([ScaleSpecificEvidencePooling(c, embed_dim) for c in self.channels])
        self.patch_projections = nn.ModuleList([nn.Linear(c, embed_dim) for c in self.channels])
        self.cross_scale_attention = CrossScaleMutualAttention(
            embed_dim=embed_dim, num_heads=8, num_layers=3, num_scales=4, tokens_per_scale=144
        )
        self.ordinal_head = OrdinalScaleAttentiveHead(embed_dim=embed_dim, num_thresholds=4, num_scales=4)
        self.referable_head = nn.Linear(embed_dim, 1)

    def _to_nchw(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D feature tensor, got {x.shape}")
        if x.shape[-1] in self.channels and x.shape[1] == x.shape[2]:
            return x.permute(0, 3, 1, 2).contiguous()
        if x.shape[1] in self.channels and x.shape[2] == x.shape[3]:
            return x
        raise ValueError(f"Unexpected Swin feature shape: {x.shape}")

    def forward(self, x):
        """Exact copy of the notebook's DRClassificationModel.forward()."""
        features = self.backbone(x)
        adapted_features = []
        evidence_vectors = []
        patch_token_groups = []

        for feature, adapter, pooling, projection in zip(
            features, self.adapters, self.evidence_pooling, self.patch_projections
        ):
            feature = self._to_nchw(feature)
            adapted = adapter(feature)
            adapted_features.append(adapted)
            evidence_vectors.append(pooling(adapted))
            pooled = F.adaptive_avg_pool2d(adapted, (12, 12))
            tokens = pooled.flatten(2).transpose(1, 2)
            patch_token_groups.append(projection(tokens))

        patch_tokens = torch.cat(patch_token_groups, dim=1)
        scale_tokens = torch.stack(evidence_vectors, dim=1)

        refined_scale_tokens = self.cross_scale_attention(patch_tokens, scale_tokens)
        coral_logits, scale_attribution = self.ordinal_head(refined_scale_tokens)
        mean_scale = refined_scale_tokens.mean(dim=1)
        referable_logit = self.referable_head(mean_scale).squeeze(-1)

        return {
            "coral_logits": coral_logits,
            "referable_logit": referable_logit,
            "scale_attribution": scale_attribution,
            "scale_tokens": refined_scale_tokens,
            "patch_tokens": patch_tokens,
            "features": adapted_features,
        }


# ============================================================================
# PREPROCESSING — copied from visionx-classification2.ipynb (cells 5 & 10)
# ============================================================================


def preprocess_image(img, size=384, tol=7, sigma_ratio=30, mask_ratio=0.9, clahe_clip=2.0, clahe_grid=8):
    """Ben Graham-style enhancement + CLAHE, exactly as used for training."""
    if img is None:
        raise ValueError("Input image is None.")
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with 3 channels.")

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > tol)

    if len(ys) == 0:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = img.shape[:2]
    s = max(h, w)
    sq = np.zeros((s, s, 3), dtype=img.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = img

    img = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)

    resized_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    fg = (resized_gray > tol).astype(np.float32)
    fg = cv2.erode(fg, np.ones((7, 7), np.uint8))

    circ = np.zeros((size, size), dtype=np.float32)
    cv2.circle(circ, (size // 2, size // 2), int(size / 2 * mask_ratio), 1, -1)
    mask = fg * circ

    sigma = size / sigma_ratio
    num = cv2.GaussianBlur(img * mask[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(mask, (0, 0), sigma)[..., None] + 1e-6
    blur = num / den

    out = 4 * (img - blur) + 128
    out = np.clip(out, 0, 255)
    out = out * mask[..., None] + 128 * (1 - mask[..., None])
    out = out.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    out[..., 1] = clahe.apply(out[..., 1])

    return out


def to_model_tensor(processed_rgb_uint8, mean, std, device):
    """Equivalent of the notebook's val_transform (ToTensor + Normalize)."""
    tensor = torch.from_numpy(processed_rgb_uint8).float().permute(2, 0, 1) / 255.0
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean_t) / std_t
    return tensor.unsqueeze(0).to(device)


def assess_quality(image_bgr):
    """Lightweight, non-clinical heuristic for obviously unusable photos
    (blur / exposure / incomplete field of view). This is NOT part of the
    trained model — it's a fast pre-check so a bad photo can be flagged
    before it's spent on the network."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    fov_ratio = float((gray > 7).sum()) / gray.size

    sharpness_score = min(100.0, laplacian_var / 8.0)
    brightness_score = 100.0 - min(100.0, abs(brightness - 120.0) / 120.0 * 100.0)
    fov_score = min(100.0, fov_ratio * 140.0)
    overall = round(0.5 * sharpness_score + 0.25 * brightness_score + 0.25 * fov_score)

    issues = []
    if laplacian_var < 60:
        issues.append("Image appears blurred")
    if brightness < 40:
        issues.append("Image is very dark")
    if brightness > 210:
        issues.append("Image is overexposed")
    if fov_ratio < 0.35:
        issues.append("Retinal field of view looks incomplete")

    category = "poor" if overall < 55 else ("usable" if overall < 78 else "good")
    return {"score": int(overall), "category": category, "issues": issues}


def coral_cumulative_to_class_probs(cum_probs: torch.Tensor):
    """cum_probs: (4,) = P(grade>=1..4), from sigmoid(coral_logits) — this is
    exactly how the notebook decodes a grade (see the evaluation cells:
    `predictions = (sigmoid(coral_logits) >= 0.5).sum(dim=1)`). There is no
    separate 5-way softmax head in this architecture, so we derive a 5-way
    class distribution from the ordinal cumulative probabilities:
        P(grade==0) = 1 - P(grade>=1)
        P(grade==k) = P(grade>=k) - P(grade>=k+1)   for k = 1..3
        P(grade==4) = P(grade>=4)
    """
    ones = torch.ones(1, device=cum_probs.device, dtype=cum_probs.dtype)
    probs_ge = torch.cat([ones, cum_probs])  # P(grade>=0..4), length 5
    diffs = probs_ge[:-1] - probs_ge[1:]  # P(grade==0..3)
    class_probs = torch.cat([diffs, probs_ge[-1:]])  # + P(grade==4)
    class_probs = torch.clamp(class_probs, min=0)
    total = class_probs.sum()
    if total > 0:
        class_probs = class_probs / total
    return class_probs.detach().cpu().numpy().tolist()


# ============================================================================
# EXPLAINABILITY — Grad-CAM + real multi-scale evidence attention
# ============================================================================


def resolve_module(root, dotted_path: str):
    """Resolves strings like 'backbone.layers_3.blocks[-1]' into the actual
    submodule, straight from dr_model_config.pth's 'gradcam_target'."""
    obj = root
    for part in dotted_path.split("."):
        m = re.match(r"^(\w+)\[(-?\d+)\]$", part)
        if m:
            obj = getattr(obj, m.group(1))[int(m.group(2))]
        else:
            obj = getattr(obj, part)
    return obj


def to_spatial_nchw(t: torch.Tensor, known_channels=(96, 192, 384, 768)):
    """Normalises a Swin activation/gradient tensor to (B,C,H,W), regardless
    of whether timm returned it as (B,H,W,C), (B,C,H,W) or (B,L,C)."""
    if t.dim() == 4:
        if t.shape[1] in known_channels:
            return t
        if t.shape[-1] in known_channels:
            return t.permute(0, 3, 1, 2).contiguous()
        return t
    if t.dim() == 3:
        b, l, c = t.shape
        side = int(round(l ** 0.5))
        if side * side != l:
            raise ValueError(f"Cannot reshape {l} tokens into a square grid")
        return t.transpose(1, 2).reshape(b, c, side, side)
    raise ValueError(f"Unexpected activation shape for Grad-CAM: {tuple(t.shape)}")


class GradCAM:
    """Standard Grad-CAM over a single target layer (here:
    backbone.layers_3.blocks[-1], matching the notebook's own Grad-CAM demo
    and dr_model_config.pth's 'gradcam_target')."""

    def __init__(self, model: nn.Module, target_module: nn.Module):
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        target_module.register_forward_hook(self._forward_hook)
        target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def run(self, x: torch.Tensor, target_fn):
        """Single forward+backward pass. Returns (cam[H,W] in [0,1], outputs dict)."""
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(x)
        score = target_fn(outputs)
        score.backward()

        acts = to_spatial_nchw(self.activations.detach())
        grads = to_spatial_nchw(self.gradients.detach())
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, outputs


class ScaleAttentionRecorder:
    """Captures the real, per-scale spatial attention that each
    `ScaleSpecificEvidencePooling` module computes internally (one map per
    backbone stage: 96x96, 48x48, 24x24, 12x12 for a 384px input). These are
    genuine trained-model activations, captured via a forward hook on the
    same `attention` conv layer the model itself pools evidence with — not
    synthesized or randomly generated."""

    def __init__(self, model: "DRClassificationModel"):
        self.raw_logits = [None] * len(model.evidence_pooling)
        self.hooks = []
        for i, pooling in enumerate(model.evidence_pooling):
            self.hooks.append(pooling.attention.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, idx):
        def _hook(module, inp, out):
            self.raw_logits[idx] = out.detach()
        return _hook

    def remove(self):
        for h in self.hooks:
            h.remove()

    def get_attention_maps(self):
        """Reproduces ScaleSpecificEvidencePooling.forward()'s own softmax on
        the captured pre-softmax logits. Returns a list of (1,1,H,W) tensors,
        one per backbone scale, each summing to 1 over its spatial positions."""
        maps = []
        for logits in self.raw_logits:
            if logits is None:
                maps.append(None)
                continue
            b, c, h, w = logits.shape
            att = F.softmax(logits.flatten(2), dim=-1).view(b, c, h, w)
            maps.append(att)
        return maps


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    peak = arr.max()
    if peak > 0:
        arr = arr / peak
    return arr


def render_scale_evidence_overlay(processed_rgb_uint8, scale_attention_maps, scale_weights, image_size):
    """Combines the model's real per-scale evidence-pooling attention maps
    into a single overlay, weighted by the model's own *learned*
    `OrdinalScaleAttentiveHead.scale_weights` (how much each backbone scale
    contributes to the ordinal grade decision, averaged across the 4 CORAL
    thresholds). Every input here comes from the trained checkpoint — no
    values are invented.

    Returns None if any scale's attention map wasn't captured (e.g. hooks
    failed to fire), so the caller can gracefully omit this overlay rather
    than show something misleading.
    """
    if any(m is None for m in scale_attention_maps):
        return None

    composite = np.zeros((image_size, image_size), dtype=np.float32)
    for weight, att_map in zip(scale_weights, scale_attention_maps):
        att_np = att_map[0, 0].cpu().numpy()
        att_np = minmax_normalize(att_np)  # each scale's raw softmax values have very
        # different magnitudes (a 96x96 map sums its mass over far more pixels than a
        # 12x12 map), so normalise per-scale before combining for a fair visualisation.
        att_resized = cv2.resize(att_np, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        composite += float(weight) * att_resized

    composite = minmax_normalize(composite)
    return render_heatmap_overlay(processed_rgb_uint8, composite, colormap=cv2.COLORMAP_TURBO, alpha=0.4)


def render_heatmap_overlay(processed_rgb_uint8, cam_2d, colormap=cv2.COLORMAP_JET, alpha=0.4):
    h, w = processed_rgb_uint8.shape[:2]
    cam_resized = cv2.resize(cam_2d.astype(np.float32), (w, h))
    cam_resized -= cam_resized.min()
    if cam_resized.max() > 0:
        cam_resized /= cam_resized.max()
    heat_uint8 = np.uint8(255 * cam_resized)
    colored_bgr = cv2.applyColorMap(heat_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(processed_rgb_uint8, 1 - alpha, colored_rgb, alpha, 0)
    return encode_png_base64(overlay)


def encode_png_base64(img_rgb_uint8):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode image")
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


# ============================================================================
# MODEL LOADING (once, at import time)
# ============================================================================

GRADE_LABELS = {
    0: "No DR", 1: "Mild DR", 2: "Moderate DR", 3: "Severe DR", 4: "Proliferative DR",
}


def load_config():
    path = os.path.join(MODEL_DIR, CONFIG_FILENAME)
    cfg = torch.load(path, map_location="cpu", weights_only=False)
    return cfg


def load_model(cfg):
    input_size = int(cfg.get("input_size", 384)) if isinstance(cfg, dict) else 384
    embed_dim = int(cfg.get("embedding_dimension", 256)) if isinstance(cfg, dict) else 256
    model = DRClassificationModel(input_size=input_size, embed_dim=embed_dim, pretrained_backbone=False)
    weights_path = os.path.join(MODEL_DIR, WEIGHTS_FILENAME)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)
    log.info("Loaded weights from %s onto %s (missing=%s unexpected=%s)", weights_path, DEVICE, missing, unexpected)
    return model


log.info("Loading RetinaCare DR screening model — this happens once at startup")
CONFIG = load_config()
MODEL = load_model(CONFIG)

GRAD_CAM_TARGET = resolve_module(MODEL, CONFIG.get("gradcam_target", "backbone.layers_3.blocks[-1]"))
GRAD_CAM = GradCAM(MODEL, GRAD_CAM_TARGET)
SCALE_ATTENTION_RECORDER = ScaleAttentionRecorder(MODEL)

# The ordinal head's per-scale importance is a fixed, learned parameter — it
# doesn't depend on the input image — so it's computed once here rather than
# on every request. Shape: (num_thresholds=4, num_scales=4), each row sums to 1.
with torch.no_grad():
    SCALE_ATTRIBUTION = F.softmax(MODEL.ordinal_head.scale_weights.detach(), dim=1)
    SCALE_IMPORTANCE = SCALE_ATTRIBUTION.mean(dim=0).cpu().numpy().tolist()  # (4,), averaged across thresholds

PREPROC_CFG = CONFIG.get("preprocessing", {}) if isinstance(CONFIG, dict) else {}
NORM_MEAN = PREPROC_CFG.get("normalization_mean", [0.485, 0.456, 0.406])
NORM_STD = PREPROC_CFG.get("normalization_std", [0.229, 0.224, 0.225])
IMAGE_SIZE = PREPROC_CFG.get("resize", 384)
REFERABLE_THRESHOLD_GRADE = CONFIG.get("referable_threshold", 2) if isinstance(CONFIG, dict) else 2

# Every inference request goes through this lock. The model, Grad-CAM hooks
# and scale-attention hooks are shared, mutable state on a single MODEL
# instance per worker process; under a threaded gunicorn worker
# (--threads 4) two concurrent requests could otherwise interleave their
# forward/backward passes and read back the wrong image's activations. The
# lock trades a small amount of concurrency for correctness, which matters
# far more for a medical screening result.
INFERENCE_LOCK = threading.Lock()

log.info(
    "Model ready. classes=%s referable_threshold=%s scale_importance=%s",
    CONFIG.get("classes") if isinstance(CONFIG, dict) else None,
    REFERABLE_THRESHOLD_GRADE,
    [round(w, 3) for w in SCALE_IMPORTANCE],
)


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================


def run_inference(image_bgr):
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

    quality = assess_quality(image_bgr)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    processed = preprocess_image(image_rgb, size=IMAGE_SIZE)
    tensor = to_model_tensor(processed, NORM_MEAN, NORM_STD, DEVICE)

    with INFERENCE_LOCK:
        with torch.enable_grad():
            cam_map, outputs = GRAD_CAM.run(tensor, target_fn=lambda o: o["referable_logit"].sum())
        scale_attention_maps = SCALE_ATTENTION_RECORDER.get_attention_maps()

        coral_logits = outputs["coral_logits"].detach()
        referable_logit = outputs["referable_logit"].detach()

    cum_probs = torch.sigmoid(coral_logits)[0]  # (4,) = P(grade>=1..4)
    class_probs = coral_cumulative_to_class_probs(cum_probs)
    predicted_grade = int((cum_probs >= 0.5).sum().item())
    referable_probability = float(torch.sigmoid(referable_logit)[0].item())
    referable = referable_probability >= 0.5

    gradcam_b64 = render_heatmap_overlay(processed, cam_map, colormap=cv2.COLORMAP_JET)

    evidence_b64 = None
    try:
        evidence_b64 = render_scale_evidence_overlay(processed, scale_attention_maps, SCALE_IMPORTANCE, IMAGE_SIZE)
    except Exception:
        log.exception("Scale-attentive evidence overlay failed — continuing without it")

    return {
        "grade": predicted_grade,
        "grade_label": GRADE_LABELS.get(predicted_grade, str(predicted_grade)),
        "referable": bool(referable),
        "referable_probability": round(referable_probability, 4),
        "class_probabilities": [round(p, 4) for p in class_probs],
        "confidence": round(class_probs[predicted_grade], 4),
        "quality": quality,
        "gradcam_overlay_base64": gradcam_b64,
        # Real multi-scale evidence-attention overlay from the trained model
        # (see render_scale_evidence_overlay's docstring) — sent under this
        # key so the existing frontend, which already treats it as a
        # "Visual Evidence" attention overlay rather than a segmentation
        # mask, works without changes.
        "ggsa_overlay_base64": evidence_b64,
        "processed_image_base64": encode_png_base64(processed),
        "model": {
            "backbone": CONFIG.get("backbone") if isinstance(CONFIG, dict) else "swin_tiny_patch4_window7_224",
            "referable_definition": CONFIG.get("referable_definition") if isinstance(CONFIG, dict) else "Grade >= 2",
        },
    }


# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify(status="ok", model_loaded=MODEL is not None, device=str(DEVICE))


@app.route("/api/screen", methods=["POST"])
def screen():
    if "image" not in request.files:
        return jsonify(error="No file provided under form field 'image'"), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify(error="Empty filename"), 400
    if not allowed_file(file.filename):
        return jsonify(error="Unsupported file type — use PNG or JPG"), 400

    try:
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if image_bgr is None:
            return jsonify(error="Could not decode image"), 400
    except Exception:
        log.exception("Failed to decode uploaded image")
        return jsonify(error="Could not decode image"), 400

    try:
        result = run_inference(image_bgr)
    except Exception:
        log.exception("Inference failed")
        return jsonify(error="Inference failed — check server logs"), 500

    return jsonify(result)


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error=f"Image too large — limit is {MAX_UPLOAD_MB}MB"), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
