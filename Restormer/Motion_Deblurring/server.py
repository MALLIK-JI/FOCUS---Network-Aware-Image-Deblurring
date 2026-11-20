"""
FastAPI Server for Network-Aware Image Deblurring.

Supports:
- Original Restormer (GoPro motion deblurring)
- Fine-Tuned Custom-Loss Restormer
- Server health, GPU hardware telemetry, and latency reporting
"""

import io
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

# Setup sys.path to locate basicsr and custom modules
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
PROJECT_ROOT = REPO_DIR.parent

for p in [str(REPO_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from basicsr.models.archs.restormer_arch import Restormer

# ==========================================
# CONFIGURATION & CHECKPOINTS
# ==========================================

CONFIG_PATH = BASE_DIR / "Options" / "Deblurring_Restormer.yml"
ORIGINAL_CHECKPOINT = BASE_DIR / "pretrained_models" / "motion_deblurring.pth"
CUSTOM_CHECKPOINT = (
    REPO_DIR / "experiments" / "Deblurring_Restormer_CustomLoss" / "models" / "net_g_100.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="Network-Aware Image Deblurring Server",
    description="High-performance GPU restoration server for Restormer and classical DIP",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model registry
_models = {}


def load_model_instance(checkpoint_path: Path) -> Restormer:
    """Instantiate and load weights for a Restormer model."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    network_config = config["network_g"].copy()
    network_config.pop("type", None)

    net = Restormer(**network_config).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if "params" in ckpt:
        sd = ckpt["params"]
    elif "params_ema" in ckpt:
        sd = ckpt["params_ema"]
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


def get_model(model_type: str = "original") -> Restormer:
    """Retrieve or lazily cache the requested Restormer variant."""
    key = model_type.lower()
    if key not in _models:
        if key == "original":
            print(f"Loading Original Restormer from {ORIGINAL_CHECKPOINT} on {DEVICE}...")
            _models[key] = load_model_instance(ORIGINAL_CHECKPOINT)
            print("Original Restormer loaded successfully.")
        elif key in ["custom", "custom_loss", "finetuned"]:
            print(f"Loading Custom-Loss Restormer from {CUSTOM_CHECKPOINT} on {DEVICE}...")
            _models[key] = load_model_instance(CUSTOM_CHECKPOINT)
            print("Custom-Loss Restormer loaded successfully.")
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'original' or 'custom_loss'.")
    return _models[key]


# Preload default original model at server startup
try:
    get_model("original")
except Exception as exc:
    print(f"Warning: Could not preload original model at startup: {exc}")


# ==========================================
# HEALTH & TELEMETRY ENDPOINTS
# ==========================================

@app.get("/")
def home():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    return {
        "status": "online",
        "service": "Network-Aware Image Deblurring Server",
        "device": str(DEVICE),
        "gpu_name": gpu_name,
        "available_models": ["original", "custom_loss"]
    }


@app.get("/info")
def server_info():
    vram_used_mb = 0
    vram_total_mb = 0
    if torch.cuda.is_available():
        vram_used_mb = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
        vram_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)

    return {
        "device": str(DEVICE),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "vram_allocated_mb": vram_used_mb,
        "vram_total_mb": vram_total_mb,
        "cached_models": list(_models.keys()),
        "supported_models": ["original", "custom_loss"]
    }


# ==========================================
# INFERENCE PIPELINE & MEMORY-SAFE TILED INFERENCE
# ==========================================

def inference_tiled(model: Restormer, input_tensor: torch.Tensor, tile: int = 480, tile_overlap: int = 32) -> torch.Tensor:
    """
    Execute Restormer tile-by-tile to guarantee zero CUDA Out-Of-Memory errors on any resolution.
    Matches the official Restormer tiling evaluation scheme.
    """
    b, c, h, w = input_tensor.shape
    tile = min(tile, h, w)
    tile = (tile // 8) * 8
    stride = tile - tile_overlap

    h_idx_list = sorted(list(set(list(range(0, h - tile, stride)) + [h - tile])))
    w_idx_list = sorted(list(set(list(range(0, w - tile, stride)) + [w - tile])))

    E = torch.zeros(b, c, h, w, device=input_tensor.device)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_tensor[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
            W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)

    restored = E.div_(W)
    return restored


# ==========================================
# DEBLURRING INFERENCE ENDPOINT
# ==========================================

@app.post("/deblur")
async def deblur_image(
    file: UploadFile = File(...),
    model_type: str = Query("original", description="Model variant: 'original' or 'custom_loss'")
):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported.")

    try:
        model = get_model(model_type)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        np_img = np.array(pil_img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        _, _, height, width = tensor.shape
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8

        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        # Clear GPU cache before running inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            total_pixels = height * width
            # If image exceeds ~0.65 MP (e.g. 1280x720 or 1080p), run tiled inference to prevent OOM on 6GB VRAM
            if total_pixels > 650000 or max(height, width) > 1000:
                print(f"High-resolution image detected ({width}x{height} = {total_pixels:,} px). Executing memory-safe tiled inference...")
                restored = inference_tiled(model, tensor, tile=480, tile_overlap=32)
            else:
                try:
                    restored = model(tensor)
                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"Warning: Direct inference triggered OOM on ({width}x{height}). Seamlessly falling back to tiled inference...")
                    restored = inference_tiled(model, tensor, tile=384, tile_overlap=32)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time

        restored = torch.clamp(restored, 0.0, 1.0)
        restored = restored[:, :, :height, :width]
        restored_np = (
            restored.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        ).round().astype(np.uint8)

        output_img = Image.fromarray(restored_np)
        buf = io.BytesIO()
        output_img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Content-Disposition": 'attachment; filename="restored.png"',
                "X-Inference-Time": f"{inference_time:.4f}",
                "X-Model-Type": model_type,
                "X-Device": str(DEVICE)
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
