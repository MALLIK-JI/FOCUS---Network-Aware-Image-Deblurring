
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import gc
import torch
import yaml
import numpy as np
import cv2
import torch.nn.functional as F
import psutil

from basicsr.models.archs.restormer_arch import Restormer
from lightweight.model import LightweightDeblurCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR = "./Datasets/test/GoPro/input"

RESTORMER_CHECKPOINT = "./pretrained_models/motion_deblurring.pth"

LIGHTWEIGHT_CHECKPOINT = (
    "./lightweight/checkpoints_continued/"
    "lightweight_continued_best.pth"
)

WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def load_restormer():

    with open("./Options/Deblurring_Restormer.yml", "r") as f:
        config = yaml.safe_load(f)

    network_config = config["network_g"].copy()
    network_config.pop("type")

    model = Restormer(**network_config).to(DEVICE)

    checkpoint = torch.load(
        RESTORMER_CHECKPOINT,
        map_location=DEVICE
    )

    if "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


def load_lightweight():

    model = LightweightDeblurCNN().to(DEVICE)

    checkpoint = torch.load(
        LIGHTWEIGHT_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


def prepare_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(f"Could not read {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(image).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    return tensor


def restore_restormer(model, tensor):

    height, width = tensor.shape[2:]

    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8

    if pad_h or pad_w:
        tensor = F.pad(
            tensor,
            (0, pad_w, 0, pad_h),
            mode="reflect"
        )

    output = model(tensor)

    return torch.clamp(
        output[:, :, :height, :width],
        0, 1
    )


def restore_lightweight(model, tensor):

    return torch.clamp(model(tensor), 0, 1)


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def checkpoint_size(path):

    return os.path.getsize(path) / (1024 ** 2)


def benchmark_model(model, restore_function, images):

    timings = []

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():

        # Warm-up
        for tensor in images:
            for _ in range(WARMUP_RUNS):
                output = restore_function(model, tensor)

            del output

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        for tensor in images:

            image_timings = []

            for _ in range(BENCHMARK_RUNS):

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()

                output = restore_function(model, tensor)

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start

                image_timings.append(elapsed)

                del output

            timings.append(np.mean(image_timings))

    peak_gpu_memory = None

    if DEVICE.type == "cuda":
        peak_gpu_memory = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

    return np.mean(timings), peak_gpu_memory


def main():

    print("Benchmark device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        raise RuntimeError("No test images found!")

    images = [
        prepare_image(os.path.join(INPUT_DIR, filename))
        for filename in image_files
    ]

    process = psutil.Process(os.getpid())

    models = [
        (
            "Original Restormer",
            load_restormer,
            restore_restormer,
            RESTORMER_CHECKPOINT
        ),
        (
            "Continued Lightweight CNN",
            load_lightweight,
            restore_lightweight,
            LIGHTWEIGHT_CHECKPOINT
        )
    ]

    print("\n========== MODEL BENCHMARK ==========")

    for name, loader, restore_function, checkpoint_path in models:

        print(f"\nLoading: {name}")

        gc.collect()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        memory_before = process.memory_info().rss / (1024 ** 2)

        model = loader()

        memory_after = process.memory_info().rss / (1024 ** 2)

        parameters = count_parameters(model)
        size = checkpoint_size(checkpoint_path)

        inference_time, peak_gpu_memory = benchmark_model(
            model,
            restore_function,
            images
        )

        print("\nModel:", name)
        print(f"Parameters: {parameters:,}")
        print(f"Checkpoint size: {size:.2f} MB")
        print(f"Average inference time: {inference_time:.4f} seconds/image")
        print(f"Process RAM before loading: {memory_before:.2f} MB")
        print(f"Process RAM after loading: {memory_after:.2f} MB")
        print(f"RAM increase during loading: {memory_after - memory_before:.2f} MB")

        if peak_gpu_memory is not None:
            print(f"Peak GPU allocated memory: {peak_gpu_memory:.2f} MB")

        del model
        gc.collect()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
