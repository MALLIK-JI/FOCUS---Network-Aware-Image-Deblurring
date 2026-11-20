
import os
import sys
import time
import gc
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
import numpy as np
import cv2
import torch.nn.functional as F
import psutil

from basicsr.models.archs.restormer_arch import Restormer
from lightweight.model import LightweightDeblurCNN

INPUT_DIR = "./Datasets/test/GoPro/input"

RESTORMER_CHECKPOINT = "./pretrained_models/motion_deblurring.pth"

LIGHTWEIGHT_CHECKPOINT = (
    "./lightweight/checkpoints_continued/"
    "lightweight_continued_best.pth"
)

DEVICE = torch.device("cpu")

# Limit CPU threads for a more controlled comparison
torch.set_num_threads(4)


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


def prepare_image(path):

    image = cv2.imread(path)

    if image is None:
        raise RuntimeError(f"Could not read {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(image).permute(2, 0, 1)
    return tensor.unsqueeze(0)


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


def main():

    print("Benchmark device: CPU")
    print("CPU threads:", torch.get_num_threads())

    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

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

    print("\n========== CPU BENCHMARK ==========")

    for name, loader, restore_function, checkpoint_path in models:

        print("\nLoading:", name)

        gc.collect()

        ram_before = process.memory_info().rss / (1024 ** 2)

        model = loader()

        ram_after = process.memory_info().rss / (1024 ** 2)

        timings = []

        with torch.inference_mode():

            # Warm-up
            for tensor in images:
                output = restore_function(model, tensor)

            del output

            # Actual benchmark
            for tensor in images:

                image_timings = []

                for _ in range(5):

                    start = time.perf_counter()

                    output = restore_function(model, tensor)

                    elapsed = time.perf_counter() - start

                    image_timings.append(elapsed)

                    del output

                timings.append(np.mean(image_timings))

        average_time = np.mean(timings)

        print("\nModel:", name)
        print("Checkpoint size:", f"{os.path.getsize(checkpoint_path) / (1024**2):.2f} MB")
        print("Average CPU inference time:", f"{average_time:.4f} seconds/image")
        print("RAM before loading:", f"{ram_before:.2f} MB")
        print("RAM after loading:", f"{ram_after:.2f} MB")
        print("RAM change:", f"{ram_after - ram_before:.2f} MB")

        del model
        gc.collect()

    print("\nCPU benchmark completed!")


if __name__ == "__main__":
    main()
