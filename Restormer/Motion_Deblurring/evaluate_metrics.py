
import os
import numpy as np
from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

pairs = [
    (
        "image1",
        "Datasets/test/GoPro/input/image1.jpg",
        "Datasets/test/GoPro/ground_truth/image1.jpg",
        "results_original/image1.png",
        "results_custom_loss/image1.png"
    ),
    (
        "image2",
        "Datasets/test/GoPro/input/image2.jpg",
        "Datasets/test/GoPro/ground_truth/image2.jpg",
        "results_original/image2.png",
        "results_custom_loss/image2.png"
    )
]

def calculate_metrics(image_path, target_path):
    pred = np.asarray(
        Image.open(image_path).convert("RGB"),
        dtype=np.float32
    )

    target = np.asarray(
        Image.open(target_path).convert("RGB"),
        dtype=np.float32
    )

    mse = np.mean((pred - target) ** 2)

    psnr = peak_signal_noise_ratio(
        target, pred, data_range=255
    )

    ssim = structural_similarity(
        target, pred,
        channel_axis=2,
        data_range=255
    )

    return mse, psnr, ssim


for name, input_path, gt_path, original_path, custom_path in pairs:

    input_metrics = calculate_metrics(input_path, gt_path)
    original_metrics = calculate_metrics(original_path, gt_path)
    custom_metrics = calculate_metrics(custom_path, gt_path)

    print(f"\n{name.upper()}")
    print("=" * 75)
    print(f"{'Metric':<12}{'Blurred Input':<18}{'Original Model':<18}{'Custom Loss':<18}")

    for index, metric in enumerate(["MSE", "PSNR", "SSIM"]):
        print(
            f"{metric:<12}"
            f"{input_metrics[index]:<18.4f}"
            f"{original_metrics[index]:<18.4f}"
            f"{custom_metrics[index]:<18.4f}"
        )
