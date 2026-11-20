
from pathlib import Path
import csv
import numpy as np
from skimage import io, img_as_ubyte
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)

# --------------------------------------------------
# 1. DIRECTORY PATHS
# --------------------------------------------------

BASE = Path(__file__).resolve().parent

GT_DIR = BASE / "test_input" / "test" / "GoPro" / "ground_truth"
BLURRED_DIR = BASE / "test_input" / "test" / "GoPro" / "blurred"

ORIGINAL_DIR = BASE / "results_original" / "GoPro"
FINETUNED_DIR = BASE / "results_finetuned" / "GoPro"

OUTPUT_CSV = BASE / "final_metrics_comparison.csv"


# --------------------------------------------------
# 2. LOAD AND NORMALIZE IMAGES
# --------------------------------------------------

def load_image(path):
    image = io.imread(path)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Unsupported image format: {path}")

    return img_as_ubyte(image)


# --------------------------------------------------
# 3. FIND IMAGE WITH MATCHING NAME
# --------------------------------------------------

def find_image(folder, name):
    for extension in [".png", ".jpg", ".jpeg"]:
        image_path = folder / f"{name}{extension}"

        if image_path.exists():
            return image_path

    return None


# --------------------------------------------------
# 4. CALCULATE IMAGE QUALITY METRICS
# --------------------------------------------------

def calculate_metrics(ground_truth, predicted):

    if ground_truth.shape != predicted.shape:
        raise ValueError(
            f"Image dimensions do not match: "
            f"{ground_truth.shape} vs {predicted.shape}"
        )

    mse = mean_squared_error(ground_truth, predicted)

    psnr = peak_signal_noise_ratio(
        ground_truth,
        predicted,
        data_range=255
    )

    ssim = structural_similarity(
        ground_truth,
        predicted,
        channel_axis=-1,
        data_range=255
    )

    return mse, psnr, ssim


# --------------------------------------------------
# 5. CHECK REQUIRED DIRECTORIES
# --------------------------------------------------

required_directories = [
    GT_DIR,
    BLURRED_DIR,
    ORIGINAL_DIR,
    FINETUNED_DIR
]

for directory in required_directories:
    if not directory.exists():
        raise FileNotFoundError(
            f"Required directory not found: {directory}"
        )


# --------------------------------------------------
# 6. COMPARE ALL THREE RESULTS
# --------------------------------------------------

rows = []

image_extensions = [".jpg", ".jpeg", ".png"]

ground_truth_files = sorted(
    path for path in GT_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in image_extensions
)

if not ground_truth_files:
    raise RuntimeError("No ground-truth images found.")


for gt_path in ground_truth_files:

    image_name = gt_path.stem

    blurred_path = find_image(BLURRED_DIR, image_name)
    original_path = find_image(ORIGINAL_DIR, image_name)
    finetuned_path = find_image(FINETUNED_DIR, image_name)

    image_paths = {
        "Blurred": blurred_path,
        "Original_Restormer": original_path,
        "FineTuned_Restormer": finetuned_path
    }

    missing = [
        model_name
        for model_name, path in image_paths.items()
        if path is None
    ]

    if missing:
        print(
            f"Skipping {image_name}: missing {', '.join(missing)}"
        )
        continue

    ground_truth = load_image(gt_path)

    print("\n" + "=" * 65)
    print(f"IMAGE: {image_name}")
    print("=" * 65)

    for model_name, image_path in image_paths.items():

        predicted = load_image(image_path)

        mse, psnr, ssim = calculate_metrics(
            ground_truth,
            predicted
        )

        rows.append({
            "Image": image_name,
            "Model": model_name,
            "MSE": mse,
            "PSNR": psnr,
            "SSIM": ssim
        })

        print(f"\n{model_name}")
        print(f"MSE  : {mse:.4f}")
        print(f"PSNR : {psnr:.4f} dB")
        print(f"SSIM : {ssim:.4f}")


# --------------------------------------------------
# 7. SAVE RESULTS TO CSV
# --------------------------------------------------

if rows:

    fieldnames = [
        "Image",
        "Model",
        "MSE",
        "PSNR",
        "SSIM"
    ]

    with open(OUTPUT_CSV, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON COMPLETED")
    print("=" * 65)

    print(f"Total comparisons: {len(rows)}")
    print(f"CSV saved at: {OUTPUT_CSV}")

else:
    print("\nNo comparisons completed.")
    print("Check the image names and directory paths.")
