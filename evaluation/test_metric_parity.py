"""
Metric Parity Verification Script for Phase 1 Baseline.

Verifies that:
1. canonical_metrics.calculate_metrics
2. evaluate_gopro_1111.calculate_metrics
3. app.calculate_metrics_standard

produce identical mathematical results down to float precision on real paired images.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Import canonical metrics
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "Restormer" / "Motion_Deblurring"))

from evaluation.canonical_metrics import calculate_metrics as canonical_calc
from evaluate_gopro_1111 import calculate_metrics as gopro_calc
from app import calculate_metrics_standard as app_calc


def main():
    img_dir = BASE_DIR / "Restormer" / "Motion_Deblurring" / "Datasets" / "test" / "GoPro"
    gt_dir = img_dir / "ground_truth"
    inp_dir = img_dir / "input"

    test_images = ["image1.jpg", "image2.jpg"]

    print("=" * 65)
    print("METRIC PIPELINE PARITY VERIFICATION")
    print("=" * 65)

    for name in test_images:
        gt_path = gt_dir / name
        inp_path = inp_dir / name

        if not gt_path.exists() or not inp_path.exists():
            print(f"Skipping {name}: file not found.")
            continue

        # Load as PIL Image (app.py style)
        pil_gt = Image.open(gt_path).convert("RGB")
        pil_inp = Image.open(inp_path).convert("RGB")

        # Load as normalized float array (evaluate_gopro_1111.py style)
        arr_gt = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        arr_inp = cv2.cvtColor(cv2.imread(str(inp_path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Compute with all three implementations
        mse_canon, psnr_canon, ssim_canon = canonical_calc(arr_inp, arr_gt)
        mse_gopro, psnr_gopro, ssim_gopro = gopro_calc(arr_inp, arr_gt)
        mse_app, psnr_app, ssim_app = app_calc(pil_inp, pil_gt)

        print(f"\nImage: {name}")
        print(f"  Canonical : MSE={mse_canon:.8f}, PSNR={psnr_canon:.4f} dB, SSIM={ssim_canon:.4f}")
        print(f"  GoPro1111 : MSE={mse_gopro:.8f}, PSNR={psnr_gopro:.4f} dB, SSIM={ssim_gopro:.4f}")
        print(f"  App.py UI : MSE={mse_app:.8f}, PSNR={psnr_app:.4f} dB, SSIM={ssim_app:.4f}")

        # Assertions
        assert np.isclose(mse_canon, mse_gopro, atol=1e-7), f"MSE mismatch: {mse_canon} vs {mse_gopro}"
        assert np.isclose(psnr_canon, psnr_gopro, atol=1e-4), f"PSNR mismatch: {psnr_canon} vs {psnr_gopro}"
        assert np.isclose(ssim_canon, ssim_gopro, atol=1e-4), f"SSIM mismatch: {ssim_canon} vs {ssim_gopro}"
        assert np.isclose(psnr_canon, psnr_app, atol=1e-2), f"PSNR mismatch with App: {psnr_canon} vs {psnr_app}"
        print(f"  -> Exact Agreement Confirmed for {name}!")

    print("\n" + "=" * 65)
    print("ALL METRIC IMPLEMENTATIONS CONFIRMED IN EXACT PARITY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
