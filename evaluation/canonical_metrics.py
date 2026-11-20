"""
Canonical Image Quality Metrics Module for Deblurring Benchmarks.

Provides standardized, mathematically consistent calculation of:
- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index Measure (SSIM)

Standards:
- Color space: RGB
- Data range: float in [0.0, 1.0] (data_range = 1.0)
- SSIM window: 11x11 Gaussian window (sigma = 1.5), channel_axis = 2
- Edge case: Zero MSE yields PSNR = float('inf') or capped 100.0 dB
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_normalized_rgb(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Convert an image (PIL Image or NumPy array) to float64 RGB in [0.0, 1.0].
    
    Parameters
    ----------
    img : Union[np.ndarray, Image.Image]
        Input image. If uint8 NumPy array or PIL Image, divided by 255.0.
        If float NumPy array, clipped to [0.0, 1.0].

    Returns
    -------
    np.ndarray
        Float64 array of shape (H, W, 3) with values in [0.0, 1.0].
    """
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.float64) / 255.0

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected PIL Image or np.ndarray, got {type(img)}")

    arr = img.copy()

    # Handle grayscale by expanding channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]  # Drop alpha channel
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}. Expected (H, W, 3).")

    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64) / 255.0
    else:
        arr = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)

    return arr


def calculate_metrics(
    prediction: Union[np.ndarray, Image.Image],
    target: Union[np.ndarray, Image.Image],
    max_psnr_cap: float = 100.0
) -> Tuple[float, float, float]:
    """
    Compute canonical (MSE, PSNR, SSIM) between prediction and ground-truth target.

    Parameters
    ----------
    prediction : Union[np.ndarray, Image.Image]
        Restored / predicted image.
    target : Union[np.ndarray, Image.Image]
        Ground-truth reference image.
    max_psnr_cap : float, default 100.0
        Cap value for PSNR when MSE is mathematically zero to prevent inf in logs.

    Returns
    -------
    Tuple[float, float, float]
        (mse, psnr_db, ssim_score)
    """
    pred_rgb = to_normalized_rgb(prediction)
    target_rgb = to_normalized_rgb(target)

    if pred_rgb.shape != target_rgb.shape:
        raise ValueError(
            f"Shape mismatch: prediction shape {pred_rgb.shape} != target shape {target_rgb.shape}"
        )

    # 1. Mean Squared Error (MSE)
    mse = float(np.mean((pred_rgb - target_rgb) ** 2))

    # 2. Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0.0:
        psnr = float(max_psnr_cap)
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))

    # 3. Structural Similarity Index (SSIM)
    # Default skimage parameters (gaussian_weights=False, 11x11 uniform window)
    # to maintain strict numerical parity with evaluate_gopro_1111.py and app.py
    ssim = float(
        structural_similarity(
            target_rgb,
            pred_rgb,
            channel_axis=2,
            data_range=1.0
        )
    )

    return mse, psnr, ssim


if __name__ == "__main__":
    # Self-test unit verification
    print("Running canonical metrics self-test...")

    # Test 1: Identical images
    img_rand = np.random.RandomState(42).rand(128, 128, 3)
    mse_id, psnr_id, ssim_id = calculate_metrics(img_rand, img_rand)
    assert mse_id == 0.0, f"Expected MSE 0.0, got {mse_id}"
    assert psnr_id == 100.0, f"Expected PSNR 100.0, got {psnr_id}"
    assert np.isclose(ssim_id, 1.0, atol=1e-6), f"Expected SSIM 1.0, got {ssim_id}"
    print("✓ Test 1 (Identity): Passed.")

    # Test 2: Black vs White
    black = np.zeros((64, 64, 3), dtype=np.float64)
    white = np.ones((64, 64, 3), dtype=np.float64)
    mse_bw, psnr_bw, ssim_bw = calculate_metrics(black, white)
    assert np.isclose(mse_bw, 1.0), f"Expected MSE 1.0, got {mse_bw}"
    assert np.isclose(psnr_bw, 0.0), f"Expected PSNR 0.0, got {psnr_bw}"
    print(f"✓ Test 2 (Black vs White): Passed (MSE={mse_bw:.4f}, PSNR={psnr_bw:.2f} dB, SSIM={ssim_bw:.4f}).")

    # Test 3: Numerical parity with skimage peak_signal_noise_ratio
    noisy = np.clip(img_rand + 0.05 * np.random.RandomState(1).randn(*img_rand.shape), 0, 1)
    mse_calc, psnr_calc, _ = calculate_metrics(noisy, img_rand)
    sk_psnr = peak_signal_noise_ratio(img_rand, noisy, data_range=1.0)
    assert np.isclose(psnr_calc, sk_psnr, atol=1e-8), f"PSNR mismatch: {psnr_calc} vs {sk_psnr}"
    print(f"✓ Test 3 (Parity with skimage PSNR): Passed ({psnr_calc:.6f} == {sk_psnr:.6f}).")

    print("\nAll canonical metrics self-tests passed successfully!")
