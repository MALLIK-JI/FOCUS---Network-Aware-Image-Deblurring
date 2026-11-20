from typing import Tuple, Union
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def to_normalized_rgb(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(img, Image.Image):
        img = img.convert('RGB')
        return np.asarray(img, dtype=np.float64) / 255.0
    if not isinstance(img, np.ndarray):
        raise TypeError(f'Expected PIL Image or np.ndarray, got {type(img)}')
    arr = img.copy()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f'Unsupported image shape: {arr.shape}. Expected (H, W, 3).')
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64) / 255.0
    else:
        arr = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
    return arr

def calculate_metrics(prediction: Union[np.ndarray, Image.Image], target: Union[np.ndarray, Image.Image], max_psnr_cap: float=100.0) -> Tuple[float, float, float]:
    pred_rgb = to_normalized_rgb(prediction)
    target_rgb = to_normalized_rgb(target)
    if pred_rgb.shape != target_rgb.shape:
        raise ValueError(f'Shape mismatch: prediction shape {pred_rgb.shape} != target shape {target_rgb.shape}')
    mse = float(np.mean((pred_rgb - target_rgb) ** 2))
    if mse == 0.0:
        psnr = float(max_psnr_cap)
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))
    ssim = float(structural_similarity(target_rgb, pred_rgb, channel_axis=2, data_range=1.0))
    return (mse, psnr, ssim)
if __name__ == '__main__':
    print('Running canonical metrics self-test...')
    img_rand = np.random.RandomState(42).rand(128, 128, 3)
    (mse_id, psnr_id, ssim_id) = calculate_metrics(img_rand, img_rand)
    assert mse_id == 0.0, f'Expected MSE 0.0, got {mse_id}'
    assert psnr_id == 100.0, f'Expected PSNR 100.0, got {psnr_id}'
    assert np.isclose(ssim_id, 1.0, atol=1e-06), f'Expected SSIM 1.0, got {ssim_id}'
    print('✓ Test 1 (Identity): Passed.')
    black = np.zeros((64, 64, 3), dtype=np.float64)
    white = np.ones((64, 64, 3), dtype=np.float64)
    (mse_bw, psnr_bw, ssim_bw) = calculate_metrics(black, white)
    assert np.isclose(mse_bw, 1.0), f'Expected MSE 1.0, got {mse_bw}'
    assert np.isclose(psnr_bw, 0.0), f'Expected PSNR 0.0, got {psnr_bw}'
    print(f'✓ Test 2 (Black vs White): Passed (MSE={mse_bw:.4f}, PSNR={psnr_bw:.2f} dB, SSIM={ssim_bw:.4f}).')
    noisy = np.clip(img_rand + 0.05 * np.random.RandomState(1).randn(*img_rand.shape), 0, 1)
    (mse_calc, psnr_calc, _) = calculate_metrics(noisy, img_rand)
    sk_psnr = peak_signal_noise_ratio(img_rand, noisy, data_range=1.0)
    assert np.isclose(psnr_calc, sk_psnr, atol=1e-08), f'PSNR mismatch: {psnr_calc} vs {sk_psnr}'
    print(f'✓ Test 3 (Parity with skimage PSNR): Passed ({psnr_calc:.6f} == {sk_psnr:.6f}).')
    print('\nAll canonical metrics self-tests passed successfully!')
