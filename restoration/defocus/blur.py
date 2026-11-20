import numpy as np
from scipy.signal import fftconvolve

def _fft_conv2d_symmetric(img_2d: np.ndarray, psf: np.ndarray) -> np.ndarray:
    (h_orig, w_orig) = img_2d.shape
    (kh, kw) = psf.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(img_2d, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    conv = fftconvolve(padded, psf, mode='same')
    return conv[pad_h:pad_h + h_orig, pad_w:pad_w + w_orig]

def apply_defocus_blur(image: np.ndarray, psf: np.ndarray, noise_sigma: float=0.0, seed: int=None) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        blurred = _fft_conv2d_symmetric(img, psf)
    elif img.ndim == 3:
        channels = [_fft_conv2d_symmetric(img[:, :, c], psf) for c in range(img.shape[2])]
        blurred = np.stack(channels, axis=-1)
    else:
        raise ValueError(f'Expected 2D or 3D image, got shape {img.shape}')
    if noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=noise_sigma, size=blurred.shape)
        blurred = blurred + noise
    return np.clip(blurred, 0.0, 1.0)
