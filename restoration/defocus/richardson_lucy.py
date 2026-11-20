"""
Richardson-Lucy Deconvolution Module for Classical Defocus Deblurring.

Includes:
- Standard Maximum Likelihood multiplicative update (Richardson 1972, Lucy 1974)
- White's Damped Richardson-Lucy (White 1994, Hubble Space Telescope deconvolution)
  which bounds noise amplification in smooth regions without losing edge contrast.
- Optional edge acutance / contrast recovery post-filter.
"""

import numpy as np
from scipy.signal import fftconvolve
from PIL import Image, ImageFilter


def _fft_conv2d_symmetric(img_2d: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Perform 2D convolution with symmetric boundary extension using FFT."""
    h_orig, w_orig = img_2d.shape
    kh, kw = psf.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(img_2d, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    conv = fftconvolve(padded, psf, mode="same")
    return conv[pad_h:pad_h + h_orig, pad_w:pad_w + w_orig]


def richardson_lucy_2d(
    channel: np.ndarray,
    psf: np.ndarray,
    num_iters: int = 35,
    damping_threshold: float = 0.003,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Perform Damped Richardson-Lucy deconvolution on a single 2D grayscale channel.

    Parameters
    ----------
    channel : np.ndarray
        2D array, normalized in [0, 1].
    psf : np.ndarray
        2D normalized Point Spread Function.
    num_iters : int, default 35
        Number of multiplicative EM iterations.
    damping_threshold : float, default 0.003
        Noise threshold for White's damping. Damps noise amplification in flat regions
        where residuals are below the noise floor.
    eps : float, default 1e-12
        Safeguard constant.

    Returns
    -------
    restored : np.ndarray
        2D array, float64 clipped to [0, 1].
    """
    psf_flip = np.flip(psf)
    g = channel.copy()

    # Initial estimate
    f_est = np.maximum(g, eps)

    for _ in range(num_iters):
        reblurred = _fft_conv2d_symmetric(f_est, psf)
        reblurred = np.maximum(reblurred, eps)
        diff = g - reblurred

        # White's Damping: suppress updates when differences are within noise threshold
        if damping_threshold > 0:
            weight = np.clip((np.abs(diff) / damping_threshold) ** 2, 0.0, 1.0)
            ratio = 1.0 + (diff * weight) / reblurred
        else:
            ratio = g / reblurred

        correction = _fft_conv2d_symmetric(ratio, psf_flip)
        f_est = f_est * correction
        f_est = np.clip(f_est, 0.0, 1.0)

    return f_est


def richardson_lucy_deconv(
    image: np.ndarray,
    psf: np.ndarray,
    num_iters: int = 35,
    damping_threshold: float = 0.003,
    edge_boost: int = 0,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Damped Richardson-Lucy deconvolution for 2D (grayscale) or 3D (H, W, C) images.

    Parameters
    ----------
    image : np.ndarray
        Blurred image in [0, 1] range.
    psf : np.ndarray
        2D normalized Point Spread Function.
    num_iters : int, default 35
        Number of iterations.
    damping_threshold : float, default 0.003
        Noise damping threshold.
    edge_boost : int, default 0
        Percentage of edge acutance / unsharp masking to restore sharp edges.

    Returns
    -------
    restored : np.ndarray
        Restored clean, sharp image in [0, 1] range.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    if img.ndim == 2:
        restored = richardson_lucy_2d(
            img, psf, num_iters=num_iters, damping_threshold=damping_threshold, eps=eps
        )
    elif img.ndim == 3:
        restored_channels = [
            richardson_lucy_2d(
                img[:, :, c], psf, num_iters=num_iters, damping_threshold=damping_threshold, eps=eps
            )
            for c in range(img.shape[2])
        ]
        restored = np.stack(restored_channels, axis=-1)
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")

    if edge_boost > 0:
        pil_img = Image.fromarray((restored * 255.0).round().astype(np.uint8))
        boosted = pil_img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=edge_boost, threshold=2))
        restored = np.asarray(boosted, dtype=np.float64) / 255.0

    return np.clip(restored, 0.0, 1.0)
