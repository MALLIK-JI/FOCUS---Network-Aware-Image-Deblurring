"""
Wiener / Constrained Least Squares (Tikhonov) Deconvolution Module.

Frequency-domain linear restoration method with high-frequency noise suppression:
    F_est(u, v) = [ H*(u, v) / ( |H(u, v)|^2 + K * (1 + lambda * |P(u, v)|^2) ) ] * G(u, v)
where:
    G(u, v) : 2D Discrete Fourier Transform (DFT) of the blurred observation g(x, y)
    H(u, v) : Optical Transfer Function (DFT of the zero-padded, centered circular PSF h)
    H*(u, v) : Complex conjugate of H(u, v)
    |H(u, v)|^2 : Power spectrum of the PSF
    P(u, v) : DFT of the 2D discrete Laplacian operator [0, -1, 0; -1, 4, -1; 0, -1, 0]
    K : Base regularization parameter (Noise-to-Signal ratio).
    lambda (smoothness_weight) : Weight of the Laplacian high-frequency smoothness penalty.
        - Suppresses high-frequency speckle noise and ringing at PSF zeroes.
        - Prevents the grainy salt-and-pepper noise typical of unconstrained inverse filtering.
"""

import numpy as np
import cv2


def _pad_and_center_kernel(kernel: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Zero-pad 2D kernel to target image shape and circular-shift origin to (0, 0)."""
    th, tw = target_shape
    kh, kw = kernel.shape
    padded = np.zeros((th, tw), dtype=np.float64)

    padded[:kh, :kw] = kernel

    shift_h = -(kh // 2)
    shift_w = -(kw // 2)
    return np.roll(np.roll(padded, shift_h, axis=0), shift_w, axis=1)


def wiener_deconv_2d(
    channel: np.ndarray,
    psf: np.ndarray,
    reg_k: float = 0.015,
    smoothness_weight: float = 0.05,
    pad_ratio: float = 0.15
) -> np.ndarray:
    """
    Apply regularized Wiener deconvolution to a single 2D grayscale channel.

    Parameters
    ----------
    channel : np.ndarray
        2D array, float in [0, 1].
    psf : np.ndarray
        2D normalized PSF kernel.
    reg_k : float, default 0.015
        Noise-to-signal ratio regularization parameter (K >= 0).
    smoothness_weight : float, default 0.05
        Laplacian high-frequency grain suppression penalty (lambda >= 0).
    pad_ratio : float, default 0.15
        Boundary reflection padding fraction to minimize Fourier wrap-around ringing.

    Returns
    -------
    restored : np.ndarray
        Deconvolved 2D array, clipped to [0, 1].
    """
    h_orig, w_orig = channel.shape

    # Reflection padding around borders
    pad_h = int(np.ceil(h_orig * pad_ratio))
    pad_w = int(np.ceil(w_orig * pad_ratio))
    padded_img = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")

    target_shape = padded_img.shape
    otf_psf = _pad_and_center_kernel(psf, target_shape)

    # 2D Laplacian operator for high-frequency grain suppression
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    otf_lap = _pad_and_center_kernel(laplacian_kernel, target_shape)

    G = np.fft.fft2(padded_img)
    H = np.fft.fft2(otf_psf)
    P = np.fft.fft2(otf_lap)

    H_conj = np.conj(H)
    H_mag_sq = np.real(H_conj * H)
    P_mag_sq = np.real(np.conj(P) * P)

    # Frequency-dependent Tikhonov denominator prevents grain at PSF frequency zeros
    denom = H_mag_sq + reg_k * (1.0 + smoothness_weight * P_mag_sq)
    wiener_filter = H_conj / denom

    F_est = wiener_filter * G
    restored_padded = np.real(np.fft.ifft2(F_est))

    restored = restored_padded[pad_h:pad_h + h_orig, pad_w:pad_w + w_orig]
    return np.clip(restored, 0.0, 1.0)


def wiener_deconv(
    image: np.ndarray,
    psf: np.ndarray,
    reg_k: float = 0.015,
    smoothness_weight: float = 0.05,
    denoise_post: bool = False,
    pad_ratio: float = 0.15
) -> np.ndarray:
    """
    Wiener deconvolution for 2D (grayscale) or 3D (H, W, C) images with grain suppression.

    Parameters
    ----------
    image : np.ndarray
        Blurred image in [0, 1] range.
    psf : np.ndarray
        2D normalized Point Spread Function.
    reg_k : float, default 0.015
        Regularization parameter (noise-to-signal power ratio).
    smoothness_weight : float, default 0.05
        Weight of high-frequency Laplacian penalty to eliminate speckle grain.
    denoise_post : bool, default False
        If True, applies a gentle edge-preserving bilateral filter to eliminate residual noise.
    pad_ratio : float, default 0.15
        Boundary reflection padding ratio.

    Returns
    -------
    restored : np.ndarray
        Restored clean image, float64 in range [0, 1].
    """
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    if img.ndim == 2:
        restored = wiener_deconv_2d(
            img, psf, reg_k=reg_k, smoothness_weight=smoothness_weight, pad_ratio=pad_ratio
        )
    elif img.ndim == 3:
        restored_channels = [
            wiener_deconv_2d(
                img[:, :, c], psf, reg_k=reg_k, smoothness_weight=smoothness_weight, pad_ratio=pad_ratio
            )
            for c in range(img.shape[2])
        ]
        restored = np.stack(restored_channels, axis=-1)
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")

    if denoise_post:
        # Mild edge-preserving bilateral filter removes fine speckles without blurring edges
        img_uint8 = (restored * 255.0).round().astype(np.uint8)
        denoised = cv2.bilateralFilter(img_uint8, d=5, sigmaColor=15, sigmaSpace=5)
        restored = denoised.astype(np.float64) / 255.0

    return np.clip(restored, 0.0, 1.0)
