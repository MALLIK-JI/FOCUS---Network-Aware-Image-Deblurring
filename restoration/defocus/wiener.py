import numpy as np
import cv2

def _pad_and_center_kernel(kernel: np.ndarray, target_shape: tuple) -> np.ndarray:
    (th, tw) = target_shape
    (kh, kw) = kernel.shape
    padded = np.zeros((th, tw), dtype=np.float64)
    padded[:kh, :kw] = kernel
    shift_h = -(kh // 2)
    shift_w = -(kw // 2)
    return np.roll(np.roll(padded, shift_h, axis=0), shift_w, axis=1)

def wiener_deconv_2d(channel: np.ndarray, psf: np.ndarray, reg_k: float=0.015, smoothness_weight: float=0.05, pad_ratio: float=0.15) -> np.ndarray:
    (h_orig, w_orig) = channel.shape
    pad_h = int(np.ceil(h_orig * pad_ratio))
    pad_w = int(np.ceil(w_orig * pad_ratio))
    padded_img = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    target_shape = padded_img.shape
    otf_psf = _pad_and_center_kernel(psf, target_shape)
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    otf_lap = _pad_and_center_kernel(laplacian_kernel, target_shape)
    G = np.fft.fft2(padded_img)
    H = np.fft.fft2(otf_psf)
    P = np.fft.fft2(otf_lap)
    H_conj = np.conj(H)
    H_mag_sq = np.real(H_conj * H)
    P_mag_sq = np.real(np.conj(P) * P)
    denom = H_mag_sq + reg_k * (1.0 + smoothness_weight * P_mag_sq)
    wiener_filter = H_conj / denom
    F_est = wiener_filter * G
    restored_padded = np.real(np.fft.ifft2(F_est))
    restored = restored_padded[pad_h:pad_h + h_orig, pad_w:pad_w + w_orig]
    return np.clip(restored, 0.0, 1.0)

def wiener_deconv(image: np.ndarray, psf: np.ndarray, reg_k: float=0.015, smoothness_weight: float=0.05, denoise_post: bool=False, pad_ratio: float=0.15) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        restored = wiener_deconv_2d(img, psf, reg_k=reg_k, smoothness_weight=smoothness_weight, pad_ratio=pad_ratio)
    elif img.ndim == 3:
        restored_channels = [wiener_deconv_2d(img[:, :, c], psf, reg_k=reg_k, smoothness_weight=smoothness_weight, pad_ratio=pad_ratio) for c in range(img.shape[2])]
        restored = np.stack(restored_channels, axis=-1)
    else:
        raise ValueError(f'Expected 2D or 3D image, got shape {img.shape}')
    if denoise_post:
        img_uint8 = (restored * 255.0).round().astype(np.uint8)
        denoised = cv2.bilateralFilter(img_uint8, d=5, sigmaColor=15, sigmaSpace=5)
        restored = denoised.astype(np.float64) / 255.0
    return np.clip(restored, 0.0, 1.0)
