import numpy as np

def circular_psf(radius: float, kernel_size: int=None, anti_aliased: bool=True) -> np.ndarray:
    if radius <= 0:
        raise ValueError(f'Radius must be strictly positive, got {radius}')
    min_size = int(np.ceil(radius)) * 2 + 1
    if kernel_size is None:
        kernel_size = min_size
    else:
        if kernel_size % 2 == 0:
            raise ValueError(f'Kernel size must be an odd integer, got {kernel_size}')
        if kernel_size < min_size:
            raise ValueError(f'Kernel size {kernel_size} too small for radius {radius}; minimum is {min_size}')
    center = (kernel_size - 1) / 2.0
    if anti_aliased:
        sub = 4
        (y_sub, x_sub) = np.indices((kernel_size * sub, kernel_size * sub), dtype=np.float64)
        x_coords = (x_sub + 0.5) / sub - (center + 0.5)
        y_coords = (y_sub + 0.5) / sub - (center + 0.5)
        dist_sq = x_coords ** 2 + y_coords ** 2
        mask_sub = (dist_sq <= radius ** 2).astype(np.float64)
        psf = mask_sub.reshape(kernel_size, sub, kernel_size, sub).mean(axis=(1, 3))
    else:
        (y, x) = np.ogrid[:kernel_size, :kernel_size]
        dist_sq = (x - center) ** 2 + (y - center) ** 2
        psf = (dist_sq <= radius ** 2).astype(np.float64)
    total = np.sum(psf)
    if total == 0:
        psf[int(center), int(center)] = 1.0
    else:
        psf /= total
    return psf
