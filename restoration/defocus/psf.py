"""
Circular Point Spread Function (PSF) Generator.

In geometric optics, a circular lens aperture with defocus produces a uniform
circular disk of confusion (the pillbox PSF):
    h(x, y) = 1 / (pi * R^2)  if  x^2 + y^2 <= R^2, else 0
Energy conservation requires:
    sum_{x, y} h(x, y) = 1.0
"""

import numpy as np


def circular_psf(radius: float, kernel_size: int = None, anti_aliased: bool = True) -> np.ndarray:
    """
    Generate a 2D circular disk Point Spread Function (PSF).

    Parameters
    ----------
    radius : float
        Radius of the defocus blur disk in pixels (R > 0).
    kernel_size : int, optional
        Odd integer dimension of the square kernel grid.
        Defaults to 2 * ceil(radius) + 1.
    anti_aliased : bool, default True
        If True, supersamples edge pixels (4x4 sub-pixels) for smooth anti-aliased
        disk boundary. If False, uses binary inside/outside test.

    Returns
    -------
    psf : np.ndarray
        2D float64 array of shape (kernel_size, kernel_size) normalized so that sum(psf) == 1.0.
    """
    if radius <= 0:
        raise ValueError(f"Radius must be strictly positive, got {radius}")

    min_size = int(np.ceil(radius)) * 2 + 1
    if kernel_size is None:
        kernel_size = min_size
    else:
        if kernel_size % 2 == 0:
            raise ValueError(f"Kernel size must be an odd integer, got {kernel_size}")
        if kernel_size < min_size:
            raise ValueError(f"Kernel size {kernel_size} too small for radius {radius}; minimum is {min_size}")

    center = (kernel_size - 1) / 2.0

    if anti_aliased:
        # 4x supersampling per pixel for smooth disc boundary
        sub = 4
        y_sub, x_sub = np.indices((kernel_size * sub, kernel_size * sub), dtype=np.float64)
        x_coords = (x_sub + 0.5) / sub - (center + 0.5)
        y_coords = (y_sub + 0.5) / sub - (center + 0.5)
        dist_sq = x_coords ** 2 + y_coords ** 2
        mask_sub = (dist_sq <= radius ** 2).astype(np.float64)
        # Average pooling back to kernel_size
        psf = mask_sub.reshape(kernel_size, sub, kernel_size, sub).mean(axis=(1, 3))
    else:
        y, x = np.ogrid[:kernel_size, :kernel_size]
        dist_sq = (x - center) ** 2 + (y - center) ** 2
        psf = (dist_sq <= radius ** 2).astype(np.float64)

    total = np.sum(psf)
    if total == 0:
        # Fallback to single center impulse if disk is too tiny
        psf[int(center), int(center)] = 1.0
    else:
        psf /= total

    return psf
