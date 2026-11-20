"""
Defocus Restoration Package
Classical Digital Image Processing (DIP) methods for Defocus Deblurring:
- Circular Point Spread Function (PSF) generation
- Controlled synthetic defocus degradation (convolution + noise + boundary handling)
- Wiener Deconvolution (Frequency Domain)
- Richardson-Lucy Deconvolution (Iterative Spatial Domain)
"""

from .psf import circular_psf
from .blur import apply_defocus_blur
from .wiener import wiener_deconv
from .richardson_lucy import richardson_lucy_deconv

__all__ = [
    "circular_psf",
    "apply_defocus_blur",
    "wiener_deconv",
    "richardson_lucy_deconv",
]
