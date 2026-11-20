import numpy as np
from restoration.defocus.psf import circular_psf
from restoration.defocus.blur import apply_defocus_blur
from restoration.defocus.wiener import wiener_deconv
from restoration.defocus.richardson_lucy import richardson_lucy_deconv

def test_psf_properties():
    for r in [2.0, 3.5, 5.0, 7.0]:
        psf = circular_psf(radius=r)
        assert np.isclose(np.sum(psf), 1.0, atol=1e-12), f'PSF sum != 1 for radius {r}'
        assert psf.shape[0] % 2 == 1, f'Kernel height not odd for radius {r}'
        assert psf.shape[0] == psf.shape[1], 'Kernel not square'
        assert np.allclose(psf, np.flip(psf, axis=0), atol=1e-10), 'PSF not symmetric vertically'
        assert np.allclose(psf, np.flip(psf, axis=1), atol=1e-10), 'PSF not symmetric horizontally'
        assert np.all(psf >= 0.0), 'PSF contains negative values'

def test_blur_properties():
    img = np.linspace(0.1, 0.9, 100).reshape(10, 10)
    psf = circular_psf(radius=2.0)
    blurred = apply_defocus_blur(img, psf, noise_sigma=0.0)
    assert blurred.shape == img.shape
    assert np.all(blurred >= 0.0) and np.all(blurred <= 1.0)
    assert np.isclose(np.mean(img), np.mean(blurred), atol=0.05)

def test_wiener_deconvolution_stability():
    img = np.random.RandomState(42).rand(64, 64, 3)
    psf = circular_psf(radius=3.0)
    blurred = apply_defocus_blur(img, psf, noise_sigma=0.005, seed=42)
    restored = wiener_deconv(blurred, psf, reg_k=0.001)
    assert restored.shape == img.shape
    assert not np.isnan(restored).any(), 'Wiener produced NaNs'
    assert not np.isinf(restored).any(), 'Wiener produced Infs'
    assert np.all(restored >= 0.0) and np.all(restored <= 1.0)

def test_richardson_lucy_stability():
    img = np.random.RandomState(42).rand(64, 64, 3)
    psf = circular_psf(radius=3.0)
    blurred = apply_defocus_blur(img, psf, noise_sigma=0.005, seed=42)
    restored = richardson_lucy_deconv(blurred, psf, num_iters=15)
    assert restored.shape == img.shape
    assert not np.isnan(restored).any(), 'RL produced NaNs'
    assert not np.isinf(restored).any(), 'RL produced Infs'
    assert np.all(restored >= 0.0) and np.all(restored <= 1.0)
if __name__ == '__main__':
    print('Running unit tests...')
    test_psf_properties()
    print('✓ PSF properties test passed')
    test_blur_properties()
    print('✓ Synthetic blur properties test passed')
    test_wiener_deconvolution_stability()
    print('✓ Wiener deconvolution stability test passed')
    test_richardson_lucy_stability()
    print('✓ Richardson-Lucy stability test passed')
    print('\nAll classical defocus unit tests passed successfully!')
