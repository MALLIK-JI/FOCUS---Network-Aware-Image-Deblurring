import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from restoration.defocus.psf import circular_psf
from restoration.defocus.blur import apply_defocus_blur
from restoration.defocus.wiener import wiener_deconv
from restoration.defocus.richardson_lucy import richardson_lucy_deconv

def compute_metrics(gt: np.ndarray, pred: np.ndarray):
    mse = float(np.mean((gt - pred) ** 2))
    psnr = float('inf') if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    ssim_val = float(ssim(gt, pred, channel_axis=2, data_range=1.0))
    return (mse, psnr, ssim_val)

def save_visual_comparison(out_path: Path, sharp: np.ndarray, blurred: np.ndarray, wiener: np.ndarray, rl: np.ndarray, radius: float, sigma: float):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    err_blurred = np.abs(blurred - sharp)
    err_wiener = np.abs(wiener - sharp)
    err_rl = np.abs(rl - sharp)
    (fig, axes) = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(sharp)
    axes[0, 0].set_title('1. Ground Truth (Sharp)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(blurred)
    axes[0, 1].set_title(f'2. Defocus Blurred (R={radius}, σ={sigma})', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(np.clip(err_blurred * 3.0, 0, 1))
    axes[0, 2].set_title('Error Map: |Blurred - Sharp| (×3)', fontsize=11)
    axes[0, 2].axis('off')
    axes[1, 0].imshow(wiener)
    axes[1, 0].set_title('3. Wiener Restored (K=1e-3)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(rl)
    axes[1, 1].set_title('4. Richardson-Lucy Restored (30 iters)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    axes[1, 2].imshow(np.clip(err_rl * 3.0, 0, 1))
    axes[1, 2].set_title('Error Map: |RL - Sharp| (×3)', fontsize=11)
    axes[1, 2].axis('off')
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

def run_defocus_experiments():
    sample_images = [PROJECT_ROOT / 'Restormer' / 'Motion_Deblurring' / 'Datasets' / 'test' / 'GoPro' / 'ground_truth' / 'image1.jpg', PROJECT_ROOT / 'Restormer' / 'Motion_Deblurring' / 'Datasets' / 'test' / 'GoPro' / 'ground_truth' / 'image2.jpg']
    results_dir = PROJECT_ROOT / 'results' / 'defocus_eval'
    vis_dir = results_dir / 'visualizations'
    results_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    test_radii = [2.5, 4.0, 6.0]
    test_noises = [0.0, 0.005, 0.01]
    records = []
    print('=' * 65)
    print('Classical Defocus Deblurring Benchmark (Wiener vs Richardson-Lucy)')
    print('=' * 65)
    for img_path in sample_images:
        if not img_path.exists():
            print(f'Skipping missing image: {img_path}')
            continue
        img_name = img_path.stem
        sharp_pil = Image.open(img_path).convert('RGB')
        sharp = np.asarray(sharp_pil, dtype=np.float64) / 255.0
        print(f'\nProcessing reference image: {img_name} (Dimensions: {sharp.shape[1]}x{sharp.shape[0]})')
        for radius in test_radii:
            psf = circular_psf(radius=radius)
            for sigma in test_noises:
                print(f'  Condition -> Radius: {radius:.1f} px | Noise std: {sigma:.3f}')
                t0 = time.perf_counter()
                blurred = apply_defocus_blur(sharp, psf, noise_sigma=sigma, seed=42)
                t_blur = time.perf_counter() - t0
                (b_mse, b_psnr, b_ssim) = compute_metrics(sharp, blurred)
                records.append({'Image': img_name, 'Radius': radius, 'Noise_Sigma': sigma, 'Method': 'Blurred_Input', 'MSE': b_mse, 'PSNR_dB': b_psnr, 'SSIM': b_ssim, 'Time_sec': t_blur})
                reg_k = 0.001 if sigma == 0.0 else max(0.001, sigma ** 2 * 10)
                t0 = time.perf_counter()
                wiener_out = wiener_deconv(blurred, psf, reg_k=reg_k)
                t_wiener = time.perf_counter() - t0
                (w_mse, w_psnr, w_ssim) = compute_metrics(sharp, wiener_out)
                records.append({'Image': img_name, 'Radius': radius, 'Noise_Sigma': sigma, 'Method': 'Wiener_Deconvolution', 'MSE': w_mse, 'PSNR_dB': w_psnr, 'SSIM': w_ssim, 'Time_sec': t_wiener})
                num_iters = 30
                t0 = time.perf_counter()
                rl_out = richardson_lucy_deconv(blurred, psf, num_iters=num_iters)
                t_rl = time.perf_counter() - t0
                (rl_mse, rl_psnr, rl_ssim) = compute_metrics(sharp, rl_out)
                records.append({'Image': img_name, 'Radius': radius, 'Noise_Sigma': sigma, 'Method': f'Richardson_Lucy_{num_iters}iters', 'MSE': rl_mse, 'PSNR_dB': rl_psnr, 'SSIM': rl_ssim, 'Time_sec': t_rl})
                print(f'    Blurred Input:    PSNR = {b_psnr:6.2f} dB | SSIM = {b_ssim:.4f}')
                print(f'    Wiener (K={reg_k:.1e}): PSNR = {w_psnr:6.2f} dB | SSIM = {w_ssim:.4f} | Time = {t_wiener:.4f}s')
                print(f'    Richardson-Lucy:  PSNR = {rl_psnr:6.2f} dB | SSIM = {rl_ssim:.4f} | Time = {t_rl:.4f}s')
                vis_file = vis_dir / f'defocus_{img_name}_R{radius}_N{sigma}.png'
                save_visual_comparison(vis_file, sharp, blurred, wiener_out, rl_out, radius, sigma)
    df = pd.DataFrame(records)
    csv_path = results_dir / 'defocus_benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    print('\n' + '=' * 65)
    print(f'Benchmark completed successfully! Results saved to:\n{csv_path}')
    print(f'Visualizations saved to:\n{vis_dir}')
    print('=' * 65)
    summary = df.groupby(['Method', 'Radius', 'Noise_Sigma'])[['PSNR_dB', 'SSIM', 'Time_sec']].mean()
    print('\nBenchmark Summary (Grouped by Method, Radius, Noise):')
    print(summary.to_string())
if __name__ == '__main__':
    run_defocus_experiments()
