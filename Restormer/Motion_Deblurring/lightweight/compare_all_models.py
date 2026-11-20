import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from model import LightweightDeblurCNN
DEVICE = torch.device('cpu')
INPUT_DIR = './Datasets/test/GoPro/input'
GT_DIR = './Datasets/test/GoPro/ground_truth'
CHECKPOINTS = {'Previous Lightweight': './lightweight/checkpoints/lightweight_best.pth', 'Random Dataset Lightweight': './lightweight/checkpoints_random/lightweight_best.pth', 'Continued Lightweight': './lightweight/checkpoints_continued/lightweight_continued_best.pth'}

def calculate_metrics(pred, gt):
    pred = np.array(pred).astype(np.uint8)
    gt = np.array(gt).astype(np.uint8)
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    psnr = float('inf') if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    ssim_value = ssim(gt, pred, channel_axis=2, data_range=255)
    return (mse, psnr, ssim_value)

def evaluate_model(model, test_files):
    all_metrics = []
    for filename in test_files:
        input_path = os.path.join(INPUT_DIR, filename)
        gt_path = os.path.join(GT_DIR, filename)
        if not os.path.isfile(input_path):
            print('Missing input:', filename)
            continue
        blurred = Image.open(input_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')
        input_array = np.array(blurred).astype(np.float32) / 255.0
        tensor = torch.from_numpy(input_array.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            restored = model(tensor)
        output = restored.squeeze(0).cpu().numpy()
        output = np.clip(output.transpose(1, 2, 0), 0, 1)
        output_image = Image.fromarray((output * 255).round().astype(np.uint8))
        all_metrics.append(calculate_metrics(output_image, gt))
    if not all_metrics:
        return None
    return np.mean(all_metrics, axis=0)

def main():
    test_files = sorted([f for f in os.listdir(GT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print('Test images:', len(test_files))
    results = {}
    for (model_name, checkpoint_path) in CHECKPOINTS.items():
        print('\nEvaluating:', model_name)
        if not os.path.isfile(checkpoint_path):
            print('Checkpoint missing:', checkpoint_path)
            continue
        model = LightweightDeblurCNN().to(DEVICE)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        metrics = evaluate_model(model, test_files)
        if metrics is not None:
            results[model_name] = metrics
            print(f'MSE:  {metrics[0]:.4f}')
            print(f'PSNR: {metrics[1]:.4f}')
            print(f'SSIM: {metrics[2]:.4f}')
    print('\n========== FINAL COMPARISON ==========')
    print(f"{'Model':35s} {'MSE':>12s} {'PSNR':>12s} {'SSIM':>12s}")
    for (name, metrics) in results.items():
        print(f'{name:35s} {metrics[0]:12.4f} {metrics[1]:12.4f} {metrics[2]:12.4f}')
if __name__ == '__main__':
    main()
