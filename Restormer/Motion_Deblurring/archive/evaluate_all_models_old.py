import os
import sys
import csv
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basicsr.models.archs.restormer_arch import Restormer
from lightweight.model import LightweightDeblurCNN
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / 'Datasets/test/GoPro/input'
GT_DIR = BASE_DIR / 'Datasets/test/GoPro/ground_truth'
CONFIG_PATH = BASE_DIR / 'Options/Deblurring_Restormer.yml'
ORIGINAL_CHECKPOINT = BASE_DIR / 'pretrained_models/motion_deblurring.pth'
CUSTOM_CHECKPOINT = BASE_DIR.parent / 'experiments/Deblurring_Restormer_CustomLoss/models/net_g_100.pth'
LIGHTWEIGHT_CHECKPOINT = BASE_DIR / 'lightweight/checkpoints_continued/lightweight_continued_best.pth'
OUTPUT_CSV = BASE_DIR / 'all_models_test_results.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_metrics(prediction, target):
    prediction = np.clip(prediction, 0, 1)
    target = np.clip(target, 0, 1)
    mse = np.mean((prediction - target) ** 2)
    psnr = peak_signal_noise_ratio(target, prediction, data_range=1.0)
    ssim = structural_similarity(target, prediction, channel_axis=2, data_range=1.0)
    return (float(mse), float(psnr), float(ssim))

def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return image

def image_to_tensor(image):
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)

def load_restormer(checkpoint_path):
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    model = Restormer(**config['network_g'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(DEVICE)
    return model

def load_lightweight():
    model = LightweightDeblurCNN(channels=32, num_blocks=5)
    checkpoint = torch.load(LIGHTWEIGHT_CHECKPOINT, map_location='cpu')
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(DEVICE)
    return model

def run_restormer(model, image):
    (height, width) = image.shape[:2]
    tensor = image_to_tensor(image)
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    with torch.inference_mode():
        output = model(tensor)
    output = output[:, :, :height, :width]
    output = output.clamp(0, 1)
    return output.squeeze(0).permute(1, 2, 0).cpu().numpy()

def run_lightweight(model, image):
    tensor = image_to_tensor(image)
    with torch.inference_mode():
        output = model(tensor)
    return output.squeeze(0).permute(1, 2, 0).cpu().numpy()

def main():
    print('Evaluation device:', DEVICE)
    if DEVICE.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
    original_model = load_restormer(ORIGINAL_CHECKPOINT)
    print('Original Restormer loaded.')
    custom_model = load_restormer(CUSTOM_CHECKPOINT)
    print('Custom-loss Restormer loaded.')
    lightweight_model = load_lightweight()
    print('Continued Lightweight CNN loaded.')
    image_names = sorted((f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and (GT_DIR / f).exists()))
    if not image_names:
        raise RuntimeError('No matching input and ground-truth images found.')
    results = []
    for (index, filename) in enumerate(image_names, start=1):
        print(f'\nProcessing {index}/{len(image_names)}: {filename}')
        blurred = load_image(INPUT_DIR / filename)
        ground_truth = load_image(GT_DIR / filename)
        if blurred.shape != ground_truth.shape:
            raise ValueError(f'Image dimensions do not match: {filename}')
        models = {'Blurred Input': None, 'Original Restormer': original_model, 'Custom Loss Restormer': custom_model, 'Continued Lightweight CNN': lightweight_model}
        for (model_name, model) in models.items():
            start = time.perf_counter()
            if model_name == 'Blurred Input':
                prediction = blurred.copy()
            elif model_name == 'Continued Lightweight CNN':
                prediction = run_lightweight(model, blurred)
            else:
                prediction = run_restormer(model, blurred)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.perf_counter() - start
            (mse, psnr, ssim) = calculate_metrics(prediction, ground_truth)
            results.append({'Image': filename, 'Model': model_name, 'MSE': mse, 'PSNR': psnr, 'SSIM': ssim, 'Inference Time (seconds)': inference_time})
            print(f'{model_name}: MSE={mse:.6f}, PSNR={psnr:.4f}, SSIM={ssim:.4f}, Time={inference_time:.4f}s')
    with open(OUTPUT_CSV, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print('\n========== AVERAGE RESULTS ==========')
    model_names = list(dict.fromkeys((row['Model'] for row in results)))
    for model_name in model_names:
        model_rows = [row for row in results if row['Model'] == model_name]
        avg_mse = np.mean([row['MSE'] for row in model_rows])
        avg_psnr = np.mean([row['PSNR'] for row in model_rows])
        avg_ssim = np.mean([row['SSIM'] for row in model_rows])
        avg_time = np.mean([row['Inference Time (seconds)'] for row in model_rows])
        print(f'\n{model_name}')
        print(f'Average MSE: {avg_mse:.6f}')
        print(f'Average PSNR: {avg_psnr:.4f} dB')
        print(f'Average SSIM: {avg_ssim:.4f}')
        print(f'Average inference time: {avg_time:.4f} seconds')
    print('\nResults saved to:', OUTPUT_CSV)
if __name__ == '__main__':
    main()
