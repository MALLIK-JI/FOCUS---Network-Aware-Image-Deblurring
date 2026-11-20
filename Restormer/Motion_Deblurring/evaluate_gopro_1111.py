import sys
import csv
import time
import gc
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / 'test'
CONFIG_PATH = BASE_DIR / 'Options' / 'Deblurring_Restormer.yml'
ORIGINAL_CHECKPOINT = BASE_DIR / 'pretrained_models' / 'motion_deblurring.pth'
CUSTOM_CHECKPOINT = BASE_DIR.parent / 'experiments' / 'Deblurring_Restormer_CustomLoss' / 'models' / 'net_g_100.pth'
LIGHTWEIGHT_CHECKPOINT = BASE_DIR / 'lightweight' / 'checkpoints_continued' / 'lightweight_continued_best.pth'
OUTPUT_DIR = BASE_DIR / 'full_gopro_evaluation_1111'
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_CSV = OUTPUT_DIR / 'per_image_results.csv'
SUMMARY_CSV = OUTPUT_DIR / 'model_summary.csv'
FAILED_CSV = OUTPUT_DIR / 'failed_images.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 10
EXPECTED_IMAGES = 1111
MODEL_NAMES = ['Blurred Input', 'Original Restormer', 'Custom Loss Restormer', 'Continued Lightweight CNN']
CSV_FIELDS = ['Sequence', 'Image', 'Model', 'MSE', 'PSNR', 'SSIM', 'Inference Time (seconds)']

def check_files():
    required = [TEST_DIR, CONFIG_PATH, ORIGINAL_CHECKPOINT, CUSTOM_CHECKPOINT, LIGHTWEIGHT_CHECKPOINT]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f'Required file not found: {path}')
    if DEVICE.type == 'cuda':
        print('GPU:', torch.cuda.get_device_name(0))
        print('GPU memory:', round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2), 'GB')
    else:
        print('WARNING: CUDA unavailable. Restormer will run on CPU.')

def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0

def image_to_tensor(image, device):
    tensor = torch.from_numpy(image)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)

def calculate_metrics(prediction, target):
    prediction = np.nan_to_num(prediction, nan=0.0, posinf=1.0, neginf=0.0)
    prediction = np.clip(prediction, 0, 1)
    target = np.clip(target, 0, 1)
    mse = float(np.mean((prediction - target) ** 2))
    psnr = float(peak_signal_noise_ratio(target, prediction, data_range=1.0))
    ssim = float(structural_similarity(target, prediction, channel_axis=2, data_range=1.0))
    return (mse, psnr, ssim)

def find_test_pairs():
    pairs = []
    for sequence_dir in sorted(TEST_DIR.iterdir()):
        if not sequence_dir.is_dir():
            continue
        blur_dir = sequence_dir / 'blur'
        sharp_dir = sequence_dir / 'sharp'
        if not blur_dir.exists() or not sharp_dir.exists():
            continue
        blur_files = {p.name: p for p in blur_dir.glob('*.png')}
        sharp_files = {p.name: p for p in sharp_dir.glob('*.png')}
        if set(blur_files) != set(sharp_files):
            raise RuntimeError(f'Blur/sharp mismatch in {sequence_dir.name}')
        for filename in sorted(blur_files):
            pairs.append({'sequence': sequence_dir.name, 'filename': filename, 'blur': blur_files[filename], 'sharp': sharp_files[filename]})
    return pairs

def load_restormer(checkpoint_path):
    sys.path.insert(0, str(BASE_DIR.parent))
    from basicsr.models.archs.restormer_arch import Restormer
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    network_config = config['network_g'].copy()
    network_config.pop('type', None)
    model = Restormer(**network_config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)
    del checkpoint, state_dict
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    return model

def run_restormer(model, image):
    (height, width) = image.shape[:2]
    tensor = image_to_tensor(image, DEVICE)
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    with torch.inference_mode():
        output = model(tensor)
    output = output[:, :, :height, :width]
    output = output.clamp(0, 1)
    return output.squeeze(0).permute(1, 2, 0).cpu().numpy()

def load_lightweight():
    from lightweight.model import LightweightDeblurCNN
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
    model.eval()
    model.to('cpu')
    del checkpoint, state_dict
    gc.collect()
    return model

def run_lightweight(model, image):
    tensor = image_to_tensor(image, torch.device('cpu'))
    with torch.inference_mode():
        output = model(tensor)
    return output.squeeze(0).permute(1, 2, 0).numpy()

def get_completed_keys():
    completed = set()
    if not RESULTS_CSV.exists():
        return completed
    with open(RESULTS_CSV, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = (row['Sequence'], row['Image'], row['Model'])
            completed.add(key)
    return completed

def save_rows(rows):
    if not rows:
        return
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def save_failed(rows):
    if not rows:
        return
    write_header = not FAILED_CSV.exists()
    with open(FAILED_CSV, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Sequence', 'Image', 'Model', 'Error'])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

def synchronize():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

def evaluate_model(model_name, model, pairs, completed):
    print('\n' + '=' * 65)
    print('EVALUATING:', model_name)
    print('=' * 65)
    pending_rows = []
    failed_rows = []
    newly_completed = 0
    start_all = time.perf_counter()
    for (index, pair) in enumerate(pairs, start=1):
        sequence = pair['sequence']
        filename = pair['filename']
        key = (sequence, filename, model_name)
        if key in completed:
            continue
        try:
            blurred = load_image(pair['blur'])
            ground_truth = load_image(pair['sharp'])
            if blurred.shape != ground_truth.shape:
                raise ValueError('Blur/sharp dimensions do not match')
            if model_name == 'Blurred Input':
                prediction = blurred
                inference_time = 0.0
            else:
                synchronize()
                start = time.perf_counter()
                if model_name == 'Continued Lightweight CNN':
                    prediction = run_lightweight(model, blurred)
                else:
                    prediction = run_restormer(model, blurred)
                synchronize()
                inference_time = time.perf_counter() - start
            if prediction.shape != ground_truth.shape:
                raise ValueError(f'Output dimensions {prediction.shape} do not match target {ground_truth.shape}')
            (mse, psnr, ssim) = calculate_metrics(prediction, ground_truth)
            pending_rows.append({'Sequence': sequence, 'Image': filename, 'Model': model_name, 'MSE': mse, 'PSNR': psnr, 'SSIM': ssim, 'Inference Time (seconds)': inference_time})
            completed.add(key)
            newly_completed += 1
            print(f'{model_name} | {index}/{len(pairs)} | {sequence}/{filename} | PSNR={psnr:.4f} | SSIM={ssim:.4f} | Time={inference_time:.3f}s')
            if len(pending_rows) >= SAVE_EVERY:
                save_rows(pending_rows)
                pending_rows.clear()
        except Exception as error:
            print(f'FAILED: {sequence}/{filename} | {error}')
            failed_rows.append({'Sequence': sequence, 'Image': filename, 'Model': model_name, 'Error': str(error)})
        if len(failed_rows) >= SAVE_EVERY:
            save_failed(failed_rows)
            failed_rows.clear()
        if newly_completed > 0 and newly_completed % SAVE_EVERY == 0:
            elapsed = time.perf_counter() - start_all
            speed = elapsed / newly_completed
            remaining = len(pairs) - index
            print(f'\nCheckpoint: {newly_completed} new results | Elapsed: {elapsed / 60:.1f} min | Estimated remaining for this model: {speed * remaining / 60:.1f} min\n')
    save_rows(pending_rows)
    save_failed(failed_rows)
    print(f'\nFinished {model_name}. New successful results: {newly_completed}')

def generate_summary():
    if not RESULTS_CSV.exists():
        print('No results found.')
        return
    import pandas as pd
    df = pd.read_csv(RESULTS_CSV)
    summary = df.groupby('Model').agg(Images_Evaluated=('Image', 'nunique'), Average_MSE=('MSE', 'mean'), Std_MSE=('MSE', 'std'), Average_PSNR=('PSNR', 'mean'), Std_PSNR=('PSNR', 'std'), Average_SSIM=('SSIM', 'mean'), Std_SSIM=('SSIM', 'std'), Average_Inference_Time=('Inference Time (seconds)', 'mean')).reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)
    print('\n' + '=' * 65)
    print('CURRENT EVALUATION SUMMARY')
    print('=' * 65)
    print(summary.to_string(index=False))
    sequence_summary = df.groupby(['Sequence', 'Model']).agg(Images=('Image', 'nunique'), MSE=('MSE', 'mean'), PSNR=('PSNR', 'mean'), SSIM=('SSIM', 'mean')).reset_index()
    sequence_summary.to_csv(OUTPUT_DIR / 'sequence_summary.csv', index=False)
    print('\nResults folder:', OUTPUT_DIR)

def main():
    print('=' * 65)
    print('FULL GOPRO 1111-IMAGE EVALUATION')
    print('=' * 65)
    check_files()
    pairs = find_test_pairs()
    print('\nTotal paired images found:', len(pairs))
    if len(pairs) != EXPECTED_IMAGES:
        raise RuntimeError(f'Expected {EXPECTED_IMAGES} pairs, found {len(pairs)}. Check the test directory before evaluation.')
    completed = get_completed_keys()
    print('Existing result rows:', len(completed))
    print('\nEvaluating blurred input baseline...')
    evaluate_model('Blurred Input', None, pairs, completed)
    print('\nLoading Original Restormer...')
    original_model = load_restormer(ORIGINAL_CHECKPOINT)
    evaluate_model('Original Restormer', original_model, pairs, completed)
    del original_model
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    print('Original Restormer unloaded.')
    print('\nLoading Custom-Loss Restormer...')
    custom_model = load_restormer(CUSTOM_CHECKPOINT)
    evaluate_model('Custom Loss Restormer', custom_model, pairs, completed)
    del custom_model
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    print('Custom-Loss Restormer unloaded.')
    print('\nLoading Lightweight CNN on CPU...')
    lightweight_model = load_lightweight()
    evaluate_model('Continued Lightweight CNN', lightweight_model, pairs, completed)
    del lightweight_model
    gc.collect()
    generate_summary()
    print('\nEVALUATION SCRIPT FINISHED.')
if __name__ == '__main__':
    main()
