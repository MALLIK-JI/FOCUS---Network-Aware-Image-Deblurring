import sys
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR.parent))
from lightweight.model import LightweightDeblurCNN
INPUT_DIR = BASE_DIR / 'Datasets/test/GoPro/input'
GT_DIR = BASE_DIR / 'Datasets/test/GoPro/ground_truth'
MODEL_DIRS = {'Ground Truth': GT_DIR, 'Blurred Input': INPUT_DIR, 'Original Restormer': BASE_DIR / 'results_original', 'Custom Loss Restormer': BASE_DIR / 'results_custom_loss', 'Previous Lightweight CNN': BASE_DIR / 'lightweight/results', 'Random Dataset Lightweight CNN': BASE_DIR / 'lightweight/results_random'}
CHECKPOINT = BASE_DIR / 'lightweight/checkpoints_continued/lightweight_continued_best.pth'
OUTPUT_DIR = BASE_DIR / 'visual_comparisons'
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_lightweight():
    model = LightweightDeblurCNN(channels=32, num_blocks=5)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
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

def run_lightweight(model, image):
    image_float = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_float)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        output = model(tensor)
    output = output.squeeze(0).permute(1, 2, 0)
    output = output.cpu().numpy()
    return np.clip(output * 255, 0, 255).astype(np.uint8)

def find_image(folder, filename):
    stem = Path(filename).stem
    for extension in ['.png', '.jpg', '.jpeg']:
        path = folder / (stem + extension)
        if path.is_file():
            return path
    return None

def main():
    print('Device:', DEVICE)
    continued_model = load_lightweight()
    print('Continued Lightweight CNN loaded.')
    image_names = sorted((p.name for p in INPUT_DIR.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png'] and (GT_DIR / p.name).exists()))
    if not image_names:
        raise RuntimeError('No matching GoPro test images found.')
    for filename in image_names:
        print('\nCreating comparison:', filename)
        images = {}
        for (name, folder) in MODEL_DIRS.items():
            path = find_image(folder, filename)
            if path is None:
                raise FileNotFoundError(f'Missing {name} output for {filename}: {folder}')
            images[name] = load_image(path)
        blurred = images['Blurred Input']
        images['Continued Lightweight CNN'] = run_lightweight(continued_model, blurred)
        reference_shape = images['Ground Truth'].shape
        for (name, image) in images.items():
            if image.shape != reference_shape:
                raise ValueError(f'Dimension mismatch for {name}: {image.shape} vs {reference_shape}')
        titles = list(images.keys())
        (fig, axes) = plt.subplots(1, len(titles), figsize=(24, 6))
        for (ax, title) in zip(axes, titles):
            ax.imshow(images[title])
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        fig.suptitle(f'GoPro Motion Deblurring Comparison - {filename}', fontsize=15)
        plt.tight_layout()
        output_path = OUTPUT_DIR / f'comparison_{Path(filename).stem}.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print('Saved:', output_path)
    print('\nAll visual comparisons completed.')
if __name__ == '__main__':
    main()
