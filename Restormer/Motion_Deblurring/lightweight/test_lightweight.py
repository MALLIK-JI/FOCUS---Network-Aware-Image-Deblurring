import os
import time
import torch
from PIL import Image
from torchvision import transforms
from model import LightweightDeblurCNN
INPUT_DIR = './Datasets/test/GoPro/input'
OUTPUT_DIR = './lightweight/results_random'
CHECKPOINT = './lightweight/checkpoints_random/lightweight_best.pth'
DEVICE = torch.device('cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)
transform = transforms.ToTensor()
model = LightweightDeblurCNN().to(DEVICE)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()
image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print('Device:', DEVICE)
print('Checkpoint:', CHECKPOINT)
print('Test images:', len(image_files))
total_time = 0.0
with torch.no_grad():
    for filename in image_files:
        image_path = os.path.join(INPUT_DIR, filename)
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        start_time = time.perf_counter()
        restored = model(input_tensor)
        elapsed_time = time.perf_counter() - start_time
        total_time += elapsed_time
        restored_image = transforms.ToPILImage()(restored.squeeze(0).cpu())
        output_path = os.path.join(OUTPUT_DIR, filename)
        restored_image.save(output_path)
        print(f'\nRestored: {filename}')
        print(f'Inference time: {elapsed_time:.4f} seconds')
print('\n========== TESTING COMPLETED ==========')
print('Total inference time:', round(total_time, 4), 'seconds')
print('Average inference time:', round(total_time / max(len(image_files), 1), 4), 'seconds')
print('Results saved in:', OUTPUT_DIR)
