import os
import glob
import yaml
import copy
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.losses.custom_loss import CombinedDeblurLoss
import utils
CONFIG_PATH = './Options/Deblurring_Restormer.yml'
CHECKPOINT_PATH = './pretrained_models/motion_deblurring.pth'
INPUT_DIR = './Datasets/val/GoPro/input_crops'
TARGET_DIR = './Datasets/val/GoPro/target_crops'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
network_config = copy.deepcopy(config['network_g'])
network_config.pop('type')
model = Restormer(**network_config)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['params'], strict=True)
model = model.to(DEVICE)
model.eval()
print('Pretrained Restormer loaded successfully.')
input_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.png')) + glob.glob(os.path.join(INPUT_DIR, '*.jpg')))
target_files = sorted(glob.glob(os.path.join(TARGET_DIR, '*.png')) + glob.glob(os.path.join(TARGET_DIR, '*.jpg')))
target_lookup = {os.path.basename(path): path for path in target_files}
pairs = [(path, target_lookup[os.path.basename(path)]) for path in input_files if os.path.basename(path) in target_lookup]
if not pairs:
    raise RuntimeError('No matching input and target images found.')
(input_path, target_path) = pairs[0]
print('Input image:', input_path)
print('Target image:', target_path)
input_image = np.float32(utils.load_img(input_path)) / 255.0
target_image = np.float32(utils.load_img(target_path)) / 255.0
input_tensor = torch.from_numpy(input_image).permute(2, 0, 1)
target_tensor = torch.from_numpy(target_image).permute(2, 0, 1)
input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
target_tensor = target_tensor.unsqueeze(0).to(DEVICE)
if input_tensor.shape != target_tensor.shape:
    raise RuntimeError('Input and target image dimensions do not match.')
(height, width) = input_tensor.shape[-2:]
pad_h = (8 - height % 8) % 8
pad_w = (8 - width % 8) % 8
padded_input = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')
with torch.enable_grad():
    prediction = model(padded_input)
    if isinstance(prediction, list):
        prediction = prediction[-1]
    prediction = prediction[:, :, :height, :width]
    prediction = prediction.detach().requires_grad_(True)
loss_function = CombinedDeblurLoss(charbonnier_weight=0.5, ssim_weight=0.3, edge_weight=0.2, eps=0.001).to(DEVICE)
loss_char = loss_function.charbonnier_loss(prediction, target_tensor)
loss_ssim = loss_function.ssim_loss(prediction, target_tensor)
loss_edge = loss_function.edge_loss(prediction, target_tensor)
weights = {'Charbonnier': 0.5, 'SSIM': 0.3, 'Edge': 0.2}
losses = {'Charbonnier': loss_char, 'SSIM': loss_ssim, 'Edge': loss_edge}
weighted_losses = {name: losses[name] * weights[name] for name in losses}
total_loss = sum(weighted_losses.values())
print('\n========== LOSS COMPONENTS ==========')
for name in losses:
    raw = losses[name].item()
    weighted = weighted_losses[name].item()
    percentage = weighted / total_loss.item() * 100 if abs(total_loss.item()) > 1e-12 else float('nan')
    print(f'\n{name}')
    print(f'Raw loss:       {raw:.8f}')
    print(f'Weight:         {weights[name]:.2f}')
    print(f'Weighted loss:  {weighted:.8f}')
    print(f'Contribution:   {percentage:.2f}%')
print('\nTotal combined loss:', total_loss.item())
print('\n========== GRADIENT ANALYSIS ==========')
for name in losses:
    raw_gradient = torch.autograd.grad(losses[name], prediction, retain_graph=True)[0]
    weighted_gradient = raw_gradient * weights[name]
    raw_norm = torch.linalg.vector_norm(raw_gradient).item()
    weighted_norm = torch.linalg.vector_norm(weighted_gradient).item()
    print(f'\n{name}')
    print(f'Raw gradient norm:      {raw_norm:.8f}')
    print(f'Weighted gradient norm: {weighted_norm:.8f}')
print('\n========== DIAGNOSTIC COMPLETE ==========')
print('No model parameters were updated.')
