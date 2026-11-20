
import os
import cv2
import torch
import yaml
import numpy as np
import torch.nn.functional as F

from basicsr.models.archs.restormer_arch import Restormer

INPUT_DIR = "./Datasets/test/GoPro/input"
CHECKPOINT = "../experiments/Deblurring_Restormer_CustomLoss/models/net_g_100.pth"
OUTPUT_DIR = "./results_custom_loss"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

with open("./Options/Deblurring_Restormer.yml", "r") as f:
    config = yaml.safe_load(f)

network_config = config["network_g"].copy()
network_config.pop("type")

model = Restormer(**network_config).to(device)

checkpoint = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(checkpoint["params"])
model.eval()

image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    raise RuntimeError("No input images found!")

with torch.no_grad():
    for filename in image_files:
        image_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(image_path)

        if image is None:
            print("Could not read:", filename)
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        tensor = torch.from_numpy(image).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).to(device)

        height, width = tensor.shape[2:]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8

        if pad_h or pad_w:
            tensor = F.pad(
                tensor, (0, pad_w, 0, pad_h), mode="reflect"
            )

        restored = model(tensor)
        restored = torch.clamp(restored, 0, 1)
        restored = restored[:, :, :height, :width]

        restored = restored.squeeze(0).permute(1, 2, 0)
        restored = restored.cpu().numpy()
        restored = (restored * 255.0).round().astype(np.uint8)

        restored = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(
            OUTPUT_DIR,
            os.path.splitext(filename)[0] + ".png"
        )

        cv2.imwrite(output_path, restored)
        print("Restored:", output_path)

print("Inference completed!")
