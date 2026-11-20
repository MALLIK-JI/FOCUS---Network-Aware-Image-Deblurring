import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from model import LightweightDeblurCNN
from train_lightweight import prepare_datasets


DEVICE = torch.device("cpu")

CHECKPOINT = (
    "./lightweight/checkpoints_random/lightweight_best.pth"
)

# Load validation dataset
_, validation_dataset = prepare_datasets()

loader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# Load trained model
model = LightweightDeblurCNN().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.eval()

input_metrics = []
restored_metrics = []


def calculate_metrics(pred, target):

    mse = np.mean((pred - target) ** 2)

    psnr = (
        float("inf")
        if mse == 0
        else 10 * np.log10(1.0 / mse)
    )

    ssim_value = ssim(
        target,
        pred,
        channel_axis=2,
        data_range=1.0
    )

    return mse, psnr, ssim_value


with torch.no_grad():

    for blurred, sharp in loader:

        blurred = blurred.to(DEVICE)
        sharp = sharp.to(DEVICE)

        restored = model(blurred)
        restored = torch.clamp(restored, 0, 1)

        blurred_np = blurred.numpy().transpose(0, 2, 3, 1)
        sharp_np = sharp.numpy().transpose(0, 2, 3, 1)
        restored_np = restored.numpy().transpose(0, 2, 3, 1)

        for inp, pred, target in zip(
            blurred_np,
            restored_np,
            sharp_np
        ):

            input_metrics.append(
                calculate_metrics(inp, target)
            )

            restored_metrics.append(
                calculate_metrics(pred, target)
            )


input_avg = np.mean(input_metrics, axis=0)
restored_avg = np.mean(restored_metrics, axis=0)

print("\n========== VALIDATION COMPARISON ==========")

print("\nBlurred Input:")
print("MSE:", input_avg[0])
print("PSNR:", input_avg[1])
print("SSIM:", input_avg[2])

print("\nLightweight CNN:")
print("MSE:", restored_avg[0])
print("PSNR:", restored_avg[1])
print("SSIM:", restored_avg[2])

print("\n========== IMPROVEMENT ==========")

print("PSNR improvement:", restored_avg[1] - input_avg[1])
print("SSIM improvement:", restored_avg[2] - input_avg[2])
print("MSE reduction:", input_avg[0] - restored_avg[0])
