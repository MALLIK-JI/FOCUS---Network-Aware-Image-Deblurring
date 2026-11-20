
import torch
from torch.utils.data import DataLoader

from model import LightweightDeblurCNN
from train_lightweight import GoProDataset
from basicsr.models.losses.custom_loss import CombinedDeblurLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = GoProDataset(
    "./Datasets/train/GoPro/input_crops",
    "./Datasets/train/GoPro/target_crops",
    max_images=8
)

loader = DataLoader(dataset, batch_size=8, shuffle=False)

model = LightweightDeblurCNN().to(DEVICE)

checkpoint = torch.load(
    "./lightweight/checkpoints/lightweight_deblur.pth",
    map_location=DEVICE
)

model.load_state_dict(checkpoint)
model.eval()

criterion = CombinedDeblurLoss().to(DEVICE)

blurred, sharp = next(iter(loader))

blurred = blurred.to(DEVICE)
sharp = sharp.to(DEVICE)

with torch.no_grad():
    restored = model(blurred)

    loss_char = criterion.charbonnier_loss(restored, sharp)
    loss_ssim = criterion.ssim_loss(restored, sharp)
    loss_edge = criterion.edge_loss(restored, sharp)

    total_loss = (
        0.5 * loss_char +
        0.3 * loss_ssim +
        0.2 * loss_edge
    )

print("\nIndividual loss components")
print("---------------------------")
print("Charbonnier loss:", loss_char.item())
print("SSIM loss:", loss_ssim.item())
print("Edge loss:", loss_edge.item())
print("Total loss:", total_loss.item())

print("\nWeighted contributions")
print("----------------------")
print("Charbonnier:", (0.5 * loss_char).item())
print("SSIM:", (0.3 * loss_ssim).item())
print("Edge:", (0.2 * loss_edge).item())
