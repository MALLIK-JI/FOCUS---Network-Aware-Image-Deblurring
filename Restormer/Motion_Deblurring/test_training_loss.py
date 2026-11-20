
import torch
from basicsr.models.losses import CombinedDeblurLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = CombinedDeblurLoss().to(device)

pred = torch.rand(1, 3, 128, 128, device=device, requires_grad=True)
target = torch.rand(1, 3, 128, 128, device=device)

loss = criterion(pred, target)
loss.backward()

print("Device:", device)
print("Total loss:", loss.item())
print("Gradient available:", pred.grad is not None)
print("Sobel filter device:", criterion.sobel_x.device)

assert torch.isfinite(loss), "Loss is not finite!"
assert pred.grad is not None, "Gradient missing!"

print("TRAINING LOSS TEST PASSED")
