import torch
from basicsr.models.losses import CombinedDeblurLoss

criterion = CombinedDeblurLoss()

pred = torch.rand(1, 3, 128, 128, requires_grad=True)
target = torch.rand(1, 3, 128, 128)

loss = criterion(pred, target)

loss.backward()

print("Combined Loss:", loss.item())
print("Gradient available:", pred.grad is not None)
print("Custom loss test passed!")