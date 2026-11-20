import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class CombinedDeblurLoss(nn.Module):

    def __init__(self, charbonnier_weight=0.5, ssim_weight=0.3, edge_weight=0.2, eps=0.001):
        super().__init__()
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.eps = eps
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def charbonnier_loss(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

    def ssim_loss(self, pred, target):
        mu_x = F.avg_pool2d(pred, 3, 1, 1)
        mu_y = F.avg_pool2d(target, 3, 1, 1)
        sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        ssim = numerator / (denominator + 1e-08)
        return 1 - ssim.mean()

    def edge_loss(self, pred, target):
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        pred_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        pred_y = F.conv2d(pred_gray, self.sobel_y, padding=1)
        target_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)

    def forward(self, pred, target):
        loss_char = self.charbonnier_loss(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        loss_edge = self.edge_loss(pred, target)
        total_loss = self.charbonnier_weight * loss_char + self.ssim_weight * loss_ssim + self.edge_weight * loss_edge
        return total_loss
