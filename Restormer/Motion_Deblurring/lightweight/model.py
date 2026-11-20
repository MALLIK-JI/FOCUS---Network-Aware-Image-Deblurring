
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class LightweightDeblurCNN(nn.Module):
    def __init__(self, channels=32, num_blocks=5):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.output_conv = nn.Conv2d(
            channels, 3, kernel_size=3, padding=1
        )

    def forward(self, x):
        features = self.input_conv(x)
        features = self.residual_blocks(features)

        residual = self.output_conv(features)

        return torch.clamp(x + residual, 0, 1)


if __name__ == "__main__":
    model = LightweightDeblurCNN()

    sample = torch.rand(1, 3, 128, 128)
    output = model(sample)

    parameters = sum(p.numel() for p in model.parameters())

    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Parameters:", parameters)
    print("Model test passed!")
