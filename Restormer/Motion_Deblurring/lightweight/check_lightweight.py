
import torch
from PIL import Image
from torchvision import transforms

from model import LightweightDeblurCNN

device = torch.device("cpu")

model = LightweightDeblurCNN().to(device)

checkpoint = torch.load(
    "./lightweight/checkpoints/lightweight_deblur.pth",
    map_location=device
)

model.load_state_dict(checkpoint)
model.eval()

image = Image.open(
    "./Datasets/test/GoPro/input/image1.jpg"
).convert("RGB")

transform = transforms.ToTensor()
input_image = transform(image).unsqueeze(0)

with torch.no_grad():
    output_image = model(input_image)

difference = torch.mean(
    torch.abs(output_image - input_image)
).item()

print("Input mean:", input_image.mean().item())
print("Output mean:", output_image.mean().item())
print("Mean absolute change:", difference)

print("Diagnostic completed!")
