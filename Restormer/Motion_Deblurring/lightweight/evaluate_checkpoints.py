
import os
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from model import LightweightDeblurCNN

DEVICE = torch.device("cpu")

INPUT_DIR = "./Datasets/test/GoPro/input"
GT_DIR = "./Datasets/test/GoPro/ground_truth"
CHECKPOINT_DIR = "./lightweight/checkpoints"

def calculate_metrics(pred, gt):
    pred = np.array(pred).astype(np.uint8)
    gt = np.array(gt).astype(np.uint8)

    mse = np.mean((pred.astype(np.float64) -
                   gt.astype(np.float64)) ** 2)

    psnr = float("inf") if mse == 0 else (
        10 * np.log10((255 ** 2) / mse)
    )

    ssim_value = ssim(
        gt, pred,
        channel_axis=2,
        data_range=255
    )

    return mse, psnr, ssim_value


def main():
    model = LightweightDeblurCNN().to(DEVICE)
    model.eval()

    test_files = sorted([
        f for f in os.listdir(GT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    results = []

    for epoch in range(1, 11):
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"lightweight_epoch_{epoch}.pth"
        )

        if not os.path.isfile(checkpoint_path):
            print("Missing checkpoint:", checkpoint_path)
            continue

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        model.load_state_dict(checkpoint)

        epoch_metrics = []

        for filename in test_files:
            input_path = os.path.join(INPUT_DIR, filename)
            gt_path = os.path.join(GT_DIR, filename)

            if not os.path.isfile(input_path):
                print("Missing input:", filename)
                continue

            blurred = Image.open(input_path).convert("RGB")
            gt = Image.open(gt_path).convert("RGB")

            input_array = np.array(blurred).astype(np.float32) / 255.0

            tensor = torch.from_numpy(
                input_array.transpose(2, 0, 1)
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                restored = model(tensor)

            output = restored.squeeze(0).cpu().numpy()
            output = np.clip(output.transpose(1, 2, 0), 0, 1)
            output_image = Image.fromarray(
                (output * 255).round().astype(np.uint8)
            )

            metrics = calculate_metrics(output_image, gt)
            epoch_metrics.append(metrics)

        if epoch_metrics:
            average = np.mean(epoch_metrics, axis=0)
            results.append((epoch, *average))

            print(
                f"Epoch {epoch:2d} | "
                f"MSE: {average[0]:.4f} | "
                f"PSNR: {average[1]:.4f} | "
                f"SSIM: {average[2]:.4f}"
            )

    print("\n========== BEST CHECKPOINTS ==========")

    if results:
        best_psnr = max(results, key=lambda x: x[2])
        best_ssim = max(results, key=lambda x: x[3])
        best_mse = min(results, key=lambda x: x[1])

        print("Best PSNR:", best_psnr)
        print("Best SSIM:", best_ssim)
        print("Best MSE:", best_mse)


if __name__ == "__main__":
    main()
