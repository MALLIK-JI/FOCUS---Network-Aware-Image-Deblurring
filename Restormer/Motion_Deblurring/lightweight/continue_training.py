import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import os
import torch

from torch.utils.data import DataLoader
from train_lightweight import (
    prepare_datasets,
    validate_model,
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE
)
from model import LightweightDeblurCNN
from basicsr.models.losses.custom_loss import CombinedDeblurLoss

EPOCHS = 15

INITIAL_CHECKPOINT = (
    "./lightweight/checkpoints_random/lightweight_best.pth"
)

CHECKPOINT_DIR = "./lightweight/checkpoints_continued"


def main():

    print("Training device:", DEVICE)

    train_dataset, validation_dataset = prepare_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    model = LightweightDeblurCNN().to(DEVICE)

    checkpoint = torch.load(
        INITIAL_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    print("Loaded best checkpoint from initial training.")

    criterion = CombinedDeblurLoss(
        charbonnier_weight=0.5,
        ssim_weight=0.3,
        edge_weight=0.2
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_psnr = -float("inf")

    best_path = os.path.join(
        CHECKPOINT_DIR,
        "lightweight_continued_best.pth"
    )

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0.0

        for blurred, sharp in train_loader:

            blurred = blurred.to(DEVICE)
            sharp = sharp.to(DEVICE)

            optimizer.zero_grad()

            restored = model(blurred)
            loss = criterion(restored, sharp)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        val_mse, val_psnr, val_ssim = validate_model(
            model,
            validation_loader
        )

        scheduler.step(val_psnr)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nContinuation epoch {epoch + 1}/{EPOCHS}")
        print(f"Training loss: {average_loss:.6f}")
        print(f"Validation MSE: {val_mse:.6f}")
        print(f"Validation PSNR: {val_psnr:.4f}")
        print(f"Validation SSIM: {val_ssim:.4f}")
        print(f"Learning rate: {current_lr:.8f}")

        epoch_path = os.path.join(
            CHECKPOINT_DIR,
            f"continued_epoch_{epoch + 1}.pth"
        )

        torch.save(model.state_dict(), epoch_path)

        if val_psnr > best_psnr:

            best_psnr = val_psnr

            torch.save(
                model.state_dict(),
                best_path
            )

            print("New best continuation checkpoint saved!")

    print("\nContinuation training completed.")
    print("Best continuation PSNR:", best_psnr)
    print("Best checkpoint:", best_path)


if __name__ == "__main__":
    main()
