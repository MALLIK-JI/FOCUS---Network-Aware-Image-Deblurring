
import os
import random
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

try:
    from .model import LightweightDeblurCNN
except ImportError:
    from model import LightweightDeblurCNN
from basicsr.models.losses.custom_loss import CombinedDeblurLoss


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = "./Datasets/train/GoPro/input_crops"
TARGET_DIR = "./Datasets/train/GoPro/target_crops"

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4

MAX_IMAGES = 5000
VALIDATION_RATIO = 0.15
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = "./lightweight/checkpoints_random"


# ==========================================
# REPRODUCIBILITY
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# DATASET
# ==========================================

class GoProDataset(Dataset):

    def __init__(self, input_dir, target_dir, files):

        self.input_dir = input_dir
        self.target_dir = target_dir
        self.files = files

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        blurred = Image.open(
            os.path.join(self.input_dir, filename)
        ).convert("RGB")

        sharp = Image.open(
            os.path.join(self.target_dir, filename)
        ).convert("RGB")

        return (
            self.transform(blurred),
            self.transform(sharp)
        )


# ==========================================
# GROUP-AWARE DATA SPLIT
# ==========================================

def get_source_group(filename):

    # Remove crop suffix such as _0, _1, _10
    return filename.rsplit("_", 1)[0]


def prepare_datasets():

    input_files = set(os.listdir(INPUT_DIR))
    target_files = set(os.listdir(TARGET_DIR))

    if input_files != target_files:
        raise ValueError(
            "Input and target filenames do not match!"
        )

    all_files = sorted(input_files)

    # Select random image pairs from the full dataset
    rng = random.Random(SEED)

    selected_files = rng.sample(
        all_files,
        min(MAX_IMAGES, len(all_files))
    )

    # Group crops belonging to the same source frame
    groups = {}

    for filename in selected_files:
        group = get_source_group(filename)
        groups.setdefault(group, []).append(filename)

    group_names = sorted(groups.keys())
    rng.shuffle(group_names)

    validation_group_count = max(
        1,
        int(len(group_names) * VALIDATION_RATIO)
    )

    validation_groups = set(
        group_names[:validation_group_count]
    )

    train_files = []
    validation_files = []

    for group, filenames in groups.items():

        if group in validation_groups:
            validation_files.extend(filenames)
        else:
            train_files.extend(filenames)

    train_files.sort()
    validation_files.sort()

    train_dataset = GoProDataset(
        INPUT_DIR,
        TARGET_DIR,
        train_files
    )

    validation_dataset = GoProDataset(
        INPUT_DIR,
        TARGET_DIR,
        validation_files
    )

    return train_dataset, validation_dataset


# ==========================================
# VALIDATION
# ==========================================

def validate_model(model, loader):

    model.eval()

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_images = 0

    with torch.no_grad():

        for blurred, sharp in loader:

            blurred = blurred.to(DEVICE)
            sharp = sharp.to(DEVICE)

            restored = model(blurred)

            restored = torch.clamp(restored, 0, 1)

            restored_np = (
                restored.cpu().numpy()
                .transpose(0, 2, 3, 1)
            )

            sharp_np = (
                sharp.cpu().numpy()
                .transpose(0, 2, 3, 1)
            )

            for pred, target in zip(
                restored_np, sharp_np
            ):

                mse = np.mean(
                    (pred - target) ** 2
                )

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

                total_mse += mse
                total_psnr += psnr
                total_ssim += ssim_value

                total_images += 1

    return (
        total_mse / total_images,
        total_psnr / total_images,
        total_ssim / total_images
    )


# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================

def main():

    print("Training device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # Prepare datasets
    train_dataset, validation_dataset = (
        prepare_datasets()
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))

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

    # Create fresh model
    model = LightweightDeblurCNN().to(DEVICE)

    # Custom loss function
    criterion = CombinedDeblurLoss(
        charbonnier_weight=0.5,
        ssim_weight=0.3,
        edge_weight=0.2
    ).to(DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    best_psnr = -float("inf")

    best_checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "lightweight_best.pth"
    )

    # ======================================
    # TRAINING LOOP
    # ======================================

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for blurred, sharp in progress:

            blurred = blurred.to(DEVICE)
            sharp = sharp.to(DEVICE)

            optimizer.zero_grad()

            restored = model(blurred)

            loss = criterion(
                restored,
                sharp
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.5f}"
            )

        average_loss = (
            total_loss / len(train_loader)
        )

        # Validate after every epoch
        val_mse, val_psnr, val_ssim = validate_model(
            model,
            validation_loader
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Training loss: {average_loss:.6f}")
        print(f"Validation MSE: {val_mse:.6f}")
        print(f"Validation PSNR: {val_psnr:.4f}")
        print(f"Validation SSIM: {val_ssim:.4f}")

        # Update learning rate
        scheduler.step(val_psnr)

        # Save epoch checkpoint
        epoch_path = os.path.join(
            CHECKPOINT_DIR,
            f"lightweight_epoch_{epoch + 1}.pth"
        )

        torch.save(
            model.state_dict(),
            epoch_path
        )

        # Save best checkpoint
        if val_psnr > best_psnr:

            best_psnr = val_psnr

            torch.save(
                model.state_dict(),
                best_checkpoint_path
            )

            print("New best model saved!")

        print("Epoch checkpoint saved:", epoch_path)

    print("\n================================")
    print("TRAINING COMPLETED")
    print("================================")

    print("Best validation PSNR:", best_psnr)
    print("Best checkpoint:", best_checkpoint_path)


if __name__ == "__main__":
    main()
