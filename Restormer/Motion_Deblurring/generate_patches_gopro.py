import cv2
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
ROOT = Path('Datasets')
DATASET = Path('C:\\Users\\TANISHK RAJ\\Downloads\\GOPRO_Large')
PATCH_SIZE = 256
OVERLAP = 128
VAL_PATCH_SIZE = 256
NUM_WORKERS = 2

def get_pairs(split):
    pairs = []
    split_dir = DATASET / split
    for sequence in sorted(split_dir.iterdir()):
        if not sequence.is_dir():
            continue
        blur_dir = sequence / 'blur'
        sharp_dir = sequence / 'sharp'
        if not blur_dir.exists() or not sharp_dir.exists():
            continue
        for blur_file in sorted(blur_dir.iterdir()):
            if blur_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
            sharp_file = sharp_dir / blur_file.name
            if sharp_file.exists():
                pairs.append((blur_file, sharp_file, sequence.name))
    return pairs

def process_train(pair):
    (blur_file, sharp_file, sequence) = pair
    blur = cv2.imread(str(blur_file))
    sharp = cv2.imread(str(sharp_file))
    if blur is None or sharp is None:
        return 0
    if blur.shape != sharp.shape:
        return 0
    (h, w) = blur.shape[:2]
    count = 0
    if h < PATCH_SIZE or w < PATCH_SIZE:
        return 0
    ys = list(range(0, h - PATCH_SIZE + 1, PATCH_SIZE - OVERLAP))
    xs = list(range(0, w - PATCH_SIZE + 1, PATCH_SIZE - OVERLAP))
    if ys[-1] != h - PATCH_SIZE:
        ys.append(h - PATCH_SIZE)
    if xs[-1] != w - PATCH_SIZE:
        xs.append(w - PATCH_SIZE)
    for y in ys:
        for x in xs:
            blur_patch = blur[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            sharp_patch = sharp[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            filename = f'{sequence}_{blur_file.stem}_{count}.png'
            cv2.imwrite(str(ROOT / 'train/GoPro/input_crops' / filename), blur_patch)
            cv2.imwrite(str(ROOT / 'train/GoPro/target_crops' / filename), sharp_patch)
            count += 1
    return count

def process_val(pair):
    (blur_file, sharp_file, sequence) = pair
    blur = cv2.imread(str(blur_file))
    sharp = cv2.imread(str(sharp_file))
    if blur is None or sharp is None:
        return 0
    if blur.shape != sharp.shape:
        return 0
    (h, w) = blur.shape[:2]
    if h < VAL_PATCH_SIZE or w < VAL_PATCH_SIZE:
        return 0
    y = (h - VAL_PATCH_SIZE) // 2
    x = (w - VAL_PATCH_SIZE) // 2
    blur_patch = blur[y:y + VAL_PATCH_SIZE, x:x + VAL_PATCH_SIZE]
    sharp_patch = sharp[y:y + VAL_PATCH_SIZE, x:x + VAL_PATCH_SIZE]
    filename = f'{sequence}_{blur_file.stem}.png'
    cv2.imwrite(str(ROOT / 'val/GoPro/input_crops' / filename), blur_patch)
    cv2.imwrite(str(ROOT / 'val/GoPro/target_crops' / filename), sharp_patch)
    return 1
if __name__ == '__main__':
    for folder in [ROOT / 'train/GoPro/input_crops', ROOT / 'train/GoPro/target_crops', ROOT / 'val/GoPro/input_crops', ROOT / 'val/GoPro/target_crops']:
        folder.mkdir(parents=True, exist_ok=True)
    train_pairs = get_pairs('train')
    val_pairs = get_pairs('test')
    print('Training image pairs found:', len(train_pairs))
    print('Validation image pairs found:', len(val_pairs))
    train_counts = Parallel(n_jobs=NUM_WORKERS)((delayed(process_train)(pair) for pair in tqdm(train_pairs, desc='Generating training patches')))
    val_counts = Parallel(n_jobs=NUM_WORKERS)((delayed(process_val)(pair) for pair in tqdm(val_pairs, desc='Generating validation patches')))
    print('\nDataset preparation complete!')
    print('Training patches:', sum(train_counts))
    print('Validation images:', sum(val_counts))
