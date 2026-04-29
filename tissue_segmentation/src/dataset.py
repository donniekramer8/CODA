import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


def get_train_augmentation():
    transforms = []
    if config.RANDOM_FLIP:
        transforms.append(A.HorizontalFlip(p=0.5))
        transforms.append(A.VerticalFlip(p=0.5))
    if config.RANDOM_ROTATE:
        transforms.append(A.RandomRotate90(p=0.5))
    transforms.extend([
        A.ColorJitter(
            brightness=config.COLOR_JITTER_BRIGHTNESS,
            contrast=config.COLOR_JITTER_CONTRAST,
            saturation=config.COLOR_JITTER_SATURATION,
            hue=config.COLOR_JITTER_HUE,
            p=0.8,
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=config.GAUSSIAN_BLUR_PROB),
        A.ElasticTransform(
            alpha=80, sigma=80 * 0.05, p=config.ELASTIC_TRANSFORM_PROB
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return A.Compose(transforms)


def get_val_augmentation():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def sanitize_masks(masks: np.ndarray, num_classes: int) -> np.ndarray:
    """Remap mask values to [0, num_classes-1].
    
    Common cases:
      - Binary masks with values {0, 255} → remap 255 to 1
      - Already {0, 1, ..., C-1} → no change
      - Any value >= num_classes → clamp to num_classes-1
    """
    unique = np.unique(masks)
    print(f"  Mask unique values BEFORE sanitization: {unique}")

    # Common case: {0, 255} binary mask
    if num_classes == 2 and 255 in unique:
        masks = (masks > 0).astype(np.uint8)
    else:
        # General: clip to valid range
        masks = np.clip(masks, 0, num_classes - 1).astype(np.uint8)

    print(f"  Mask unique values AFTER  sanitization: {np.unique(masks)}")
    return masks


class TileDataset(Dataset):
    """Dataset backed by pre-loaded numpy arrays."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (H, W, 3) uint8
        mask = self.masks[idx]    # (H, W) uint8

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.long() if isinstance(mask, torch.Tensor) else torch.from_numpy(np.asarray(mask)).long()
        return image, mask


def compute_class_weights(masks: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute inverse-frequency weights. Background (class 0) is set to 0."""
    counts = np.bincount(masks.ravel(), minlength=num_classes).astype(np.float64)
    print(f"  Per-class pixel counts: {counts}")

    weights = np.zeros(num_classes, dtype=np.float64)
    for c in range(1, num_classes):  # skip background
        if counts[c] > 0:
            weights[c] = 1.0 / counts[c]
    # Normalize so weights sum to num_classes - 1 (number of foreground classes)
    fg_sum = weights.sum()
    if fg_sum > 0:
        weights = weights * (num_classes - 1) / fg_sum

    print(f"  Class weights: {weights}")
    return weights


def build_dataloaders(
    train_path: str = None,
    val_path: str = None,
    batch_size: int = None,
    num_workers: int = None,
    seed: int = None,
):
    train_path  = train_path  or config.TRAIN_DATA_PATH
    val_path    = val_path    or config.VAL_DATA_PATH
    batch_size  = batch_size  or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    seed        = seed if seed is not None else config.SEED

    # Load spatially-split train and val sets
    train_data = np.load(train_path)
    val_data   = np.load(val_path)

    train_images = train_data["he"]     # (N, H, W, 3)
    train_masks  = train_data["masks"]  # (N, H, W)
    val_images   = val_data["he"]
    val_masks    = val_data["masks"]

    train_masks = sanitize_masks(train_masks, config.NUM_CLASSES)
    val_masks   = sanitize_masks(val_masks,   config.NUM_CLASSES)

    # Compute class weights from training masks only
    class_weights = compute_class_weights(train_masks, config.NUM_CLASSES)

    train_dataset = TileDataset(train_images, train_masks, transform=get_train_augmentation())
    val_dataset   = TileDataset(val_images,   val_masks,   transform=get_val_augmentation())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} tiles | Val: {len(val_dataset)} tiles")
    return train_loader, val_loader, class_weights
