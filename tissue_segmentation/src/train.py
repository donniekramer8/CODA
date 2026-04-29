import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm


import config
from model import build_model
from dataset import build_dataloaders


# Helpers
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-7):
    """Per-class Dice on foreground pixels only.
    Both preds and targets must already be filtered to foreground (1-D tensors).
    """
    pred_oh   = F.one_hot(preds,   num_classes).float()   # (N_fg, C)
    target_oh = F.one_hot(targets, num_classes).float()
    intersection   = (pred_oh * target_oh).sum(dim=0)     # (C,)
    cardinality    = pred_oh.sum(dim=0) + target_oh.sum(dim=0)
    dice_per_class = ((2.0 * intersection + eps) / (cardinality + eps)).tolist()
    mean_dice      = float(np.mean(dice_per_class[1:]))   # exclude class 0
    return mean_dice, dice_per_class


def iou_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1e-7):
    """Per-class IoU on foreground pixels only.
    Both preds and targets must already be filtered to foreground (1-D tensors).
    """
    pred_oh   = F.one_hot(preds,   num_classes).float()   # (N_fg, C)
    target_oh = F.one_hot(targets, num_classes).float()
    intersection  = (pred_oh * target_oh).sum(dim=0)      # (C,)
    union         = pred_oh.sum(dim=0) + target_oh.sum(dim=0) - intersection
    iou_per_class = ((intersection + eps) / (union + eps)).tolist()
    mean_iou      = float(np.mean(iou_per_class[1:]))     # exclude class 0
    return mean_iou, iou_per_class


def build_scheduler(optimizer, steps_per_epoch: int):
    warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch
    total_steps = config.EPOCHS * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        min_ratio = config.MIN_LR / config.LEARNING_RATE
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Train / Val one epoch
def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    total        = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=config.MIXED_PRECISION):
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VAL)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        total        += images.size(0)
        lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

    return running_loss / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total        = 0
    all_dice     = []
    all_iou      = []

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        with autocast(enabled=config.MIXED_PRECISION):
            logits = model(images)
            loss   = criterion(logits, masks)

        preds = logits.argmax(dim=1)

        # Step 1: zero out predictions where GT is background
        preds_masked = preds.clone()
        preds_masked[masks == 0] = 0

        # Step 2: select only foreground (non-zero GT) pixels for metric computation
        fg_mask        = masks.view(-1) > 0
        preds_fg       = preds_masked.view(-1)[fg_mask]
        targets_fg     = masks.view(-1)[fg_mask]

        if fg_mask.any():
            mean_dice, _ = dice_score(preds_fg, targets_fg, config.NUM_CLASSES)
            mean_iou,  _ = iou_score( preds_fg, targets_fg, config.NUM_CLASSES)
        else:
            mean_dice = mean_iou = 0.0

        all_dice.append(mean_dice)
        all_iou.append(mean_iou)

        running_loss += loss.item() * images.size(0)
        total        += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         dice=f"{mean_dice:.4f}",
                         iou=f"{mean_iou:.4f}")

    return running_loss / total, float(np.mean(all_dice)), float(np.mean(all_iou))


# Early Stopping
class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = None
        self.counter   = 0

    def step(self, metric: float) -> bool:
        if self.best is None or metric > self.best + self.min_delta:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping: no improvement for {self.counter}/{self.patience} epochs")
        return self.counter >= self.patience


# Main
def main():
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, class_weights = build_dataloaders()

    model = build_model(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        variant=config.CONVNEXT_VARIANT,
        pretrained=config.PRETRAINED,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {param_count / 1e6:.2f}M")

    weight_tensor = torch.from_numpy(class_weights).float().to(device)
    print(f"Class weights: {weight_tensor.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler     = build_scheduler(optimizer, len(train_loader))
    scaler        = GradScaler(enabled=config.MIXED_PRECISION)

    save_dir = Path(config.SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_dice     = 0.0
    early_stopper = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
    )

    for epoch in range(1, config.EPOCHS + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
        elapsed    = time.time() - t0
        lr         = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{config.EPOCHS} ({elapsed:.1f}s)  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f}  lr={lr:.2e}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            ckpt_path = save_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou":  val_iou,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  ✓ Saved best model (dice={best_dice:.4f}, iou={val_iou:.4f}) → {ckpt_path}")

        if epoch % 10 == 0:
            ckpt_path = save_dir / f"checkpoint_epoch{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou":  val_iou,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  Saved periodic checkpoint → {ckpt_path}")

        if early_stopper.step(val_dice):
            print(f"\nEarly stopping triggered at epoch {epoch}. Best val dice: {best_dice:.4f}")
            break

    print(f"\nTraining complete. Best val dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
