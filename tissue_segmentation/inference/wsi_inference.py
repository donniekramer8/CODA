import sys
sys.path.append("../src")

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pyvips
from tqdm import tqdm

import config
from model import build_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path


# ──────────────────── I/O ────────────────────

def read_ndpi(pth: str, level: int = 1) -> np.ndarray:
    img = pyvips.Image.openslideload(pth, level=level)
    if img.format != "uchar":
        img = img.cast("uchar")
    if img.bands not in (3, 4):
        img = img.colourspace("srgb")
    buf = img.write_to_memory()
    arr = np.ndarray(
        buffer=buf,
        dtype=np.uint8,
        shape=(img.height, img.width, img.bands),
    )
    return arr[..., :3]  # always return (H, W, 3)


# ──────────────────── Tissue mask ────────────────────

def get_TA_from_imHE(
    he_image: np.ndarray,
    g_thresh: int = 190,
    scale_factor: int = 3,
    min_background_pixels: int = 100,
) -> np.ndarray:
    print(f'{he_image.shape}  |  g_thresh={g_thresh}  |  scale_factor={scale_factor}  |  min_bg_pixels={min_background_pixels}')
    h, w = he_image.shape[:2]
    img_rgb = he_image[..., :3]

    if scale_factor > 1:
        small = cv2.resize(img_rgb, (w // scale_factor, h // scale_factor),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img_rgb

    print(f'Small image for tissue mask: {small.shape}')

    tissue_small = (small[:, :, 1] < g_thresh).astype(np.uint8)

    background_small = 1 - tissue_small
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background_small, connectivity=8)
    
    # Vectorized: find all small background components in one shot
    area_threshold = min_background_pixels // (scale_factor ** 2)
    areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background label 0
    small_label_ids = np.where(areas < area_threshold)[0] + 1  # +1 to offset label 0
    if small_label_ids.size > 0:
        tissue_small[np.isin(labels, small_label_ids)] = 1

    print(f'Rescaling...')

    if scale_factor > 1:
        tissue_small = cv2.resize(tissue_small, (w, h), interpolation=cv2.INTER_NEAREST)

    return tissue_small.astype(bool)


# ──────────────────── Tiling helpers ────────────────────

def build_tile_coords(
    h: int, w: int, tile_size: int, overlap: int
) -> list[tuple[int, int]]:
    """Return (y, x) top-left corners for tiles with the given overlap."""
    stride = tile_size - overlap
    coords = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            coords.append((min(y, h - tile_size), min(x, w - tile_size)))
    # deduplicate while preserving order
    seen = set()
    unique = []
    for c in coords:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def gaussian_weight_map(tile_size: int) -> np.ndarray:
    """2-D Gaussian that down-weights tile edges — kills the grid artefact."""
    sigma = tile_size / 4.0
    ax = np.arange(tile_size) - tile_size / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel.astype(np.float32)          # (tile_size, tile_size)


# ──────────────────── Inference ────────────────────

def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        variant=config.CONVNEXT_VARIANT,
        pretrained=False,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}  val_dice={ckpt['val_dice']:.4f}")
    return model


preprocess = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


@torch.no_grad()
def infer_wsi(
    image: np.ndarray,
    tissue_mask: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 1024,
    overlap: int = 128,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Tile the WSI, run inference on tissue tiles, and blend predictions back
    using a global Gaussian-weighted accumulator (no strip boundaries →
    perfectly smooth blending everywhere).

    Returns:
        prediction map (H, W) int32 with class indices.
    """
    H, W = image.shape[:2]
    num_classes = config.NUM_CLASSES
    stride = tile_size - overlap
    gauss = gaussian_weight_map(tile_size)  # (T, T)

    # Global accumulators — RAM cost: num_classes * H * W * 4 bytes
    logit_acc  = np.zeros((num_classes, H, W), dtype=np.float32)
    weight_acc = np.zeros((H, W),              dtype=np.float32)

    # All tile top-left corners in row-major order
    y_starts = sorted(set(min(y, H - tile_size) for y in range(0, H, stride)))
    x_starts = sorted(set(min(x, W - tile_size) for x in range(0, W, stride)))
    all_coords = [(y, x) for y in y_starts for x in x_starts]

    # Filter to tissue tiles
    tissue_coords = [
        (y, x) for y, x in all_coords
        if tissue_mask[y:y + tile_size, x:x + tile_size].any()
    ]
    skipped = len(all_coords) - len(tissue_coords)
    print(f"Total tiles: {len(all_coords)} | tissue: {len(tissue_coords)} | skipped (bg): {skipped}")

    for b_start in tqdm(range(0, len(tissue_coords), batch_size), desc="Inference"):
        batch = tissue_coords[b_start: b_start + batch_size]
        tiles = []
        for y, x in batch:
            tile = image[y:y + tile_size, x:x + tile_size]
            ph = tile_size - tile.shape[0]
            pw = tile_size - tile.shape[1]
            if ph > 0 or pw > 0:
                tile = np.pad(tile, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            tiles.append(preprocess(image=tile)["image"])

        tensor = torch.stack(tiles).to(device)
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()  # (B, C, T, T)

        for i, (y, x) in enumerate(batch):
            th = min(tile_size, H - y)
            tw = min(tile_size, W - x)
            w  = gauss[:th, :tw]
            logit_acc[:, y:y + th, x:x + tw] += probs[i, :, :th, :tw] * w[np.newaxis]
            weight_acc[y:y + th,   x:x + tw] += w

    # Normalise and argmax
    valid = weight_acc > 0
    weight_acc[~valid] = 1.0          # avoid divide-by-zero in background
    logit_acc /= weight_acc[np.newaxis]
    pred = logit_acc.argmax(axis=0).astype(np.int32)
    pred[~tissue_mask] = 0
    return pred


# ──────────────────── Core pipeline (importable) ────────────────────

COLORS = [
    "#000000",  # 0 background
    "#e6194b",  # 1
    "#3cb44b",  # 2
    "#4363d8",  # 3
    "#f58231",  # 4
    "#911eb4",  # 5
    "#42d4f4",  # 6
    "#f032e6",  # 7
    "#bfef45",  # 8
    "#fabed4",  # 9
]


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def pred_to_colormap(pred: np.ndarray, num_classes: int) -> np.ndarray:
    palette = [hex_to_bgr(c) for c in COLORS[:num_classes]]
    color_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for cls_idx, bgr in enumerate(palette):
        color_img[pred == cls_idx] = bgr
    return color_img


def process_single_wsi(
    ndpi_path: str,
    checkpoint_path: str,
    out_dir: str,
    device: torch.device,
    level: int = 1,
    tile_size: int = None,
    overlap: int = 64,
    batch_size: int = 4,
    g_thresh: int = 190,
    save_scale: int = 4,
    save_colormap: bool = True,
) -> None:
    """Run inference on a single NDPI file and save outputs to out_dir.

    Args:
        ndpi_path:      Path to the .ndpi file.
        checkpoint_path: Path to best_model.pth.
        out_dir:        Directory to write outputs into.
        device:         torch.device.
        level:          OpenSlide pyramid level.
        tile_size:      Inference tile size (defaults to config.TILE_SIZE).
        overlap:        Tile overlap in pixels.
        batch_size:     Inference batch size.
        g_thresh:       Green-channel tissue threshold.
        save_scale:     Downscale factor for saved images (1 = full res).
        save_colormap:  If True, also save a colourmap PNG alongside the label map.
    """
    tile_size = tile_size or config.TILE_SIZE
    stem = Path(ndpi_path).stem
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {ndpi_path}")

    image = read_ndpi(ndpi_path, level=level)
    print(f"  Image shape: {image.shape}")

    tissue_mask = get_TA_from_imHE(image, g_thresh=g_thresh, scale_factor=3)
    print(f"  Tissue coverage: {tissue_mask.mean()*100:.1f}%")

    model = load_model(checkpoint_path, device)

    pred = infer_wsi(
        image=image,
        tissue_mask=tissue_mask,
        model=model,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
    )

    # Downscale if requested
    if save_scale > 1:
        H, W = pred.shape
        out_h, out_w = max(1, H // save_scale), max(1, W // save_scale)
        pred_save = cv2.resize(pred.astype(np.uint8), (out_w, out_h),
                               interpolation=cv2.INTER_NEAREST).astype(np.int32)
        print(f"  Saving at 1/{save_scale} scale: {W}×{H} → {out_w}×{out_h}")
    else:
        pred_save = pred

    # Save raw label map
    raw_path = os.path.join(out_dir, f"{stem}.png")
    cv2.imwrite(raw_path, pred_save.astype(np.uint8))
    print(f"  Saved label map  → {raw_path}")

    # Optionally save colourmap
    if save_colormap:
        color_img = pred_to_colormap(pred_save, config.NUM_CLASSES)
        color_path = os.path.join(out_dir, f"{stem}_colormap.png")
        cv2.imwrite(color_path, color_img)
        print(f"  Saved colormap   → {color_path}")


# ──────────────────── CLI (single file) ────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="WSI inference with ConvNeXt-UNet")
    parser.add_argument("--ndpi",       required=True,  help="Path to .ndpi file")
    parser.add_argument("--checkpoint", required=True,  help="Path to best_model.pth")
    parser.add_argument("--out",        required=True,  help="Output directory for prediction maps")
    parser.add_argument("--level",      type=int, default=1)
    parser.add_argument("--tile_size",  type=int, default=config.TILE_SIZE)
    parser.add_argument("--overlap",    type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--g_thresh",   type=int, default=190)
    parser.add_argument("--save_scale", type=int, default=4,
                        help="Downscale factor for saved images (default: 4).")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    process_single_wsi(
        ndpi_path=args.ndpi,
        checkpoint_path=args.checkpoint,
        out_dir=args.out,
        device=device,
        level=args.level,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        g_thresh=args.g_thresh,
        save_scale=args.save_scale,
        save_colormap=True,
    )


if __name__ == "__main__":
    main()

# python wsi_inference.py \
#     --ndpi        /path/to/slide.ndpi \
#     --checkpoint  checkpoints/best_model.pth \
#     --out         /path/to/output_dir \
#     --level       1
