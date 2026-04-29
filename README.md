# CODA - Computational Pathology Pipeline

A two-module pipeline for H&E histopathology whole-slide image (WSI) analysis:

1. **Tissue Segmentation** - annotate, tile, and train a ConvNeXt-UNet model for semantic segmentation
2. **Serial Section Alignment** - affine + elastic registration of consecutive tissue sections

---

## Overview

```
Raw WSIs (.ndpi)
│
├─── Annotate in Aperio ImageScope → export XML
│         ↓
│    Generate large mosaics (paired H&E image + annotation masks)
│         ↓
│    Random-crop into 256×256 tiles → .npz datasets
│         ↓
│    Train ConvNeXtUNet (semantic segmentation, N classes)
│         ↓
│    Run inference on new WSIs (single or batch)
│
└─── Alignment: register serial sections (affine → elastic)
          ↓
     Aligned images, tissue masks, transform files
```

---

## Repository Structure

```
CODA/
├── tissue_segmentation/
│   ├── src/
│   │   ├── config.py              # Training hyperparameters & paths
│   │   ├── model.py               # ConvNeXt-UNet architecture
│   │   ├── dataset.py             # Data loading, augmentation, class weighting
│   │   └── train.py               # Training loop, metrics, checkpointing
│   ├── inference/
│   │   ├── wsi_inference.py       # Inference on a single WSI
│   │   └── batch_wsi_inference.py # Batch inference over a directory
│   └── tile_generation/
│       └── util.py                # Annotation parsing, tile extraction, mosaic assembly
└── alignment/
    ├── alignment_pipeline.py      # Orchestration: sorting, anchoring, pass logic
    └── registration.py            # Affine (ORB/ECC) + elastic (phase-correlation) registration
```

---

## Module 1: Tissue Segmentation

### Step 1 — Annotate WSIs

Draw region annotations in **Aperio ImageScope** on `.ndpi` files. Each annotation layer corresponds to a tissue class. Export annotations as XML; the XML filename must match the WSI filename it was drawn on.

### Step 2 — Generate Tiles

`tile_generation/util.py` provides the building blocks used to go from WSI + XML → training data:

- **`read_annotations_xml()`** — parses Aperio XML into labeled polygon lists
- **`cluster_annotations()`** — groups overlapping annotations via Union-Find
- **`build_mosaic()`** — stochastically assembles extracted tiles onto a large canvas (default 10240×10240), handling rotation and overlap penalties to maximize fill
- **`make_segmentation_mask()`** — rasterizes polygons into integer label maps

The output is `.npz` files containing:
- `he` — uint8 RGB image tiles
- `masks` — integer label arrays (0 = background, 1–9 = tissue classes)

### Step 3 — Train

Edit `src/config.py` to point at your `.npz` dataset files, then:

```bash
cd tissue_segmentation
python src/train.py
```

Key config options:

| Parameter | Default | Description |
|---|---|---|
| `BACKBONE` | `convnext_tiny` | ConvNeXt variant (`tiny`, `base`, `large`) |
| `NUM_CLASSES` | `10` | Number of segmentation classes (including background) |
| `TILE_SIZE` | `256` | Input tile size (pixels) |
| `BATCH_SIZE` | `4` | Training batch size |
| `LR` | `1e-4` | Initial learning rate |
| `MAX_EPOCHS` | `100` | Maximum training epochs |
| `EARLY_STOP_PATIENCE` | `15` | Epochs without improvement before stopping |

Training uses:
- **Loss**: CrossEntropyLoss with inverse-frequency class weighting (background excluded)
- **Scheduler**: Linear warmup (5 epochs) + cosine annealing
- **AMP**: Mixed precision enabled by default
- **Augmentation**: Flips, rotation, color jitter, Gaussian blur, elastic deformation (albumentations)

Checkpoints are written to `src/checkpoints/` — best model by validation Dice score plus periodic saves every 10 epochs.

### Step 4 — Inference

**Single WSI:**

```bash
python inference/wsi_inference.py \
    --ndpi /path/to/slide.ndpi \
    --checkpoint src/checkpoints/best_model.pth \
    --out /path/to/output/
```

**Batch (directory of `.ndpi` files):**

```bash
python inference/batch_wsi_inference.py \
    --folder /path/to/slides/ \
    --checkpoint src/checkpoints/best_model.pth \
    --out /path/to/output/
```

Inference pipeline:
1. Green-channel tissue masking to skip background
2. 1024×1024 tiles with 128-pixel overlap
3. Gaussian-weighted blending to eliminate tile boundary artifacts
4. Output: integer label map + optional colormap PNG

Key inference flags:

| Flag | Default | Description |
|---|---|---|
| `--level` | `0` | WSI pyramid level to read |
| `--tile-size` | `1024` | Inference tile size |
| `--overlap` | `128` | Overlap between tiles |
| `--scale` | `4` | Output downscale factor |
| `--no-colormap` | — | Skip colormap PNG output |

---

## Module 2: Serial Section Alignment

Registers consecutive H&E sections from serially cut tissue blocks.

### Usage

```bash
python alignment/alignment_pipeline.py \
    --input /path/to/slides/ \
    --output /path/to/aligned/ \
    --level 3
```

### Pipeline Logic

1. Slides are discovered and sorted by trailing number in filename
2. The middle slide is selected as the anchor (no transform applied)
3. A **forward pass** registers slides from middle+1 → end
4. A **backward pass** registers slides from middle-1 → 0
5. Each slide registers to the most recent successfully-aligned neighbor
6. If registration IoU falls below `--iou-threshold`, the pipeline skips up to `--max-skip` slides (damage handling)

### Registration

Two-stage per pair:

1. **Affine** — choice of method via `--method`:
   - `orb`: ORB feature matching → RANSAC affine
   - `ecc`: Enhanced Correlation Coefficient (intensity-based)
   - `combined` *(default)*: ORB coarse initialization → multi-scale ECC refinement

2. **Elastic** — sparse phase-correlation on a grid of patches → Gaussian-smoothed displacement field → thin-plate-spline interpolation to dense field

### Output

```
output/
├── aligned/          # Registered PNG images
├── masks/            # Tissue masks
├── transforms/
│   ├── affine/       # Affine matrices (.pkl, 2×3)
│   └── elastic/      # Displacement fields (.npy, H×W×2)
└── alignment_results.json
```

Key alignment flags:

| Flag | Default | Description |
|---|---|---|
| `--level` | `3` | Pyramid level for registration |
| `--method` | `combined` | Affine method: `orb`, `ecc`, `combined` |
| `--iou-threshold` | `0.5` | Minimum tissue overlap to accept registration |
| `--max-skip` | `3` | Max consecutive slides to skip when looking for a better match |
| `--padding` | `200` | White border added to anchor image |

---

## Dependencies

```
torch
torchvision
albumentations
opencv-python
scikit-image
scipy
pyvips
Pillow
tifffile
numpy
matplotlib
tqdm
```

Install with:

```bash
pip install torch torchvision albumentations opencv-python scikit-image scipy pyvips Pillow tifffile numpy matplotlib tqdm
```

> **Note**: `pyvips` requires the libvips system library. On Ubuntu/Debian: `sudo apt install libvips-dev`

---

## Data Formats

| Format | Used for |
|---|---|
| `.ndpi` | Whole-slide images (Hamamatsu; pyvips/OpenSlide compatible) |
| Aperio ImageScope XML | Polygon annotations drawn on WSIs |
| `.npz` | Packed tile datasets (`he`, `masks` arrays) |
| `.pth` | PyTorch model checkpoints |
| `.pkl` | Affine transform matrices |
| `.npy` | Elastic displacement fields |
| `.json` | Alignment run metadata |
