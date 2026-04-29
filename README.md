# CODA — Computational Pathology Pipeline

A two-module pipeline for H&E histopathology whole-slide image (WSI) analysis:

1. **Tissue Segmentation** — annotate, tile, and train a ConvNeXt-UNet model for semantic segmentation
2. **Serial Section Alignment** — affine + elastic registration of consecutive tissue sections

---

## Overview

```
Raw WSIs (.ndpi)
│
├─── Annotate in Aperio ImageScope → export XML
│         ↓
│    [make_mosaic.ipynb]  Build large mosaic canvas from WSI + XML pairs
│         ↓
│    [make_tiles.ipynb]   Random-crop into tiles → spatially split train/val → .npz
│         ↓
│    [src/train.py]       Train ConvNeXtUNet (semantic segmentation, N classes)
│         ↓
│    [wsi_inference.py / batch_wsi_inference.py]  Run inference on new WSIs
│
└─── [align_serial_sections.ipynb]  Register serial sections (affine → elastic)
          ↓
     [apply_transforms.ipynb]  Warp segmentation masks using saved transforms
```

---

## Repository Structure

```
CODA/
├── tissue_segmentation/
│   ├── src/
│   │   ├── config.py                  # Training hyperparameters & paths
│   │   ├── model.py                   # ConvNeXt-UNet architecture
│   │   ├── dataset.py                 # Data loading, augmentation, class weighting
│   │   └── train.py                   # Training loop, metrics, checkpointing
│   ├── inference/
│   │   ├── wsi_inference.py           # Inference on a single WSI
│   │   ├── batch_wsi_inference.py     # Batch inference over a directory
│   │   └── test.ipynb                 # Visualize model predictions on .npz tiles
│   └── tile_generation/
│       ├── util.py                    # Annotation parsing, tile extraction, mosaic assembly
│       ├── make_mosaic.ipynb          # Step 1: WSI + XML → mosaic canvas
│       └── make_tiles.ipynb           # Step 2: mosaic → random-crop tile dataset
└── alignment/
    ├── alignment_pipeline.py          # Orchestration: sorting, anchoring, pass logic
    ├── registration.py                # Affine (ORB/ECC) + elastic (phase-correlation) registration
    ├── align_serial_sections.ipynb    # Run alignment pipeline interactively
    └── apply_transforms.ipynb         # Warp new images using saved transforms
```

---

## Module 1: Tissue Segmentation

### Step 1 — Annotate WSIs in Aperio ImageScope

Draw region annotations directly on `.ndpi` files in **Aperio ImageScope**. Each annotation layer name corresponds to a tissue class (e.g. `PDAC`, `Duct`, `ECM`, `Islet`). Export the annotations as an XML file — the XML filename must share the same stem as the WSI it was drawn on so the pipeline can auto-pair them.

### Step 2 — Build Mosaic

**Notebook:** [tissue_segmentation/tile_generation/make_mosaic.ipynb](tissue_segmentation/tile_generation/make_mosaic.ipynb)

This notebook takes a directory of matched `.ndpi` / `.xml` pairs and builds a large mosaic canvas containing the annotated regions from all slides.

**What to configure at the top of the notebook:**

| Variable | Description |
|---|---|
| `SLIDE_DIR` | Directory containing matched `.ndpi` + `.xml` pairs (same filename stem) |
| `LEVEL` | WSI pyramid level to read (higher = faster, lower resolution) |
| `CANVAS_SIZE` | Side length of the output mosaic canvas in pixels (default `20480`) |
| `PRIORITY_ORDER` | List of annotation label names, highest priority last (controls label painting order) |

**What it does:**
1. Discovers all `.ndpi` / `.xml` pairs in `SLIDE_DIR`
2. Parses annotation polygons from each XML (`read_annotations_xml`)
3. Clusters overlapping annotations (`cluster_annotations`)
4. Extracts paired H&E image + segmentation mask for each cluster (`read_region`, `make_segmentation_mask`)
5. Stochastically packs all extracted regions onto the mosaic canvas (`build_mosaic`), with rotation and overlap-penalty scoring to maximize fill
6. Saves two files: `mosaic_he.png` (RGB) and `mosaic_mask.png` (integer label map)

### Step 3 — Generate Tile Dataset

**Notebook:** [tissue_segmentation/tile_generation/make_tiles.ipynb](tissue_segmentation/tile_generation/make_tiles.ipynb)

Reads the saved mosaic and randomly crops it into fixed-size tiles with a spatially-guaranteed train/validation split.

**What to configure:**

| Variable | Description |
|---|---|
| `MOSAIC_DIR` | Path to directory containing `mosaic_he.png` and `mosaic_mask.png` |
| `TILE_SIZE` | Tile size in pixels (default `1024`) |
| `MIN_TISSUE_FRAC` | Minimum fraction of tile pixels that must be non-background (default `0.3`) |
| `TARGET_TILES` | Number of training tiles to collect |
| `N_VAL_TILES` | Number of validation tiles (sampled first; training tiles are guaranteed non-overlapping) |

**What it does:**
1. Loads the mosaic pair
2. Computes class-frequency distribution and flags rare classes for oversampling
3. Samples `N_VAL_TILES` non-overlapping tiles for validation
4. Samples up to `TARGET_TILES` training tiles from positions with zero pixel overlap with validation tiles
5. Saves `tiles_train.npz` and `tiles_val.npz` — each contains `he` (uint8 RGB stack) and `masks` (integer label stack)

### Step 4 — Train

Edit [tissue_segmentation/src/config.py](tissue_segmentation/src/config.py) to point `TRAIN_NPZ` and `VAL_NPZ` at your `.npz` files, then:

```bash
cd tissue_segmentation
python src/train.py
```

Key config options:

| Parameter | Default | Description |
|---|---|---|
| `BACKBONE` | `convnext_tiny` | ConvNeXt variant (`tiny`, `base`, `large`) |
| `NUM_CLASSES` | `10` | Segmentation classes including background |
| `TILE_SIZE` | `256` | Input tile size in pixels |
| `BATCH_SIZE` | `4` | Training batch size |
| `LR` | `1e-4` | Initial learning rate |
| `MAX_EPOCHS` | `100` | Maximum training epochs |
| `EARLY_STOP_PATIENCE` | `15` | Epochs without improvement before stopping |

Training uses:
- **Loss**: CrossEntropyLoss with inverse-frequency class weighting (background excluded)
- **Scheduler**: Linear warmup (5 epochs) + cosine annealing
- **AMP**: Mixed precision enabled by default
- **Augmentation**: Flips, rotation, color jitter, Gaussian blur, elastic deformation (albumentations)

Checkpoints are written to `src/checkpoints/` — best model by validation Dice score, plus periodic saves every 10 epochs.

### Step 5 — Inspect Predictions

**Notebook:** [tissue_segmentation/inference/test.ipynb](tissue_segmentation/inference/test.ipynb)

Loads a checkpoint and an `.npz` tile file and visualizes model predictions side-by-side with ground-truth masks. Useful for sanity-checking a trained model before running full WSI inference.

Set `pth_model` to the checkpoint path and `pth_data` to the `.npz` file path at the top of the notebook.

### Step 6 — Inference on WSIs

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
1. Green-channel tissue masking to skip background regions
2. 1024×1024 tiles with 128-pixel overlap
3. Gaussian-weighted blending to eliminate tile boundary artifacts
4. Output: integer label map + optional colormap PNG

Key inference flags:

| Flag | Default | Description |
|---|---|---|
| `--level` | `0` | WSI pyramid level to read |
| `--tile-size` | `1024` | Inference tile size |
| `--overlap` | `128` | Overlap between adjacent tiles |
| `--scale` | `4` | Output downscale factor |
| `--no-colormap` | — | Skip colormap PNG output |

---

## Module 2: Serial Section Alignment

Registers consecutive H&E sections from serially cut tissue blocks.

### Step 1 — Run Alignment

**Notebook:** [alignment/align_serial_sections.ipynb](alignment/align_serial_sections.ipynb)

The recommended way to run alignment. Provides slide previews, a live summary table, per-slide IoU bar chart, and side-by-side visual checks of aligned sections.

**What to configure:**

| Variable | Default | Description |
|---|---|---|
| `SLIDE_FOLDER` | — | Directory containing `.ndpi` files |
| `OUTPUT_FOLDER` | — | Where to save aligned images and transforms |
| `LEVEL` | `5` | Pyramid level for registration (higher = faster) |
| `IOU_THRESHOLD` | `0.87` | Min tissue IoU to accept an alignment |
| `MAX_SKIP` | `2` | Max consecutive damaged slides to skip |
| `AFFINE_METHOD` | `ORB` | `ORB`, `ECC`, or `combined` |
| `DO_ELASTIC` | `True` | Whether to run elastic registration after affine |

**Pipeline logic:**
1. Slides are discovered and sorted by trailing number in filename
2. The middle slide is selected as the anchor (no transform applied)
3. A **forward pass** registers slides middle+1 → end
4. A **backward pass** registers slides middle-1 → 0
5. Each slide registers to the most recent successfully-aligned neighbor
6. If IoU falls below `IOU_THRESHOLD`, the pipeline skips up to `MAX_SKIP` slides (damage/artifact handling)

**Registration per pair:**
1. **Affine** — ORB feature matching → RANSAC, or ECC intensity-based, or ORB-initialized multi-scale ECC
2. **Elastic** — sparse phase-correlation grid → Gaussian-smoothed displacement field → thin-plate-spline dense field

**Output:**
```
OUTPUT_FOLDER/
├── aligned/          # Registered PNG images
├── masks/            # Tissue masks
├── transforms/
│   ├── affine/       # Affine matrices (.pkl, 2×3)
│   └── elastic/      # Displacement fields (.npy, H×W×2)
└── alignment_results.json
```

### Step 2 — Apply Transforms to Other Images

**Notebook:** [alignment/apply_transforms.ipynb](alignment/apply_transforms.ipynb)

Use this after alignment to warp a separate set of images (e.g. segmentation label maps) using the already-computed transforms. This avoids re-running registration — the transforms are loaded from the saved `.pkl` / `.npy` files and applied at the correct scale.

**What to configure:**

| Variable | Description |
|---|---|
| `ALIGNMENT_OUTPUT_FOLDER` | Output folder from the alignment run (contains `transforms/`) |
| `NEW_IMAGE_FOLDER` | Directory of images to warp (e.g. segmentation PNGs) |
| `ORIGINAL_SLIDE_FOLDER` | Original `.ndpi` files used during alignment (needed for scale calculation) |
| `REGISTRATION_LEVEL` | Pyramid level that was used during alignment |
| `SEG_DOWNSAMPLE` | Downscale factor of the new images relative to level-0 |
| `IS_LABEL_MAP` | `True` to use nearest-neighbor interpolation (label maps); `False` for bilinear |

---

## Environment Setup

A conda environment file is provided at [environment.yml](environment.yml).

**1. Install system dependency for pyvips (Ubuntu/Debian):**

```bash
sudo apt install libvips-dev
```

**2. Create and activate the environment:**

```bash
conda env create -f environment.yml
conda activate coda_env
```

**3. Register the environment as a Jupyter kernel (for notebooks):**

```bash
python -m ipykernel install --user --name coda_env --display-name "CODA"
```

**CUDA version:** The `environment.yml` defaults to CUDA 12.1 (`cu121`). If your system uses a different CUDA version, edit the two `torch` / `torchvision` lines before creating the environment:

| Your CUDA | Change `cu121` to |
|---|---|
| CUDA 11.8 | `cu118` |
| CPU only | `cpu` (and remove `+cu121` suffix) |

Check your CUDA version with `nvidia-smi`.

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
