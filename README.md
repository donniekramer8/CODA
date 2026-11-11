# CODA — Computational Pathology Pipeline

A two-module pipeline for H&E histopathology whole-slide image (WSI) analysis:

1. **Tissue Segmentation** — annotate, tile, and train a ConvNeXt-UNet model for semantic segmentation
2. **Serial Section Alignment** — rigid (FFT on tissue masks) + elastic registration of consecutive tissue sections

Adapted and refactored based on:
Kiemen, A.L., Braxton, A.M., Grahn, M.P. et al. CODA: quantitative 3D reconstruction of large tissues at cellular resolution. Nat Methods 19, 1490–1499 (2022). https://doi.org/10.1038/s41592-022-01650-9

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
└─── [align_serial_sections.ipynb]  Register serial sections (rigid FFT → elastic)
          ↓
     [apply_transforms.ipynb]          Warp label maps / images using saved transforms
     [apply_transforms_IF_channels.ipynb]  Warp N-channel IF-prediction tifs
     [roi_fullres.py]                  Extract a full-resolution ROI in the aligned frame
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
    ├── alignment_pipeline.py              # Orchestration: sorting, anchoring, pass logic
    ├── registration.py                    # Rigid FFT (default) or affine (ORB/ECC/combined) + elastic (phase-correlation) registration
    ├── rigid_fft.py                       # Rigid FFT rotation search on tissue masks (the default first step)
    ├── roi_fullres.py                     # Map a saved transform to level-0; extract a full-res ROI in the aligned frame
    ├── align_serial_sections.ipynb        # Run alignment pipeline interactively
    ├── apply_transforms.ipynb             # Warp new images (label maps / RGB) using saved transforms
    └── apply_transforms_IF_channels.ipynb # Warp N-channel IF-prediction tifs using saved transforms
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

Edit [tissue_segmentation/src/config.py](tissue_segmentation/src/config.py) to point `TRAIN_DATA_PATH` and `VAL_DATA_PATH` at your `.npz` files, then:

```bash
cd tissue_segmentation
python src/train.py
```

Key config options:

| Parameter | Default | Description |
|---|---|---|
| `CONVNEXT_VARIANT` | `tiny` | ConvNeXt backbone variant (`tiny`, `base`, `large`) |
| `NUM_CLASSES` | `10` | Segmentation classes including background |
| `IN_CHANNELS` | `3` | Input channels (RGB) |
| `TILE_SIZE` | `256` | Input tile size in pixels |
| `BATCH_SIZE` | `4` | Training batch size |
| `LEARNING_RATE` | `1e-4` | Initial learning rate |
| `WEIGHT_DECAY` | `1e-2` | AdamW weight decay |
| `EPOCHS` | `100` | Maximum training epochs |
| `WARMUP_EPOCHS` | `5` | Linear-warmup epochs before cosine annealing |
| `EARLY_STOPPING_PATIENCE` | `15` | Epochs without improvement before stopping |

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

**Batch (directory of `.ndpi` files, searched recursively):**

```bash
python inference/batch_wsi_inference.py \
    --ndpi_dir /path/to/slides/ \
    --checkpoint src/checkpoints/best_model.pth \
    --out_dir /path/to/output/ \
    --level 1 \
    --save_scale 8 \
    --no_colormap
```

Batch inference skips slides whose output PNG already exists, so it is safe to re-run.

Inference pipeline:
1. Green-channel tissue masking to skip background regions
2. Tiled inference with overlap
3. Gaussian-weighted blending to eliminate tile boundary artifacts
4. Output: integer label map + optional colormap PNG

Key inference flags (same for both scripts, except the input/output flag names):

| Flag | Default | Description |
|---|---|---|
| `--ndpi` / `--ndpi_dir` | — | Input slide (single) or folder (batch, recursive) |
| `--out` / `--out_dir` | — | Output directory for label maps |
| `--level` | `1` | WSI pyramid level to read |
| `--tile_size` | `256` (`config.TILE_SIZE`) | Inference tile size |
| `--overlap` | `64` | Overlap between adjacent tiles |
| `--batch_size` | `4` | Tiles per forward pass |
| `--g_thresh` | `190` | Green-channel tissue threshold |
| `--save_scale` | `4` | Output downscale factor |
| `--no_colormap` | — | Skip colormap PNG output |

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
| `LEVEL` | `5` | Pyramid level for registration (higher = faster; level 5 ≈ 32× downsampled) |
| `IOU_THRESHOLD` | `0.85` | Min tissue IoU to accept an alignment |
| `MAX_SKIP` | `2` | Max consecutive damaged slides to probe for a better reference |
| `G_THRESH` | `180` | Green-channel tissue threshold |
| `AFFINE_METHOD` | `FFT` | First (global) step: `FFT` = rigid FFT rotation search on tissue masks; `ORB`, `ECC`, or `combined` = affine |
| `FFT_OPTIONS` | `{}` | Options for the FFT step, e.g. `{"allow_flips": True}` (also fit mirrored sections), `{"backend": "numpy"}` (default `auto` = torch on GPU if available), `{"mask_kwargs": {"offset": 5}}` (more inclusive tissue mask) |
| `DO_ELASTIC` | `True` | Whether to run elastic registration after the global step |
| `TILE_SIZE` | `800` | Elastic registration tile size (px at working level) |
| `BUFFER` | `400` | Padding around image before tiling |
| `GRID_SPACING` | `100` | Distance between elastic tile centres |
| `TISSUE_CUTOFF` | `0.05` | Min tissue fraction in a tile to attempt elastic registration |
| `ELASTIC_PASSES` | `3` | Number of coarse-to-fine elastic passes |
| `RBF_SMOOTHING` | `0.05` | RBF smoothing of the displacement field (lower = sharper) |
| `KEEP_IF_IMPROVES` | `True` | Auto-revert any pass that lowers grayscale NCC |
| `PADDING` | `200` | White border (px at `LEVEL`) added before registration to avoid edge-cropping |

> For range-based multi-section files (e.g. `*_Sec_1-9.ndpi`) or the section-manifest
> format (`section_bbox_manifest.json`), use the `align_serial_sections_LC_CODA.ipynb`
> variant instead.

**Pipeline logic:**
1. Slides are discovered and sorted by trailing number in filename
2. The middle slide is selected as the anchor (no transform applied)
3. A **forward pass** registers slides middle+1 → end
4. A **backward pass** registers slides middle-1 → 0
5. Each slide registers to the most recent successfully-aligned neighbor
6. If IoU falls below `IOU_THRESHOLD`, the pipeline probes up to `MAX_SKIP` alternate references (behind and ahead, affine-only) and re-registers against whichever gives the best fit — so **every** slide still gets an output image, tagged in the metadata when it landed below threshold
7. The run is resumable: slides whose transforms already exist on disk are skipped on re-run

**Registration per pair:**
1. **Rigid (default, `AFFINE_METHOD = "FFT"`)**: an exhaustive rotation search on tissue masks ([alignment/rigid_fft.py](alignment/rigid_fft.py)).
   - **Masks:** each section is masked with an automatic green-channel threshold (triangle method). Small gaps are closed, dust is dropped, pieces touching the slide/scan edge (coverslip edges) are dropped, and small holes are filled while lumens are kept. Padding and warp fill are ignored, so a padded section gets the same mask as the bare scan.
   - **Search:** for each candidate angle the moving mask is rotated, and its overlap with the fixed mask is computed for every translation at once by FFT cross-correlation. The best (angle, shift) maximizes mask Dice. The search covers the full circle every 3° on a 256 px canvas, then ±3° in 0.25° steps at full resolution, with sub-degree / sub-pixel parabola refinement.
   - **Speed and output:** it runs on the GPU via torch when available, about 1 s per pair at level 5. The verbose log reports the mask Dice and a rotation-ambiguity *margin*; a margin near 0 means another rotation fits almost as well.
   - **Slide sizes:** the fit is done on each section as scanned, without stretching it to the reference canvas. The saved 2×3 matrix still applies after the canvas resize, so the `.pkl` format and the apply/ROI tools are unchanged.
   - **Alternatives:** `ORB` (feature matching → RANSAC), `ECC` (intensity-based), or `combined` (ORB-initialized multi-scale ECC) give a 6-DOF affine instead.
2. **Elastic (`DO_ELASTIC = True`)**: a coarse-to-fine refinement of the rigid result (`elastic_registration_multipass` in [alignment/registration.py](alignment/registration.py)). It works on grayscale images and the `G_THRESH` tissue masks. The fixed image is the neighbour's final (rigid + elastic) image.
   - **One pass:** both images are padded by `BUFFER` px and blurred (σ = 3 px). Tiles of `TILE_SIZE` px are taken on a grid every `GRID_SPACING` px. A tile is used only if both sections have more than `TISSUE_CUTOFF` tissue in it. Each tile's shift is measured by sub-pixel phase correlation (Hanning window, tile downsampled 2×), run in both directions and averaged.
   - **Dense field:** the sparse tile shifts are Gaussian-smoothed over the grid (σ = 2 grid points). A thin-plate spline then interpolates them into a dense displacement field (`RBF_SMOOTHING`; smaller = sharper). The spline is evaluated on a grid of at most 512 px and upsampled. With fewer than 4 usable tiles the field is zero.
   - **Three passes (`ELASTIC_PASSES = 3`):** each pass measures the misalignment left after the field so far and adds its correction on top (the fields are composed). Tile size shrinks ×0.7 and spacing ×0.6 each pass. With the notebook settings that is tiles 800 → 560 → 392 px and spacing 100 → 60 → 36 px, so early passes fix broad warps and later ones local detail.
   - **Keep only improvements (`KEEP_IF_IMPROVES = True`):** a pass is kept only if it raises grayscale NCC inside the overlap of the two tissue masks. Otherwise it is dropped, and the next, finer pass starts from the previous field.
   - **Output:** the field is a pullback, `aligned(x) = rigid(x + d(x))`, applied with `cv2.remap`. It is saved as `transforms/elastic/<slide>.npy` (H × W × 2, in registration-level px). The IoU reported for the slide is measured after this step.

**Output:**
```
OUTPUT_FOLDER/
├── aligned/          # Registered PNG images
├── masks/            # Tissue masks
├── transforms/
│   ├── affine/       # Global-step matrices (.pkl, 2×3, applied after the canvas resize)
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

### Step 2b — Apply Transforms to Multi-channel IF Predictions

**Notebook:** [alignment/apply_transforms_IF_channels.ipynb](alignment/apply_transforms_IF_channels.ipynb)

A variant of `apply_transforms.ipynb` for warping `(H, W, N)` IF-prediction tifs
(e.g. pix2pix DAPI/marker channels) through the saved H&E registration. It writes
the warped combined tif plus one subfolder per channel.

**Additional configuration (beyond `apply_transforms.ipynb`):**

| Variable | Description |
|---|---|
| `NEW_IMAGE_EXT` | Extension of the images to warp (default `.tif`) |
| `PADDING` | Padding (registration-level px) that was used during alignment |
| `CHANNEL_NAMES` | Per-channel subfolder names; extra channels fall back to `ch{i}` |
| `FILL_VALUE` | Border fill for warped output (IF background = `0`) |
| `SAVE_COMBINED_TIF` | Also write the full N-channel warped tif |

### Step 3 — Extract a Full-Resolution ROI in the Aligned Frame

**Module:** [alignment/roi_fullres.py](alignment/roi_fullres.py)

After alignment (done at a low pyramid level), use this to pull a region of
interest at **level-0 resolution** in the common anchor frame — without loading
whole slides. You pick an ROI box once in the anchor's aligned frame; for each
slide the ROI is mapped back through that slide's saved affine + elastic
transform to its own level-0 image, only the needed region is read from the
`.ndpi` (via `pyvips.extract_area`), and it is warped into the shared ROI frame.

```python
from roi_fullres import extract_roi

# roi is (x, y, w, h) in the anchor level-0 aligned frame
crop = extract_roi(slide_path, alignment_output_folder, roi_xywh=(x, y, w, h))
```

Returns an `(h, w, 3)` uint8 crop. Call it per slide to build a full-resolution
aligned stack of the same ROI across the serial sections.

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
| `.tif` | Multi-channel IF-prediction images to warp through the alignment |
| `.pth` | PyTorch model checkpoints |
| `.pkl` | Affine transform matrices |
| `.npy` | Elastic displacement fields |
| `.json` | Alignment run metadata |
