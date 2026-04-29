import numpy as np
import pyvips
import xml.etree.ElementTree as ET
from collections import namedtuple
from typing import Optional
import cv2
import bisect

Annotation = namedtuple("Annotation", ["label", "coords"])
BoundingBox = namedtuple("BoundingBox", ["x_min", "y_min", "x_max", "y_max", "labels", "annotation_indices"])
# labels             : list[str]  – unique labels of all annotations in this cluster
# annotation_indices : list[int]  – indices into the original annotation list


def _bbox(coords: np.ndarray) -> tuple[float, float, float, float]:
    return coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()


def _boxes_overlap(a: tuple, b: tuple, padding: float = 0.0) -> bool:
    """Check if two (x_min, y_min, x_max, y_max) boxes overlap (with optional padding)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (ax0 - padding <= bx1 + padding and ax1 + padding >= bx0 - padding and
            ay0 - padding <= by1 + padding and ay1 + padding >= by0 - padding)


def cluster_annotations(
    annotations: list[Annotation],
    padding: float = 0.0,
) -> list[BoundingBox]:
    """Group overlapping annotations into clusters and return one BoundingBox per cluster.

    Args:
        annotations: list of Annotation namedtuples.
        padding:     extra pixels to expand each bbox before checking overlap,
                     useful when annotations are adjacent but not touching.

    Returns:
        List of BoundingBox namedtuples sorted top-to-bottom, left-to-right (y_min, x_min).
    """
    n = len(annotations)
    bboxes = [_bbox(a.coords) for a in annotations]

    # --- Union-Find ---
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_overlap(bboxes[i], bboxes[j], padding):
                union(i, j)

    # --- Collect clusters ---
    from collections import defaultdict
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    # --- Build one BoundingBox per cluster ---
    result: list[BoundingBox] = []
    for indices in clusters.values():
        all_coords = np.vstack([annotations[i].coords for i in indices])
        x_min, y_min, x_max, y_max = _bbox(all_coords)
        labels = sorted(set(annotations[i].label for i in indices))
        result.append(BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                                  labels=labels, annotation_indices=sorted(indices)))

    # --- Sort top-to-bottom, left-to-right ---
    result.sort(key=lambda b: (b.y_min, b.x_min))
    return result


def read_annotations_xml(xml_path: str) -> list[Annotation]:
    """Parse an Aperio ImageScope XML and return a list of Annotations.

    Structure:
      <Annotations MicronsPerPixel="...">
        <Annotation Name="Islet" ...>
          <Regions>
            <Region Id="..." ...>
              <Vertices>
                <Vertex X="..." Y="..." Z="..."/>
                ...
              </Vertices>
            </Region>
            ...
          </Regions>
        </Annotation>
        ...
      </Annotations>

    Each <Region> becomes one Annotation entry with the parent <Annotation>'s Name as label.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations: list[Annotation] = []

    for annot in root.iter("Annotation"):
        label = annot.get("Name") or annot.get("PartOfGroup") or "unknown"

        for region in annot.iter("Region"):
            coords = []
            for vertex in region.iter("Vertex"):
                x = vertex.get("X")
                y = vertex.get("Y")
                if x is not None and y is not None:
                    coords.append((float(x), float(y)))

            if coords:
                annotations.append(
                    Annotation(label=label, coords=np.array(coords, dtype=np.float64))
                )

    return annotations


def read_npdi(pth):
    img = pyvips.Image.openslideload(pth, level=0)

    # Ensure 8-bit RGB(A)
    if img.format != "uchar":
        img = img.cast("uchar")
    if img.bands not in (3, 4):
        img = img.colourspace("srgb")

    # Materialize in RAM
    buf = img.write_to_memory()
    arr = np.ndarray(
        buffer=buf,
        dtype=np.uint8,
        shape=(img.height, img.width, img.bands),
    )

    return arr


def read_region(
    slide_path: str,
    bbox: BoundingBox,
    level: int = 1,
) -> np.ndarray:
    """Read a bounding-box region from an NDPI slide at the given pyramid level.

    Coordinates in BoundingBox are assumed to be at level 0.
    The function downscales them to the requested level before cropping.

    Args:
        slide_path: path to the .ndpi (or any OpenSlide-compatible file).
        bbox:       BoundingBox with level-0 pixel coordinates.
        level:      pyramid level to read from (default=1, typically 2× downsampled).

    Returns:
        np.ndarray of shape (H, W, bands) uint8.
    """
    # Open to get level dimensions (no full decode needed)
    ref = pyvips.Image.openslideload(slide_path, level=0, access="random")
    w0, h0 = ref.width, ref.height

    img = pyvips.Image.openslideload(slide_path, level=level, access="random")
    wL, hL = img.width, img.height

    # Scale factor from level-0 coords to this level's coords
    scale_x = wL / w0
    scale_y = hL / h0

    x0 = int(np.floor(bbox.x_min * scale_x))
    y0 = int(np.floor(bbox.y_min * scale_y))
    x1 = int(np.ceil(bbox.x_max  * scale_x))
    y1 = int(np.ceil(bbox.y_max  * scale_y))

    # Clamp to image bounds
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, wL), min(y1, hL)
    crop_w, crop_h = x1 - x0, y1 - y0

    region = img.crop(x0, y0, crop_w, crop_h)

    if region.format != "uchar":
        region = region.cast("uchar")
    if region.bands not in (3, 4):
        region = region.colourspace("srgb")

    buf = region.write_to_memory()
    arr = np.ndarray(
        buffer=buf,
        dtype=np.uint8,
        shape=(region.height, region.width, region.bands),
    )
    return arr.copy()   # copy so buf can be GC'd


def make_segmentation_mask(
    bbox: BoundingBox,
    annotations: list[Annotation],
    label_to_class: dict[str, int],
    level: int = 1,
    slide_path: str = None,
) -> np.ndarray:
    """Rasterize annotation polygons into a segmentation mask aligned to the bbox region.

    Coordinates are in level-0 pixel space. They are scaled to the requested
    pyramid level and then shifted so that bbox.x_min / bbox.y_min becomes (0, 0).

    Args:
        bbox:           BoundingBox (level-0 coords) for the cluster.
        annotations:    Full list of Annotation namedtuples (level-0 coords).
        label_to_class: Mapping from label string to integer class index, e.g.
                        {"background": 0, "Islet": 1, "Tumor": 2}.
                        Labels not in the dict are skipped.
        level:          Pyramid level (default=1). Used only to compute the mask
                        size; pass slide_path too so scale factors can be derived.
        slide_path:     Path to the slide file. Required when level > 0 so that
                        the level-0 → level-N scale factor can be inferred.
                        If None, level is assumed to be 0 (no scaling).

    Returns:
        np.ndarray of shape (H, W) dtype uint8, where each pixel value is the
        class index. Background (unlabelled) pixels are 0.
    """
    # --- Compute scale factor level-0 → requested level ---
    if level == 0 or slide_path is None:
        scale_x = scale_y = 1.0
        w0 = int(np.ceil(bbox.x_max)) - int(np.floor(bbox.x_min))
        h0 = int(np.ceil(bbox.y_max)) - int(np.floor(bbox.y_min))
        mask_w, mask_h = w0, h0
    else:
        ref = pyvips.Image.openslideload(slide_path, level=0, access="random")
        lvl = pyvips.Image.openslideload(slide_path, level=level, access="random")
        scale_x = lvl.width  / ref.width
        scale_y = lvl.height / ref.height

        x0_px = int(np.floor(bbox.x_min * scale_x))
        y0_px = int(np.floor(bbox.y_min * scale_y))
        x1_px = int(np.ceil(bbox.x_max  * scale_x))
        y1_px = int(np.ceil(bbox.y_max  * scale_y))
        mask_w = x1_px - x0_px
        mask_h = y1_px - y0_px

    # Origin of the bbox in level-N pixel coords
    origin_x = bbox.x_min * scale_x
    origin_y = bbox.y_min * scale_y

    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

    # Draw polygons from highest class first so lower indices = background
    # Sort by class index ascending so higher-priority classes paint over lower ones
    relevant = [
        (annotations[i], label_to_class[annotations[i].label])
        for i in bbox.annotation_indices
        if annotations[i].label in label_to_class
    ]
    relevant.sort(key=lambda x: x[1])  # ascending: background first, foreground last

    for annot, class_idx in relevant:
        # Scale + shift coords to mask space
        pts = annot.coords.copy()
        pts[:, 0] = pts[:, 0] * scale_x - origin_x
        pts[:, 1] = pts[:, 1] * scale_y - origin_y
        polygon = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon], color=int(class_idx))

    return mask


def make_content_mask(
    bbox: BoundingBox,
    annotations: list[Annotation],
    level: int = 1,
    slide_path: str = None,
) -> np.ndarray:
    """Rasterize ALL annotation polygons in this cluster into a binary content mask.

    Every annotation polygon (regardless of label) is filled with 1.
    This defines the "content region" — pixels outside are background/whitespace
    that should not appear on the mosaic canvas.

    Returns:
        np.ndarray (H, W) dtype bool — True = inside at least one annotation polygon.
    """
    if level == 0 or slide_path is None:
        scale_x = scale_y = 1.0
        mask_w = int(np.ceil(bbox.x_max)) - int(np.floor(bbox.x_min))
        mask_h = int(np.ceil(bbox.y_max)) - int(np.floor(bbox.y_min))
    else:
        ref = pyvips.Image.openslideload(slide_path, level=0, access="random")
        lvl = pyvips.Image.openslideload(slide_path, level=level, access="random")
        scale_x = lvl.width  / ref.width
        scale_y = lvl.height / ref.height

        x0_px = int(np.floor(bbox.x_min * scale_x))
        y0_px = int(np.floor(bbox.y_min * scale_y))
        x1_px = int(np.ceil(bbox.x_max  * scale_x))
        y1_px = int(np.ceil(bbox.y_max  * scale_y))
        mask_w = x1_px - x0_px
        mask_h = y1_px - y0_px

    origin_x = bbox.x_min * scale_x
    origin_y = bbox.y_min * scale_y

    content = np.zeros((mask_h, mask_w), dtype=np.uint8)

    for i in bbox.annotation_indices:
        annot = annotations[i]
        pts = annot.coords.copy()
        pts[:, 0] = pts[:, 0] * scale_x - origin_x
        pts[:, 1] = pts[:, 1] * scale_y - origin_y
        polygon = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(content, [polygon], color=1)

    return content.astype(bool)


def _rotate_pair(
    img: np.ndarray,
    mask: np.ndarray,
    content: np.ndarray,
    angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate img (H,W,3), mask (H,W), and content mask (H,W) by angle degrees CCW."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    img_rot  = cv2.warpAffine(img,  M, (new_w, new_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))  # white border for H&E
    mask_rot = cv2.warpAffine(mask, M, (new_w, new_h),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
    content_u8 = content.astype(np.uint8)
    content_rot = cv2.warpAffine(content_u8, M, (new_w, new_h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    return img_rot, mask_rot, content_rot.astype(bool)


def build_mosaic(
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    canvas_size: int = 10240,
    fill_threshold: float = 0.5,
    gap: int = 4,
    allow_rotation: bool = True,
    placement_candidates: int = 32,
    overlap_penalty: float = 0.3,
    grid_cell: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack (H&E image, mask, content_mask) triples onto a square mosaic canvas.

    Strategy: coverage-aware stochastic placement with z-ordering.
    Tiles are sorted large→small. For each tile we sample `placement_candidates`
    random (x, y) positions that fit on the canvas, score them by
        score = new_uncovered_pixels - overlap_penalty * already_covered_pixels
    and pick the best. Large tiles may paint over smaller ones already placed.
    A coarse grid occupancy map makes scoring fast.

    Args:
        pairs:               List of (img, mask, content_mask) tuples.
        canvas_size:         Side length of the square canvas.
        fill_threshold:      Stop once this fraction of canvas content is filled.
        gap:                 Minimum pixel gap between tile bounding boxes.
        allow_rotation:      Randomly rotate each tile in [0, 360) before placing.
        placement_candidates: How many random positions to evaluate per tile.
                              Higher → denser packing, slower runtime.
        overlap_penalty:     Weight applied to already-covered pixels in the
                             placement score. 0 = ignore overlap, 1 = strongly avoid.
        grid_cell:           Side of each cell in the coarse occupancy grid (px).
                             Smaller → more accurate scoring but more memory.
        rng:                 numpy random Generator for reproducibility.

    Returns:
        (he_mosaic, mask_mosaic) — both canvas_size × canvas_size uint8.
    """
    if rng is None:
        rng = np.random.default_rng()

    he_mosaic   = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    EMPTY       = np.uint8(254)
    mask_mosaic = np.full((canvas_size, canvas_size), EMPTY, dtype=np.uint8)

    target_pixels = int(canvas_size * canvas_size * fill_threshold)

    # ── Coarse occupancy grid ──────────────────────────────────────────────────
    # covered_grid[gy, gx] = approximate number of content pixels placed in cell
    grid_dim   = (canvas_size + grid_cell - 1) // grid_cell
    # We track two things: total placed content pixels (for fill) and
    # a boolean "any content here" grid for fast scoring.
    occupied   = np.zeros((grid_dim, grid_dim), dtype=np.uint32)  # pixel count per cell

    filled_pixels = 0

    def _grid_score(dx: int, dy: int, cw: int, ch: int, content_area: int) -> float:
        """Estimate (new_pixels - penalty*overlap_pixels) using the coarse grid."""
        gx0 = dx // grid_cell
        gy0 = dy // grid_cell
        gx1 = min((dx + cw - 1) // grid_cell + 1, grid_dim)
        gy1 = min((dy + ch - 1) // grid_cell + 1, grid_dim)
        cell_total    = occupied[gy0:gy1, gx0:gx1].sum()
        # approximate: fraction of tile area that is already occupied
        tile_cells    = max((gx1 - gx0) * (gy1 - gy0), 1)
        avg_density   = cell_total / (tile_cells * grid_cell * grid_cell)
        est_overlap   = avg_density * content_area
        est_new       = content_area - est_overlap
        return est_new - overlap_penalty * est_overlap

    def _update_grid(dx: int, dy: int, content: np.ndarray) -> None:
        ch, cw = content.shape
        # Accumulate content pixel counts into grid cells using reduceat
        # Fastest approach: reshape into grid cells with padding
        gx0 = dx // grid_cell
        gy0 = dy // grid_cell
        gx1 = min((dx + cw - 1) // grid_cell + 1, grid_dim)
        gy1 = min((dy + ch - 1) // grid_cell + 1, grid_dim)

        # Build a padded content array aligned to grid cells
        pad_x0 = dx - gx0 * grid_cell
        pad_y0 = dy - gy0 * grid_cell
        pw = (gx1 - gx0) * grid_cell
        ph = (gy1 - gy0) * grid_cell
        buf = np.zeros((ph, pw), dtype=np.uint8)
        buf[pad_y0:pad_y0 + ch, pad_x0:pad_x0 + cw] = content.astype(np.uint8)

        # Sum each grid cell
        block = buf.reshape(gy1 - gy0, grid_cell, gx1 - gx0, grid_cell)
        occupied[gy0:gy1, gx0:gx1] += block.sum(axis=(1, 3)).astype(np.uint32)

    # ── Tile preparation ───────────────────────────────────────────────────────
    def _rotate_pair_safe(img, mask, content, angle):
        """Thin wrapper — delegates to your existing _rotate_pair."""
        return _rotate_pair(img, mask, content, angle)

    def _prepare_tile(img, mask, content, angle):
        img, mask, content = img[..., :3].copy(), mask.copy(), content.copy()
        if angle is not None:
            img, mask, content = _rotate_pair_safe(img, mask, content, angle)
        if not content.any():
            return None
        img[~content] = 255
        rows = np.where(np.any(content, axis=1))[0]
        cols = np.where(np.any(content, axis=0))[0]
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        img_c, mask_c, content_c = img[r0:r1, c0:c1], mask[r0:r1, c0:c1], content[r0:r1, c0:c1]
        ch, cw = img_c.shape[:2]
        if cw > canvas_size or ch > canvas_size:
            return None
        content_area = int(content_c.sum())
        if content_area == 0:
            return None
        return (img_c, mask_c, content_c, ch, cw, content_area)

    # Prepare all tiles
    processed = []
    for img, mask, content in pairs:
        angle = float(rng.uniform(0, 360)) if allow_rotation else None
        tile  = _prepare_tile(img, mask, content, angle)
        if tile is not None:
            processed.append(tile)

    if not processed:
        mask_mosaic[mask_mosaic == EMPTY] = 0
        return he_mosaic, mask_mosaic

    # Sort large → small so big tiles get first pick of empty canvas
    processed.sort(key=lambda t: t[5], reverse=True)

    # ── Placement loop ─────────────────────────────────────────────────────────
    # We cycle through tiles repeatedly (with fresh rotations each cycle)
    # until fill_threshold is reached or a full cycle adds nothing.
    max_x_range = max(canvas_size - processed[-1][4], 1)  # smallest tile width
    max_y_range = max(canvas_size - processed[-1][3], 1)

    cycle = 0
    while filled_pixels < target_pixels:
        gained_this_cycle = 0

        if cycle > 0 and allow_rotation:
            # Re-rotate for next cycle to fill differently shaped gaps
            new_processed = []
            for img, mask, content in pairs:
                angle = float(rng.uniform(0, 360))
                tile  = _prepare_tile(img, mask, content, angle)
                if tile is not None:
                    new_processed.append(tile)
            if not new_processed:
                break
            new_processed.sort(key=lambda t: t[5], reverse=True)
            processed = new_processed

        for img_crop, mask_crop, content_crop, ch, cw, content_area in processed:
            if filled_pixels >= target_pixels:
                break

            max_dx = canvas_size - cw
            max_dy = canvas_size - ch
            if max_dx < 0 or max_dy < 0:
                continue  # tile larger than canvas

            # Sample candidate positions
            # Weight candidate positions toward empty regions using the grid:
            # pick grid cells with low occupancy, then refine to pixel coords.
            n = placement_candidates

            if max_dx == 0:
                xs = np.zeros(n, dtype=int)
            else:
                xs = rng.integers(0, max_dx + 1, size=n)

            if max_dy == 0:
                ys = np.zeros(n, dtype=int)
            else:
                ys = rng.integers(0, max_dy + 1, size=n)

            # Apply gap: nudge candidates away from tile edges naively
            # (gap is enforced only as a soft suggestion via scoring; hard
            #  enforcement would require a distance transform — expensive)
            best_score = -1e18
            best_dx = best_dy = 0

            for dx, dy in zip(xs.tolist(), ys.tolist()):
                s = _grid_score(dx, dy, cw, ch, content_area)
                if s > best_score:
                    best_score = s
                    best_dx, best_dy = dx, dy

            # Place tile at best position (paint over whatever is there)
            dx, dy = best_dx, best_dy
            dest_slice_he   = he_mosaic  [dy:dy+ch, dx:dx+cw]
            dest_slice_mask = mask_mosaic[dy:dy+ch, dx:dx+cw]

            # Count genuinely new pixels (were EMPTY before)
            new_mask   = content_crop & (dest_slice_mask == EMPTY)
            new_pixels = int(new_mask.sum())

            # Write all content pixels (overwriting previously placed tiles)
            dest_slice_he  [content_crop] = img_crop [content_crop]
            dest_slice_mask[content_crop] = mask_crop[content_crop]

            _update_grid(dx, dy, content_crop)
            filled_pixels  += new_pixels
            gained_this_cycle += new_pixels

        if gained_this_cycle == 0:
            break  # canvas is saturated, no point continuing
        cycle += 1

    mask_mosaic[mask_mosaic == EMPTY] = 0
    return he_mosaic, mask_mosaic


def get_TA_from_imHE(
    he_image: np.ndarray,
    g_thresh: int = 170,
    scale_factor: int = 1,
    min_background_pixels: int = 500,
) -> np.ndarray:
    """Compute a tissue mask from an H&E image using the green channel.

    Args:
        he_image:              H&E image (H, W, 3) or (H, W, 4) uint8.
        g_thresh:              Green channel threshold — pixels with G < g_thresh are tissue.
        scale_factor:          Downscale factor for efficiency (1 = no downscale).
        min_background_pixels: Minimum size of a background (non-tissue) connected component
                               to keep. Smaller background regions are filled in as tissue,
                               removing small holes/specks of background inside tissue.

    Returns:
        Boolean np.ndarray of shape (H, W): True = tissue, False = background.
    """
    h, w = he_image.shape[:2]
    img_rgb = he_image[..., :3]

    if scale_factor > 1:
        small = cv2.resize(img_rgb,
                           (w // scale_factor, h // scale_factor),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img_rgb

    g_channel    = small[:, :, 1]
    tissue_small = (g_channel < g_thresh).astype(np.uint8)

    # Remove small background components (fill holes in tissue)
    background_small = 1 - tissue_small
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background_small, connectivity=8)
    for label_id in range(1, num_labels):   # skip background-of-background (0)
        if stats[label_id, cv2.CC_STAT_AREA] < min_background_pixels // (scale_factor ** 2):
            tissue_small[labels == label_id] = 1   # fill small background hole → tissue

    if scale_factor > 1:
        tissue_small = cv2.resize(tissue_small, (w, h), interpolation=cv2.INTER_NEAREST)

    return tissue_small.astype(bool)


def apply_tissue_mask(
    mask: np.ndarray,
    he_image: np.ndarray,
    g_thresh: int = 170,
    scale_factor: int = 1,
    min_background_pixels: int = 500,
) -> np.ndarray:
    """Apply tissue mask to annotation mask — set non-tissue pixels to background (0).

    Args:
        mask:                  Annotation mask with class labels (H, W) uint8.
        he_image:              Corresponding H&E image (H, W, 3) uint8.
        g_thresh:              Green channel threshold for tissue detection.
        scale_factor:          Downscaling factor for efficiency.
        min_background_pixels: Small background regions smaller than this are filled as tissue.

    Returns:
        Masked annotation where non-tissue pixels are set to 0.
    """
    tissue_mask = get_TA_from_imHE(he_image, g_thresh, scale_factor, min_background_pixels)
    masked = mask.copy()
    masked[~tissue_mask] = 0
    return masked

