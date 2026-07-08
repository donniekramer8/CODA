"""
Full-resolution ROI extraction following the serial-section registration.

The alignment pipeline registers slides at a low pyramid level (e.g. level 5)
and saves, per slide, an affine matrix + a dense elastic displacement field that
map that slide's *raw padded low-res image* directly into the common anchor
frame (the `fixed` target during registration is always an already-anchor-framed
image, so transforms are moving->anchor, not chained).

This module lets you:
  1. Pick an ROI box once, in the anchor's aligned (registered) frame.
  2. For each slide, map that ROI back through the slide's transform to find the
     matching region in its OWN level-0 image, read ONLY that region from the
     .ndpi (via pyvips extract_area — tiled, no full-slide load), and warp it
     into the common anchor-frame ROI at full level-0 resolution.

Coordinate conventions (VERIFIED against saved aligned PNGs, exact match)
-------------------------------------------------------------------------
Forward (what the pipeline did at the registration level L), in order:
    1. raw_L  = read slide at level L                       (h0, w0)
    2. padded = pad raw_L by `padding` px on all sides      (h0+2p, w0+2p)
    3. moving = resize padded -> anchor canvas (H, W)       (match_size step!)
    4. affine = cv2.warpAffine(moving, M)                   pullback
    5. anchor = cv2.remap(affine, coord + disp)             elastic pullback

So for an output (anchor) pixel x the pullback chain to a raw-level-0 source is:
    after_elastic = x + disp(x)                       # elastic pullback (step 5)
    after_affine  = M_inv @ after_elastic             # affine pullback  (step 4)
    in_padded     = after_affine / match_scale        # undo resize      (step 3)
    in_raw_L      = in_padded - padding               # undo pad         (step 2)
    in_raw_L0     = in_raw_L * level_downsample        # L -> L0          (step 1)

`match_scale = (W / (w0 + 2p), H / (h0 + 2p))` is the per-axis resize factor from
the padded raw-L image to the anchor canvas. It is usually very close to 1 but
not exactly 1, so it must not be ignored.
"""

import os
import re
import json
import pickle
import numpy as np
import cv2
import pyvips

from registration import get_level_dimensions, _bg_color
from alignment_pipeline import load_transform, _read_level


# ── High-level orchestration ────────────────────────────────────────────────

def _anchor_padding(aln_dir: str) -> int:
    """Recover the padding (level-L px) used during registration from metadata."""
    meta = json.load(open(os.path.join(aln_dir, "alignment_results.json")))
    for r in meta:
        m = re.search(r"padding=(\d+)", r.get("notes", ""))
        if m:
            return int(m.group(1))
    return 0


def load_slide_transform(aln_dir: str, slide_path: str):
    """Load and pre-compose everything needed to map an anchor-L0 ROI into one
    slide's raw level-0 image.

    Returns a dict: {M_full, disp_lr, level_downsample, fill, has_transform}.
    Anchor slides (no affine matrix saved) get an identity M_full so the ROI
    maps straight through.
    """
    stem = os.path.splitext(os.path.basename(slide_path))[0]
    affine_dir = os.path.join(aln_dir, "transforms", "affine")
    pkl = os.path.join(affine_dir, stem + ".pkl")
    if not os.path.exists(pkl):
        return None

    rec = pickle.load(open(pkl, "rb"))
    t = load_transform(affine_dir, stem)
    L = rec["level"]
    pad = _anchor_padding(aln_dir)

    dims = get_level_dimensions(slide_path)
    ds = dims[0][0] / dims[L][0]

    raw = _read_level(slide_path, L)
    rh, rw = raw.shape[:2]
    fill = _bg_color(raw)

    # Anchor canvas dims = raw padded then resized; for the anchor itself the
    # saved aligned PNG defines the canvas. Use it if present, else padded raw.
    aligned_png = os.path.join(aln_dir, "aligned", stem + ".png")
    if os.path.exists(aligned_png):
        from PIL import Image
        H, W = np.array(Image.open(aligned_png)).shape[:2]
    else:
        H, W = rh + 2 * pad, rw + 2 * pad

    M = t["affine_matrix"]
    disp = t["elastic_transform"]
    if M is None:
        # Anchor: anchor frame == padded raw resized to (H,W); compose with
        # identity affine so the same machinery applies.
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

    M_full = build_anchorL0_to_rawL0_affine(np.asarray(M), rh, rw, H, W, pad, ds)
    return {
        "M_full": M_full,
        "disp_lr": disp,
        "level_downsample": ds,
        "fill": fill,
        "anchor_hw": (H, W),
    }


def extract_roi(slide_path: str, aln_dir: str, roi_xywh, margin: int = 128):
    """One-call full-res ROI extraction for a single slide.

    `roi_xywh` is (x, y, w, h) in the ANCHOR level-0 frame. Returns the (h, w, 3)
    uint8 crop in that frame, reading only the needed level-0 region.
    """
    info = load_slide_transform(aln_dir, slide_path)
    if info is None:
        raise FileNotFoundError(f"No saved transform for {slide_path}")
    M_full, disp, ds, fill = (info["M_full"], info["disp_lr"],
                              info["level_downsample"], info["fill"])
    l, tp, bw, bh = anchor_roi_to_source_bbox(roi_xywh, M_full, disp, ds, margin)
    region = read_ndpi_region(slide_path, l, tp, bw, bh)
    return warp_region_to_anchor_roi(region, (l, tp), roi_xywh, M_full, disp, ds, fill=fill)


# ── Transform reconstruction ────────────────────────────────────────────────

def _to_3x3(M):
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (2, 3):
        M = np.vstack([M, [0, 0, 1]])
    return M


def upsample_displacement(disp_lr: np.ndarray, out_h: int, out_w: int,
                          scale: float) -> np.ndarray:
    """Upsample a low-res (H, W, 2) displacement field to (out_h, out_w, 2),
    scaling the displacement magnitudes by `scale` (px are resolution-relative)."""
    dx = cv2.resize(disp_lr[:, :, 0], (out_w, out_h), interpolation=cv2.INTER_LINEAR) * scale
    dy = cv2.resize(disp_lr[:, :, 1], (out_w, out_h), interpolation=cv2.INTER_LINEAR) * scale
    return np.stack([dx, dy], axis=-1)


def build_anchorL0_to_rawL0_affine(
    M_lr: np.ndarray,         # 2x3 affine saved by the pipeline (pullback, level L)
    raw_h: int, raw_w: int,   # raw slide dims at level L (before padding)
    anchor_h: int, anchor_w: int,  # anchor canvas dims at level L
    padding_lr: int,          # padding added before registration, level-L px
    level_downsample: float,  # L0 size / level-L size (e.g. 32 for level 5)
) -> np.ndarray:
    """Compose the full pullback affine: anchor-L0 coords -> raw-L0 moving coords.

    Mirrors the pipeline's forward chain (resize-after-pad, then affine) but as a
    single matrix so the elastic field is the only separately-applied piece.
    The elastic pullback (x + disp) happens in anchor-L0 coords BEFORE this
    matrix is applied.

    Chain (pullback = output -> source), all at level L unless noted:
        anchor_L  --M_inv-->  moving_resized
        moving_resized  --(1/match_scale)-->  padded
        padded  --(- padding)-->  raw_L
        raw_L   --(* downsample)-->  raw_L0     (and anchor_L = anchor_L0 / downsample)
    """
    padded_w = raw_w + 2 * padding_lr
    padded_h = raw_h + 2 * padding_lr
    # Forward match_size scale (padded -> anchor canvas), per axis.
    msx = anchor_w / padded_w
    msy = anchor_h / padded_h

    Minv = _to_3x3(cv2.invertAffineTransform(np.asarray(M_lr, np.float64)))

    # undo resize: divide coords by match_scale
    S_unresize = np.diag([1.0 / msx, 1.0 / msy, 1.0])
    # undo padding: translate by -padding
    T_unpad = np.array([[1, 0, -padding_lr],
                        [0, 1, -padding_lr],
                        [0, 0, 1]], dtype=np.float64)
    # level-L raw -> level-0 raw (and we receive anchor in L0, so pre-divide input by ds)
    D_out = np.diag([level_downsample, level_downsample, 1.0])   # raw_L -> raw_L0
    D_in = np.diag([1.0 / level_downsample, 1.0 / level_downsample, 1.0])  # anchor_L0 -> anchor_L

    full = D_out @ T_unpad @ S_unresize @ Minv @ D_in
    return full[:2, :]


# ── Geometry: map the anchor ROI back to a raw level-0 read window ───────────

def _anchorL0_to_rawL0_points(
    pts_l0,               # (N,2) anchor-L0 coords
    M_full,               # 2x3 anchor-L0 -> raw-L0 (from build_anchorL0_to_rawL0_affine)
    disp_lr,              # (Hlr,Wlr,2) elastic field in anchor level-L coords, or None
    level_downsample,
):
    """Map anchor-L0 points to raw-L0 moving points, applying the elastic
    pullback in level-L space first, then the composed affine."""
    pts = np.asarray(pts_l0, dtype=np.float64).copy()
    if disp_lr is not None:
        # Elastic pullback is defined in anchor level-L coords: look up disp at
        # the L-scaled location, then add the (L-scale) displacement, scaled to L0.
        lr_x = np.clip((pts[:, 0] / level_downsample).astype(int), 0, disp_lr.shape[1] - 1)
        lr_y = np.clip((pts[:, 1] / level_downsample).astype(int), 0, disp_lr.shape[0] - 1)
        pts[:, 0] += disp_lr[lr_y, lr_x, 0] * level_downsample
        pts[:, 1] += disp_lr[lr_y, lr_x, 1] * level_downsample
    ones = np.ones((pts.shape[0], 1))
    src = (M_full @ np.hstack([pts, ones]).T).T
    return src


def anchor_roi_to_source_bbox(
    roi_xywh, M_full, disp_lr, level_downsample, margin: int = 128,
):
    """Map a grid of points across the anchor-frame ROI back into the raw
    level-0 moving image to get the bounding box to read with extract_area.

    Returns (left, top, width, height) in raw level-0 moving coords + margin.
    """
    x, y, w, h = roi_xywh
    nx = max(2, w // 256)
    ny = max(2, h // 256)
    gx = np.linspace(x, x + w - 1, nx)
    gy = np.linspace(y, y + h - 1, ny)
    mx, my = np.meshgrid(gx, gy)
    pts = np.stack([mx.ravel(), my.ravel()], axis=-1)

    src = _anchorL0_to_rawL0_points(pts, M_full, disp_lr, level_downsample)
    left = int(np.floor(src[:, 0].min())) - margin
    top = int(np.floor(src[:, 1].min())) - margin
    right = int(np.ceil(src[:, 0].max())) + margin
    bottom = int(np.ceil(src[:, 1].max())) + margin
    return left, top, right - left, bottom - top


# ── Level-0 region read (only the needed tiles) ──────────────────────────────

def read_ndpi_region(path: str, left: int, top: int, width: int, height: int) -> np.ndarray:
    """Read a level-0 region [left, top, width, height] as uint8 RGB.

    Clamps the request to the slide bounds; pads with white where the request
    extends outside the slide so the returned array is always (height, width, 3).
    """
    img = pyvips.Image.openslideload(path, level=0)
    W0, H0 = img.width, img.height

    # Clamp the readable sub-rectangle to the slide.
    rl = max(0, left)
    rt = max(0, top)
    rr = min(W0, left + width)
    rb = min(H0, top + height)

    out = np.full((height, width, 3), 255, dtype=np.uint8)
    if rr <= rl or rb <= rt:
        return out  # entirely outside the slide

    region = img.extract_area(rl, rt, rr - rl, rb - rt)
    if region.bands == 4:
        region = region.extract_band(0, n=3)
    elif region.bands != 3:
        region = region.colourspace("srgb")
    if region.format != "uchar":
        region = region.cast("uchar")
    buf = region.write_to_memory()
    arr = np.ndarray(buffer=buf, dtype=np.uint8,
                     shape=(region.height, region.width, region.bands)).copy()

    # Place into the (possibly larger) output canvas at the clamped offset.
    oy, ox = rt - top, rl - left
    out[oy:oy + arr.shape[0], ox:ox + arr.shape[1]] = arr
    return out


# ── Warp a raw level-0 region into the anchor-frame ROI ──────────────────────

def warp_region_to_anchor_roi(
    region: np.ndarray,        # raw level-0 region read with read_ndpi_region
    region_origin,             # (left, top) of `region` in raw L0 coords
    roi_xywh,                  # (x, y, w, h) in anchor-L0 frame
    M_full: np.ndarray,        # anchor-L0 -> raw-L0 (build_anchorL0_to_rawL0_affine)
    disp_lr,                   # (Hlr,Wlr,2) elastic in anchor level-L coords, or None
    level_downsample: float,
    fill=255,
) -> np.ndarray:
    """Produce the (h, w, 3) output crop in the anchor-L0 frame by pulling each
    output pixel from `region`."""
    x, y, w, h = roi_xywh
    left, top = region_origin

    oy, ox = np.mgrid[0:h, 0:w]
    pts = np.stack([(ox + x).ravel(), (oy + y).ravel()], axis=-1).astype(np.float64)

    src = _anchorL0_to_rawL0_points(pts, M_full, disp_lr, level_downsample)
    map_x = (src[:, 0] - left).reshape(h, w).astype(np.float32)
    map_y = (src[:, 1] - top).reshape(h, w).astype(np.float32)

    fill_v = (fill, fill, fill) if isinstance(fill, int) else tuple(fill)
    return cv2.remap(region, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=fill_v)
