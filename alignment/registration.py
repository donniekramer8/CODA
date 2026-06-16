"""
Serial section registration utilities.

Provides affine and sparse-grid elastic registration of H&E tissue sections
read from .ndpi whole-slide images.
"""

import numpy as np
import cv2
import pyvips
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RBFInterpolator
from typing import Optional, Tuple


# ── I/O helpers ────────────────────────────────────────────────────────────────

def read_ndpi_level(path: str, level: int = 3) -> np.ndarray:
    """Read an .ndpi slide at the requested pyramid level as a uint8 RGB array."""
    img = pyvips.Image.openslideload(path, level=level)
    if img.format != "uchar":
        img = img.cast("uchar")
    if img.bands == 4:
        # Drop alpha
        img = img.extract_band(0, n=3)
    elif img.bands != 3:
        img = img.colourspace("srgb")
    buf = img.write_to_memory()
    arr = np.ndarray(buffer=buf, dtype=np.uint8,
                     shape=(img.height, img.width, img.bands))
    return arr.copy()


def get_level_dimensions(path: str):
    """Return list of (width, height) for each pyramid level."""
    base = pyvips.Image.openslideload(path, level=0)
    n_levels = int(base.get("openslide.level-count"))
    dims = []
    for lv in range(n_levels):
        w = int(base.get(f"openslide.level[{lv}].width"))
        h = int(base.get(f"openslide.level[{lv}].height"))
        dims.append((w, h))
    return dims


def _is_proper_downsample(dims: list, level: int, tol: float = 0.15) -> bool:
    """
    Return True if dims[level] is roughly half of dims[level-1] in both axes.
    A thumbnail is typically a much larger or irregular jump, so it fails this check.
    """
    if level == 0 or level >= len(dims):
        return False
    w_prev, h_prev = dims[level - 1]
    w_cur,  h_cur  = dims[level]
    # Expected: each axis is ~0.5× the level above
    rx = w_cur / w_prev if w_prev > 0 else 0
    ry = h_cur / h_prev if h_prev > 0 else 0
    return abs(rx - 0.5) <= tol and abs(ry - 0.5) <= tol


def read_ndpi_at_target_level(
    path: str,
    target_level: int,
    halving_tol: float = 0.15,
) -> Tuple[np.ndarray, int, float]:
    """
    Read an .ndpi file at the resolution closest to `target_level`, ensuring
    the level is a genuine 2× downsample and not a thumbnail.

    Strategy
    --------
    1. Walk from `target_level` down to 0 to find the highest valid level
       whose dimensions are a proper halving of the level above.
    2. Read that level, then resize to the target dimensions via cv2.
       If the valid level IS the target level no resize is needed.

    Returns
    -------
    arr        : uint8 RGB array at the target (or best-approximated) resolution
    used_level : the actual pyramid level that was read
    scale      : resize factor applied after reading (1.0 means no resize)
    """
    dims = get_level_dimensions(path)
    n_levels = len(dims)

    # Clamp target to available range
    actual_target = min(target_level, n_levels - 1)

    # Find the best valid level <= actual_target
    best_level = actual_target
    if actual_target > 0 and not _is_proper_downsample(dims, actual_target, halving_tol):
        # Walk down to find a level that IS a proper halving
        best_level = 0
        for lv in range(actual_target - 1, 0, -1):
            if _is_proper_downsample(dims, lv, halving_tol):
                best_level = lv
                break

    arr = read_ndpi_level(path, level=best_level)

    # If we had to fall back, resize to match the target dimensions
    if best_level != actual_target:
        tw, th = dims[actual_target] if actual_target < n_levels else dims[best_level]
        # Compute from level-0 what the target dims would be
        w0, h0 = dims[0]
        # Target downsample factor expected at target_level = 2^target_level
        expected_scale = 0.5 ** target_level
        tw = max(1, int(round(w0 * expected_scale)))
        th = max(1, int(round(h0 * expected_scale)))
        scale = tw / arr.shape[1]
        arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    return arr, best_level, scale


def get_scale_factor(path: str, level: int) -> Tuple[float, float]:
    """Return (sx, sy) scale from level-0 to the requested level."""
    dims = get_level_dimensions(path)
    w0, h0 = dims[0]
    wL, hL = dims[level]
    return wL / w0, hL / h0


# ── Tissue masking ─────────────────────────────────────────────────────────────

def tissue_mask(
    he_image: np.ndarray,
    g_thresh: int = 170,
    min_bg_pixels: int = 500,
) -> np.ndarray:
    """Binary tissue mask from H&E image using green channel thresholding.

    Returns:
        Boolean array (H, W): True = tissue.
    """
    g = he_image[:, :, 1]
    tissue = (g < g_thresh).astype(np.uint8)

    # Fill small background holes. Vectorized: build a per-label lookup of which
    # background components are too small, then map it across the label image in
    # one pass. (A per-label Python loop with `labels == lid` rescans the whole
    # image for every component — pathologically slow when a noisy background at
    # high resolution produces tens of thousands of tiny components.)
    bg = 1 - tissue
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    if n_labels > 1:
        small = stats[:, cv2.CC_STAT_AREA] < min_bg_pixels
        small[0] = False  # label 0 is the tissue side of `bg`, never fill it
        fill = small[labels]            # (H, W) bool, one vectorized lookup
        tissue[fill] = 1

    return tissue.astype(bool)


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-Union of two boolean masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def masked_ncc(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Normalized cross-correlation of two grayscale images over a mask.

    Unlike IoU (which only sees the tissue silhouette), NCC sees internal
    texture — so it actually rewards fine elastic alignment of structures
    inside the tissue. Returned in [-1, 1]; higher is better.
    """
    f = fixed_gray.astype(np.float32)
    m = moving_gray.astype(np.float32)
    if mask is not None:
        sel = mask.astype(bool)
    else:
        sel = np.ones(f.shape[:2], dtype=bool)
    if sel.sum() < 16:
        return 0.0
    fv = f[sel]
    mv = m[sel]
    fv = fv - fv.mean()
    mv = mv - mv.mean()
    denom = np.sqrt((fv * fv).sum() * (mv * mv).sum())
    if denom < 1e-8:
        return 0.0
    return float((fv * mv).sum() / denom)


def compose_displacements(
    disp_a: np.ndarray,
    disp_b: np.ndarray,
) -> np.ndarray:
    """Compose two displacement fields so applying the result once equals
    applying disp_a then disp_b (both consumed by apply_elastic_transform,
    i.e. cv2.remap pullback: out(x) = in(x + disp)).

    For pullback warps, warping by disp_a then disp_b samples the source at
    x + disp_b(x) + disp_a(x + disp_b(x)). We resample disp_a at the
    disp_b-displaced coordinates and add.
    """
    H, W = disp_b.shape[:2]
    row, col = np.mgrid[0:H, 0:W]
    map_x = (col + disp_b[:, :, 0]).astype(np.float32)
    map_y = (row + disp_b[:, :, 1]).astype(np.float32)
    a_x = cv2.remap(disp_a[:, :, 0], map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    a_y = cv2.remap(disp_a[:, :, 1], map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    out = np.empty_like(disp_b)
    out[:, :, 0] = disp_b[:, :, 0] + a_x
    out[:, :, 1] = disp_b[:, :, 1] + a_y
    return out


# ── Pre-processing ─────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _match_size(moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Resize moving to match fixed dimensions if they differ."""
    fh, fw = fixed.shape[:2]
    mh, mw = moving.shape[:2]
    if (mh, mw) != (fh, fw):
        interp = cv2.INTER_AREA if mh > fh else cv2.INTER_LINEAR
        moving = cv2.resize(moving, (fw, fh), interpolation=interp)
    return moving


def _bg_color(img: np.ndarray) -> tuple:
    """Compute per-channel mode value of an image to use as background fill color."""
    if img.ndim == 2:
        vals, counts = np.unique(img.ravel(), return_counts=True)
        return int(vals[np.argmax(counts)])
    result = []
    for c in range(img.shape[2]):
        vals, counts = np.unique(img[:, :, c].ravel(), return_counts=True)
        result.append(int(vals[np.argmax(counts)]))
    return tuple(result)


# ── Affine Registration ───────────────────────────────────────────────────────

def estimate_affine_transform(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: Optional[np.ndarray] = None,
    moving_mask: Optional[np.ndarray] = None,
    method: str = "ECC",
    n_iterations: int = 200,
    termination_eps: float = 1e-7,
) -> Tuple[np.ndarray, float]:
    """Estimate a 2×3 affine matrix aligning *moving* to *fixed*.

    Methods:
        "ECC"      – Enhanced Correlation Coefficient (intensity-based, sub-pixel).
        "ORB"      – Feature-based using ORB descriptors + RANSAC.
        "combined" – Try ORB first for a coarse estimate, then refine with ECC.

    Returns:
        (M, score) where M is 2×3 float64 affine matrix and score is a
        quality metric (ECC correlation or inlier ratio).
    """
    fh, fw = fixed_gray.shape[:2]

    if method in ("ORB", "combined"):
        M_orb, orb_score = _affine_orb(fixed_gray, moving_gray,
                                        fixed_mask, moving_mask)
    else:
        M_orb, orb_score = None, 0.0

    if method == "ORB":
        if M_orb is not None:
            return M_orb, orb_score
        else:
            return np.eye(2, 3, dtype=np.float64), 0.0

    # ECC-based refinement
    # Blur slightly for robustness
    f_blur = cv2.GaussianBlur(fixed_gray, (5, 5), 0)
    m_blur = cv2.GaussianBlur(moving_gray, (5, 5), 0)

    # Multi-scale ECC: start coarse, refine
    scales = [0.25, 0.5, 1.0]
    warp_matrix = np.eye(2, 3, dtype=np.float64)
    if M_orb is not None:
        warp_matrix = M_orb.copy()

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                n_iterations, termination_eps)

    best_cc = -1.0
    for s in scales:
        sh, sw = int(fh * s), int(fw * s)
        if sh < 32 or sw < 32:
            continue

        f_s = cv2.resize(f_blur, (sw, sh), interpolation=cv2.INTER_AREA)
        m_s = cv2.resize(m_blur, (sw, sh), interpolation=cv2.INTER_AREA)

        # Build scaled warp matrix
        S = np.array([[s, 0, 0], [0, s, 0]], dtype=np.float64)
        S_inv = np.array([[1/s, 0, 0], [0, 1/s, 0]], dtype=np.float64)
        M_s = S @ np.vstack([warp_matrix, [0, 0, 1]])
        # Actually we need: dst(x) = src(M^-1 * x), but ECC convention is
        # warpAffine-style: dst(x) = src(M * x) i.e. M maps dst→src.
        # Scale translation component
        M_scaled = warp_matrix.copy()
        M_scaled[0, 2] *= s
        M_scaled[1, 2] *= s

        mask_s = None
        if fixed_mask is not None:
            mask_s = cv2.resize(fixed_mask.astype(np.uint8), (sw, sh),
                                interpolation=cv2.INTER_NEAREST)

        try:
            cc, M_scaled = cv2.findTransformECC(
                f_s, m_s, M_scaled, cv2.MOTION_AFFINE, criteria,
                inputMask=mask_s, gaussFiltSize=5,
            )
        except cv2.error:
            continue

        # Un-scale
        warp_matrix = M_scaled.copy()
        warp_matrix[0, 2] /= s
        warp_matrix[1, 2] /= s
        best_cc = cc

    return warp_matrix, best_cc


def _affine_orb(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: Optional[np.ndarray] = None,
    moving_mask: Optional[np.ndarray] = None,
    n_features: int = 5000,
) -> Tuple[Optional[np.ndarray], float]:
    """ORB feature-based affine estimation."""
    orb = cv2.ORB_create(nfeatures=n_features)

    fm = fixed_mask.astype(np.uint8) * 255 if fixed_mask is not None else None
    mm = moving_mask.astype(np.uint8) * 255 if moving_mask is not None else None

    kp1, des1 = orb.detectAndCompute(fixed_gray, fm)
    kp2, des2 = orb.detectAndCompute(moving_gray, mm)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 6:
        return None, 0.0

    pts_f = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_m = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Full 6-DOF affine (translation + rotation + scale + shear).
    M, inliers = cv2.estimateAffine2D(pts_m, pts_f, method=cv2.RANSAC,
                                      ransacReprojThreshold=5.0)
    if M is None:
        return None, 0.0

    inlier_ratio = float(inliers.sum()) / len(good)
    return M, inlier_ratio


def apply_affine(
    image: np.ndarray,
    M: np.ndarray,
    output_shape: Tuple[int, int],
    border_value=255,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Apply a 2×3 affine matrix to an image.

    Args:
        image:        Input image (H, W, C) or (H, W).
        M:            2×3 affine matrix.
        output_shape: (height, width) of the output.
        border_value: Value for out-of-bounds pixels.

    Returns:
        Warped image.
    """
    h, w = output_shape
    # Ensure M is a proper (2, 3) float64 numpy array
    if not isinstance(M, np.ndarray):
        M = np.array(M, dtype=np.float64)
    else:
        M = M.astype(np.float64)
    if M.shape == (3, 3):
        M = M[:2, :]
    if M.shape != (2, 3):
        raise ValueError(f"apply_affine: unexpected matrix shape {M.shape}, expected (2, 3)")
    if image.ndim == 3:
        bv = (border_value,) * image.shape[2] if isinstance(border_value, int) else border_value
    else:
        bv = border_value
    return cv2.warpAffine(image, M, (w, h),
                          flags=interpolation,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bv)


# ── Elastic: sparse-grid phase-correlation (MATLAB port) ──────────────────────

def _phase_correlation_shift(patch_fixed: np.ndarray, patch_moving: np.ndarray) -> Tuple[float, float]:
    """Sub-pixel translation estimate via phase correlation.

    Uses cv2.phaseCorrelate (Hanning-windowed cross-power spectrum with
    sub-pixel peak fitting), which is far more accurate than a hand-rolled
    argmax over the inverse FFT — the latter has no windowing, so spectral
    leakage biases the peak toward (0, 0) and yields near-zero shifts.

    Returns (dx, dy) such that patch_moving shifted by (dx, dy) aligns to
    patch_fixed.
    """
    f = np.ascontiguousarray(patch_fixed, dtype=np.float32)
    m = np.ascontiguousarray(patch_moving, dtype=np.float32)
    win = cv2.createHanningWindow((f.shape[1], f.shape[0]), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(f, m, win)
    return float(dx), float(dy)


def _reg_ims_els(patch_moving: np.ndarray, patch_fixed: np.ndarray, downsample: int = 2) -> Tuple[float, float]:
    """Port of MATLAB reg_ims_ELS: bidirectional phase correlation, averaged.

    Args:
        patch_moving: (sz, sz) float32 moving patch.
        patch_fixed:  (sz, sz) float32 fixed patch.
        downsample:   Downsample factor before correlation (speed).

    Returns:
        (X, Y): displacement in original (full) pixel coordinates,
                moving → fixed direction.
    """
    h, w = patch_fixed.shape
    nh, nw = max(h // downsample, 4), max(w // downsample, 4)
    f_ds = cv2.resize(patch_fixed,  (nw, nh), interpolation=cv2.INTER_AREA)
    m_ds = cv2.resize(patch_moving, (nw, nh), interpolation=cv2.INTER_AREA)

    # phaseCorrelate(a, b) returns the shift to apply to b to align it to a.
    # The two calls measure the SAME displacement in opposite directions
    # (fixed->moving vs moving->fixed), so to average them we must subtract:
    #   est = (shift(fixed,moving) - shift(moving,fixed)) / 2
    # Adding them cancels the signal to ~0 (they are equal and opposite).
    dx1, dy1 = _phase_correlation_shift(f_ds, m_ds)
    dx2, dy2 = _phase_correlation_shift(m_ds, f_ds)

    dx = (dx1 - dx2) / 2
    dy = (dy1 - dy2) / 2

    # Scale back to full resolution. The displacement is consumed by
    # apply_elastic_transform via cv2.remap (map = coord + disp), which is a
    # pullback, so the sign here is correct as-is (no negation).
    X = dx * downsample
    Y = dy * downsample
    return X, Y


def elastic_registration(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: Optional[np.ndarray] = None,
    moving_mask: Optional[np.ndarray] = None,
    tile_size: int = 400,
    buffer: int = 200,
    grid_spacing: int = 150,
    tissue_cutoff: float = 0.15,
    downsample: int = 2,
    smooth_sigma: float = 2.0,
    rbf_smoothing: float = 0.1,
) -> np.ndarray:
    """Sparse-grid elastic registration via tile-wise phase correlation.

    Direct port of MATLAB calculate_elastic_registration / reg_ims_ELS.

    Args:
        fixed_gray:     Fixed image (H, W) uint8.
        moving_gray:    Moving image already affine-aligned (H, W) uint8.
        fixed_mask:     Boolean tissue mask for fixed (H, W). If None, all tissue.
        moving_mask:    Boolean tissue mask for moving (H, W). If None, all tissue.
        tile_size:      Side length of registration tiles in pixels (MATLAB szE=400).
        buffer:         Buffer padding around image before registration (MATLAB bfE=200).
        grid_spacing:   Distance between tile centres (MATLAB diE=150).
        tissue_cutoff:  Min fraction of tissue in both tiles to attempt registration.
        downsample:     Downsample factor inside each tile for speed (MATLAB rf=2).
        smooth_sigma:   Gaussian smoothing sigma applied to the sparse displacement grid.

    Returns:
        Dense displacement field (H, W, 2) float32, channel 0 = dx, channel 1 = dy.
        Apply with apply_elastic_transform().
    """
    H, W = fixed_gray.shape[:2]
    sz = int(tile_size)    # guard against float being passed in
    bf = int(buffer)
    di = int(grid_spacing)
    m = (sz - 1) // 2  # half-tile radius (MATLAB: m=(sz-1)/2+1, 1-indexed)

    # ── Pad + blur (MATLAB: padarray + imgaussfilt) ────────────────────────
    pad_val_f = int(np.median(fixed_gray))
    pad_val_m = int(np.median(moving_gray))

    f_pad = cv2.copyMakeBorder(fixed_gray,  bf, bf, bf, bf, cv2.BORDER_CONSTANT, value=pad_val_f)
    m_pad = cv2.copyMakeBorder(moving_gray, bf, bf, bf, bf, cv2.BORDER_CONSTANT, value=pad_val_m)
    f_pad = cv2.GaussianBlur(f_pad.astype(np.float32), (0, 0), sigmaX=3)
    m_pad = cv2.GaussianBlur(m_pad.astype(np.float32), (0, 0), sigmaX=3)

    if fixed_mask is not None:
        fmask_pad = cv2.copyMakeBorder(fixed_mask.astype(np.uint8), bf, bf, bf, bf,
                                        cv2.BORDER_CONSTANT, value=0).astype(bool)
    else:
        fmask_pad = np.ones(f_pad.shape[:2], dtype=bool)

    if moving_mask is not None:
        mmask_pad = cv2.copyMakeBorder(moving_mask.astype(np.uint8), bf, bf, bf, bf,
                                        cv2.BORDER_CONSTANT, value=0).astype(bool)
    else:
        mmask_pad = np.ones(m_pad.shape[:2], dtype=bool)

    pH, pW = f_pad.shape[:2]

    # ── Build sparse grid (MATLAB: meshgrid with random offset) ───────────
    SENTINEL = -5000.0

    rng = np.random.default_rng(42)
    y0 = int(rng.integers(1, max(di // 2, 2))) + bf + m
    x0 = int(rng.integers(1, max(di // 2, 2))) + bf + m

    ys = np.arange(y0, pH - m - bf, di)
    xs = np.arange(x0, pW - m - bf, di)
    xg, yg = np.meshgrid(xs, ys)          # shape (ny, nx)
    ny, nx = xg.shape

    disp_x = np.full((ny, nx), SENTINEL, dtype=np.float32)
    disp_y = np.full((ny, nx), SENTINEL, dtype=np.float32)

    for iy in range(ny):
        for ix in range(nx):
            cy, cx = int(yg[iy, ix]), int(xg[iy, ix])

            # Extract tile
            y1, y2 = cy - m, cy - m + sz
            x1, x2 = cx - m, cx - m + sz
            if y2 > pH or x2 > pW or y1 < 0 or x1 < 0:
                continue

            # Tissue check (MATLAB: checkS = min(mvS, rfS) / sz^2 > cutoff)
            # Inner region (exclude 10-px border) for moving mask check
            cc = 10
            fm_tile = fmask_pad[y1:y2, x1:x2]
            mm_tile = mmask_pad[y1:y2, x1:x2]
            mm_inner = mm_tile[cc:-cc, cc:-cc]

            rf_cov  = fm_tile.sum()
            mv_cov  = mm_inner.sum() if mm_inner.size > 0 else 0
            check   = min(rf_cov, mv_cov) / (sz * sz)

            if check <= tissue_cutoff:
                continue

            patch_f = f_pad[y1:y2, x1:x2]
            patch_m = m_pad[y1:y2, x1:x2]

            try:
                dx, dy = _reg_ims_els(patch_m, patch_f, downsample=downsample)
            except Exception:
                continue

            disp_x[iy, ix] = dx
            disp_y[iy, ix] = dy

    # ── Smooth + interpolate sparse → dense (MATLAB: make_final_grids) ────
    dense = _make_dense_field(disp_x, disp_y, xg, yg, H, W, bf,
                               SENTINEL, smooth_sigma, rbf_smoothing)
    return dense


def elastic_registration_multipass(
    fixed_gray: np.ndarray,
    moving_gray: np.ndarray,
    fixed_mask: Optional[np.ndarray] = None,
    moving_mask: Optional[np.ndarray] = None,
    tile_size: int = 400,
    buffer: int = 200,
    grid_spacing: int = 150,
    tissue_cutoff: float = 0.15,
    downsample: int = 2,
    smooth_sigma: float = 2.0,
    rbf_smoothing: float = 0.1,
    n_passes: int = 3,
    spacing_decay: float = 0.6,
    tile_decay: float = 0.7,
    keep_if_improves: bool = True,
    min_ncc_gain: float = 1e-4,
    verbose: bool = False,
) -> np.ndarray:
    """Iterative coarse-to-fine elastic registration.

    Runs `elastic_registration` repeatedly. Each pass measures the *residual*
    misalignment on the already-warped moving image, then composes the new
    field onto the accumulated one. Grid spacing and tile size shrink each
    pass (coarse-to-fine), so early passes fix large smooth warps and later
    passes fix fine local detail — yielding much stronger alignment than a
    single pass for near-identical serial sections.

    A pass is kept only if it improves masked grayscale NCC (when
    `keep_if_improves`), so an occasional bad pass (low-texture region,
    spurious tile) cannot degrade a good alignment.

    Args:
        n_passes:      Number of refinement passes.
        spacing_decay: Per-pass multiplier on grid_spacing (finer over time).
        tile_decay:    Per-pass multiplier on tile_size.
        keep_if_improves: Revert a pass if NCC does not improve.
        min_ncc_gain:  Minimum NCC improvement to accept a pass.

    Returns:
        Accumulated dense displacement field (H, W, 2) float32.
    """
    H, W = fixed_gray.shape[:2]
    accum = np.zeros((H, W, 2), dtype=np.float32)

    def _warp_gray(disp):
        return apply_elastic_transform(
            moving_gray, disp, (H, W),
            default_value=float(np.median(moving_gray)),
        )

    def _warp_mask(disp):
        if moving_mask is None:
            return None
        return apply_elastic_transform(
            moving_mask.astype(np.uint8), disp, (H, W),
            default_value=0.0, is_label=True,
        ).astype(bool)

    # Score region: intersection of both tissue masks (where comparison is meaningful)
    def _score(warped_gray, warped_mask):
        if fixed_mask is not None and warped_mask is not None:
            region = fixed_mask & warped_mask
        elif fixed_mask is not None:
            region = fixed_mask
        else:
            region = warped_mask
        return masked_ncc(fixed_gray, warped_gray, region)

    cur_ncc = _score(moving_gray, moving_mask)
    if verbose:
        print(f"    [elastic-mp] start NCC={cur_ncc:.4f}", flush=True)

    sp = float(grid_spacing)
    ts = float(tile_size)
    for p in range(n_passes):
        sp_p = max(int(round(sp)), 8)
        ts_p = max(int(round(ts)), 32)

        warped_moving = _warp_gray(accum)
        warped_mvmask = _warp_mask(accum)

        residual = elastic_registration(
            fixed_gray, warped_moving,
            fixed_mask=fixed_mask,
            moving_mask=warped_mvmask if warped_mvmask is not None else None,
            tile_size=ts_p, buffer=buffer, grid_spacing=sp_p,
            tissue_cutoff=tissue_cutoff, downsample=downsample,
            smooth_sigma=smooth_sigma, rbf_smoothing=rbf_smoothing,
        )

        candidate = compose_displacements(accum, residual)
        cand_gray = _warp_gray(candidate)
        cand_mask = _warp_mask(candidate)
        cand_ncc = _score(cand_gray, cand_mask)

        gain = cand_ncc - cur_ncc
        accept = (not keep_if_improves) or (gain >= min_ncc_gain)
        if verbose:
            tag = "keep" if accept else "drop"
            print(f"    [elastic-mp] pass {p+1}/{n_passes} "
                  f"tile={ts_p} spacing={sp_p} NCC {cur_ncc:.4f}->{cand_ncc:.4f} "
                  f"({gain:+.4f}) [{tag}]", flush=True)

        if accept:
            accum = candidate
            cur_ncc = cand_ncc
        # else: keep accum, but still shrink grid for the next try

        sp *= spacing_decay
        ts *= tile_decay

    return accum


def _make_dense_field(
    disp_x: np.ndarray,
    disp_y: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    H: int, W: int, bf: int,
    sentinel: float,
    smooth_sigma: float,
    rbf_smoothing: float = 0.1,
) -> np.ndarray:
    """Smooth valid sparse displacements and interpolate to a dense (H, W, 2) field."""

    # Collect valid points
    valid = (disp_x != sentinel) & (disp_y != sentinel)
    if valid.sum() < 4:
        # Not enough points — return zero field
        return np.zeros((H, W, 2), dtype=np.float32)

    # Grid coordinates in *unpadded* image space
    pts = np.stack([xg[valid] - bf, yg[valid] - bf], axis=-1)   # (N, 2): col, row
    vals_x = disp_x[valid]
    vals_y = disp_y[valid]

    # Gaussian-smooth the values at valid points (approximates MATLAB imgaussfilt on grid)
    # We do this by smoothing the full grid arrays then re-sampling
    dx_grid = np.where(disp_x != sentinel, disp_x, 0.0)
    dy_grid = np.where(disp_y != sentinel, disp_y, 0.0)
    w_grid  = valid.astype(np.float32)

    dx_smooth = gaussian_filter(dx_grid * w_grid, sigma=smooth_sigma, mode="nearest")
    dy_smooth = gaussian_filter(dy_grid * w_grid, sigma=smooth_sigma, mode="nearest")
    w_smooth  = gaussian_filter(w_grid,            sigma=smooth_sigma, mode="nearest")
    w_smooth  = np.maximum(w_smooth, 1e-8)
    dx_smooth /= w_smooth
    dy_smooth /= w_smooth

    # Re-sample smoothed values at valid locations
    vals_xs = dx_smooth[valid]
    vals_ys = dy_smooth[valid]

    # RBF interpolation (thin-plate-spline) to a dense field.
    #
    # The displacement field is smooth by construction (sparse control points,
    # heavily regularised), so we evaluate the expensive TPS on a coarse grid
    # and bilinearly upsample to full (H, W). Evaluating TPS at every pixel is
    # O(H*W*N) and dominated runtime; the coarse-eval result is visually
    # identical but 1-2 orders of magnitude faster.
    src_pts = pts.astype(np.float64)

    # Coarse grid resolution: cap the longest side so cost is independent of
    # the working level / image size. ~512 px on the long edge is plenty for a
    # field this smooth.
    max_coarse = 512
    scale = min(1.0, max_coarse / max(H, W))
    cH = max(int(round(H * scale)), 2)
    cW = max(int(round(W * scale)), 2)

    # Coarse query points in full-resolution coordinates.
    coarse_cols = np.linspace(0, W - 1, cW)
    coarse_rows = np.linspace(0, H - 1, cH)
    qc, qr = np.meshgrid(coarse_cols, coarse_rows)
    query_pts = np.stack([qc.ravel(), qr.ravel()], axis=-1).astype(np.float64)

    try:
        rbf_x = RBFInterpolator(src_pts, vals_xs, kernel="thin_plate_spline", smoothing=rbf_smoothing)
        rbf_y = RBFInterpolator(src_pts, vals_ys, kernel="thin_plate_spline", smoothing=rbf_smoothing)
        coarse_x = rbf_x(query_pts).reshape(cH, cW).astype(np.float32)
        coarse_y = rbf_y(query_pts).reshape(cH, cW).astype(np.float32)
    except Exception:
        return np.zeros((H, W, 2), dtype=np.float32)

    # Upsample coarse field to full resolution (displacement magnitudes are in
    # pixels and resolution-independent, so no rescaling of values is needed).
    if (cH, cW) != (H, W):
        dense_x = cv2.resize(coarse_x, (W, H), interpolation=cv2.INTER_LINEAR)
        dense_y = cv2.resize(coarse_y, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        dense_x, dense_y = coarse_x, coarse_y

    return np.stack([dense_x, dense_y], axis=-1)


def apply_elastic_transform(
    image: np.ndarray,
    displacement: np.ndarray,
    reference_shape: Tuple[int, int],
    default_value = 255.0,
    is_label: bool = False,
) -> np.ndarray:
    """Apply a dense displacement field (H, W, 2) to an image using cv2.remap.

    Args:
        image:           (H, W) or (H, W, C) uint8.
        displacement:    (H, W, 2) float32 field; ch0=dx (col shift), ch1=dy (row shift).
        reference_shape: (H, W) of the output.
        default_value:   Fill value for out-of-bounds pixels (scalar or per-channel tuple).
        is_label:        Use nearest-neighbour interpolation.

    Returns:
        Warped image as uint8.
    """
    rH, rW = reference_shape
    dH, dW = displacement.shape[:2]

    # Scale displacement field if it doesn't match reference_shape
    if (dH, dW) != (rH, rW):
        scale_y = rH / dH
        scale_x = rW / dW
        disp_x = cv2.resize(displacement[:, :, 0], (rW, rH), interpolation=cv2.INTER_LINEAR) * scale_x
        disp_y = cv2.resize(displacement[:, :, 1], (rW, rH), interpolation=cv2.INTER_LINEAR) * scale_y
        displacement = np.stack([disp_x, disp_y], axis=-1)

    row_coords, col_coords = np.mgrid[0:rH, 0:rW]
    map_x = (col_coords + displacement[:, :, 0]).astype(np.float32)
    map_y = (row_coords + displacement[:, :, 1]).astype(np.float32)

    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR

    if image.ndim == 3:
        if isinstance(default_value, (tuple, list)):
            fill_values = list(default_value)
        else:
            fill_values = [int(default_value)] * image.shape[2]
        channels = []
        for c in range(image.shape[2]):
            ch = cv2.remap(image[:, :, c], map_x, map_y,
                           interpolation=interp,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=fill_values[c])
            channels.append(ch)
        return np.stack(channels, axis=-1)
    else:
        return cv2.remap(image, map_x, map_y,
                         interpolation=interp,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=int(default_value if not isinstance(default_value, (tuple, list)) else default_value[0]))


# ── Combined registration pipeline for a pair ─────────────────────────────────

def register_pair(
    fixed: np.ndarray,
    moving: np.ndarray,
    g_thresh: int = 170,
    affine_method: str = "ORB",
    do_elastic: bool = True,
    tile_size: int = 400,
    buffer: int = 200,
    grid_spacing: int = 150,
    tissue_cutoff: float = 0.15,
    iou_threshold: float = 0.0,
    elastic_passes: int = 3,
    rbf_smoothing: float = 0.1,
    keep_if_improves: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Full registration pipeline for one (fixed, moving) pair.

    IoU is evaluated after affine. Elastic registration only runs if the
    affine IoU meets iou_threshold (it is a refinement step, not a recovery step).

    Returns:
        (aligned_image, aligned_mask, iou_after_affine, affine_M, displacement_field)
        displacement_field is (H, W, 2) float32 or None.
    """
    moving = _match_size(moving, fixed)
    fh, fw = fixed.shape[:2]

    # Compute background fill color from the moving image (mode per channel)
    bg = _bg_color(moving)

    fixed_mask  = tissue_mask(fixed, g_thresh)
    moving_mask = tissue_mask(moving, g_thresh)

    fixed_gray  = _to_gray(fixed)
    moving_gray = _to_gray(moving)

    # 1. Affine
    if verbose:
        print(f"    [affine/{affine_method}] estimating transform ...", flush=True)
    M, affine_score = estimate_affine_transform(
        fixed_gray, moving_gray,
        fixed_mask=fixed_mask.astype(np.uint8),
        moving_mask=moving_mask.astype(np.uint8),
        method=affine_method,
    )
    if verbose:
        print(f"    [affine/{affine_method}] done (score={affine_score:.4f})", flush=True)

    aligned = apply_affine(moving, M, (fh, fw), border_value=bg)
    aligned_mask_u8 = apply_affine(
        moving_mask.astype(np.uint8), M, (fh, fw),
        border_value=0, interpolation=cv2.INTER_NEAREST,
    )
    aligned_mask = aligned_mask_u8.astype(bool)

    iou_affine = compute_iou(fixed_mask, aligned_mask)
    if verbose:
        print(f"    [affine] IoU = {iou_affine:.4f}", flush=True)

    # 2. Elastic — only runs if affine IoU meets the threshold
    displacement = None
    iou = iou_affine
    if do_elastic:
        if iou_affine < iou_threshold:
            if verbose:
                print(f"    [elastic] skipped (affine IoU {iou_affine:.4f} < threshold {iou_threshold})", flush=True)
        else:
            if verbose:
                print(f"    [elastic] tile={tile_size} buf={buffer} spacing={grid_spacing} "
                      f"passes={elastic_passes} smoothing={rbf_smoothing} ...", flush=True)
            aligned_gray = _to_gray(aligned)
            displacement = elastic_registration_multipass(
                fixed_gray, aligned_gray,
                fixed_mask=fixed_mask,
                moving_mask=aligned_mask,
                tile_size=tile_size,
                buffer=buffer,
                grid_spacing=grid_spacing,
                tissue_cutoff=tissue_cutoff,
                rbf_smoothing=rbf_smoothing,
                n_passes=elastic_passes,
                keep_if_improves=keep_if_improves,
                verbose=verbose,
            )
            aligned = apply_elastic_transform(aligned, displacement, (fh, fw),
                                              default_value=bg)
            aligned_mask = apply_elastic_transform(
                aligned_mask.astype(np.uint8), displacement, (fh, fw),
                default_value=0.0, is_label=True,
            ).astype(bool)
            iou = compute_iou(fixed_mask, aligned_mask)
            if verbose:
                print(f"    [elastic] done (IoU {iou_affine:.4f} -> {iou:.4f})", flush=True)

    return aligned, aligned_mask, iou, M, displacement


def crop_to_valid(
    images: list,
    fill_value: int = 0,
) -> list:
    """Crop a list of same-shape images to their common non-fill bounding box.

    Finds the largest rectangle that contains non-fill pixels in ALL images,
    then crops all images to that rectangle.

    Args:
        images:     List of (H, W) or (H, W, C) uint8 arrays, all same shape.
        fill_value: Border fill value to ignore (0 for black, 255 for white).

    Returns:
        List of cropped arrays.
    """
    if not images:
        return images

    H, W = images[0].shape[:2]
    # For each image, find the bounding box of non-fill pixels
    row_min_all, row_max_all = 0, H
    col_min_all, col_max_all = 0, W

    for img in images:
        if img.ndim == 3:
            mask = np.any(img != fill_value, axis=2)
        else:
            mask = img != fill_value

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any():
            continue

        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]

        # Intersection: take the most restrictive bounds across all images
        row_min_all = max(row_min_all, r0)
        row_max_all = min(row_max_all, r1 + 1)
        col_min_all = max(col_min_all, c0)
        col_max_all = min(col_max_all, c1 + 1)

    if row_min_all >= row_max_all or col_min_all >= col_max_all:
        return images  # No valid crop possible

    return [img[row_min_all:row_max_all, col_min_all:col_max_all] for img in images]
