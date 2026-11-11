"""
Rigid registration of tissue masks by FFT cross-correlation.

The default first step of the two-step (rigid -> elastic) serial-section
registration. Ported from the rigid_alignment_model study (scratch.py,
notebooks 02-04).

For every candidate rotation the moving mask is rotated about the canvas centre,
and its overlap with the fixed mask is computed for *every* integer translation
at once by FFT cross-correlation. The best (angle, shift) maximises mask Dice.
The search runs coarse to fine:

1. Centre both masks on a shared square canvas (``center_pair``).
2. Search the full circle every ``coarse_step`` degrees on a canvas shrunk to
   ``coarse_size`` px.
3. Search +/- ``fine_range`` degrees around the best coarse angle in
   ``fine_step`` steps at full resolution.
4. Refine the angle with a parabola through the best three fine scores, and
   the shift with parabolas through the correlation peak (sub-pixel).

Because it matches tissue *silhouettes* exhaustively, it is not misled by
repeated texture the way feature matching can be. It recovers any rotation
(-180..180 degrees), and it reports a QC margin that shows when another
rotation fits almost as well.

Conventions
-----------
- Masks are bool arrays (H, W) indexed [y, x]. Points are (x, y) pixel
  coordinates, with y pointing down.
- Transforms are 3x3 homogeneous matrices acting on column vectors (x, y, 1).
  The returned ``T`` maps *moving* pixel coordinates to *fixed* pixel
  coordinates, so ``cv2.warpAffine(moving, T[:2], (w_fixed, h_fixed))`` lands
  on fixed. This is the same convention as ``registration.apply_affine``.
- Fixed and moving may have different shapes.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import fft as sp_fft
from scipy import ndimage as ndi


# ── Transforms ─────────────────────────────────────────────────────────────────

def image_center(shape) -> np.ndarray:
    h, w = shape[:2]
    return np.array([(w - 1) / 2, (h - 1) / 2])


def rigid_matrix(theta_deg: float, tx: float, ty: float, center) -> np.ndarray:
    """Rotation by theta_deg about `center`, followed by translation (tx, ty).

    Because y points down, a positive angle rotates clockwise on screen.
    """
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.asarray(center, dtype=float)
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2] = c - R @ c + (tx, ty)
    return T


def flip_matrix(width: int) -> np.ndarray:
    """Mirror x -> (width - 1) - x (a section mounted face down)."""
    return np.array([[-1.0, 0, width - 1], [0, 1, 0], [0, 0, 1]])


def tissue_points(mask: np.ndarray) -> np.ndarray:
    """(N, 2) array of (x, y) coordinates of the True pixels."""
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], 1).astype(float)


# ── Tissue mask (automatic green-channel threshold) ───────────────────────────

def _fill_color(rgb: np.ndarray) -> np.ndarray:
    """Most common value of each channel: the glass, or the constant padding / warp fill."""
    return np.array([np.bincount(rgb[..., c].ravel(), minlength=256).argmax() for c in range(3)],
                    dtype=rgb.dtype)


def outside_scan(rgb: np.ndarray, open_px: int = 3) -> np.ndarray:
    """Pixels outside the scanned area, as a bool mask.

    These are solid areas of exactly the fill colour that connect to the image
    edge: CODA's padding, the fill around an already-warped section, or
    unscanned slide area. The opening drops glass pixels that only happen to
    equal the fill colour.
    """
    f = (rgb == _fill_color(rgb)).all(-1).astype(np.uint8)
    f = cv2.morphologyEx(f, cv2.MORPH_OPEN, np.ones((2 * open_px + 1,) * 2, np.uint8))
    lab, _ = ndi.label(f)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, edge[edge > 0])


def green_threshold(rgb: np.ndarray, method: str = "triangle", blur_sigma: float = 1.0,
                    min_bg_margin: int = 6, exclude: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Automatic per-image threshold on the green channel. Returns (blurred green, threshold).

    Both stains absorb green, so tissue is darker than glass. "triangle" puts the
    cut at the foot of the background peak, which keeps pale tissue such as fat.
    "otsu" splits dense tissue from everything else and tends to lose fat. The
    cut is kept at least min_bg_margin grey levels below the background peak so
    glass noise stays out.

    `exclude` (bool mask) leaves pixels out of the histogram, e.g. outside_scan.
    Padding makes the background peak taller, and the triangle cut then drifts
    toward glass, pulling faint glass noise into the mask.
    """
    g = cv2.GaussianBlur(rgb[..., 1], (0, 0), blur_sigma) if blur_sigma else rgb[..., 1]
    vals = g
    if exclude is not None and (~exclude).sum() >= 256:
        vals = g[~exclude].reshape(-1, 1)
    flag = {"triangle": cv2.THRESH_TRIANGLE, "otsu": cv2.THRESH_OTSU}[method]
    thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + flag)
    bg_peak = np.bincount(vals.ravel(), minlength=256).argmax()
    return g, min(thr, bg_peak - min_bg_margin)


def auto_tissue_mask(rgb: np.ndarray, method: str = "triangle", offset: int = 0, close_px: int = 3,
                     min_object_frac: float = 0.002, min_object_px: int = 50, drop_border: bool = True,
                     max_hole_px: int = 400, ignore_fill: bool = True, **threshold_kw) -> np.ndarray:
    """Binary tissue mask of an H&E image from an automatic green-channel threshold.

    Used for the FFT rigid step. (registration.tissue_mask, a fixed threshold,
    is still used for IoU and the elastic step.)

    `offset` shifts the threshold (positive = more inclusive). Cleanup steps:
    - Close small gaps such as the fat lattice.
    - Drop pieces touching the image border (coverslip edge, slide edges),
      unless it is the largest piece.
    - Drop specks (dust) smaller than min_object_px, or smaller than
      min_object_frac of the largest piece.
    - Fill holes smaller than max_hole_px. Bigger holes, such as lumens, are
      kept because they help alignment.

    With ignore_fill, constant padding or warp fill is treated as off the slide.
    It is left out of the threshold histogram, and the scan edge it borders
    counts as image border. So a padded or warped image gets the same mask as
    the bare scan.
    """
    outside = outside_scan(rgb) if ignore_fill else None
    g, thr = green_threshold(rgb, method, exclude=outside, **threshold_kw)
    m = g < thr + offset
    if close_px:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool)
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum_labels(m, lab, np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = (sizes >= min_object_frac * sizes.max()) & (sizes >= min_object_px)
    keep[1 + sizes.argmax()] = True
    if drop_border:
        edges = [lab[0], lab[-1], lab[:, 0], lab[:, -1]]
        if outside is not None:
            # touching the scan edge = touching the image edge (+2 px for the blended edge of a warp)
            near = cv2.dilate(outside.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            edges.append(lab[near])
        border = np.unique(np.concatenate(edges))
        keep[border[border != 1 + sizes.argmax()]] = False
    keep[0] = False
    m = keep[lab]
    holes, _ = ndi.label(ndi.binary_fill_holes(m) & ~m)
    small = np.bincount(holes.ravel()) < max_hole_px
    small[0] = False
    return m | small[holes]


# ── FFT rotation search ───────────────────────────────────────────────────────

def center_pair(fixed: np.ndarray, moving: np.ndarray, margin: int = 4, multiple: int = 32):
    """Paste both masks onto one square canvas with each tissue centroid at the canvas centre.

    The canvas is just big enough that rotating either mask about the centre
    never clips it. Its side is rounded up to a multiple of `multiple`, which
    gives fast FFT sizes that repeat, so GPU FFT plans get reused. Returns
    (fixed_c, moving_c, C_f, C_m), where C maps original pixel coordinates to
    canvas coordinates (an integer shift, so no resampling).
    """
    pts = [tissue_points(m) for m in (fixed, moving)]
    r = max(np.linalg.norm(P - P.mean(0), axis=1).max() for P in pts)
    side = int(np.ceil((2 * np.ceil(r) + 2 * margin + 1) / multiple) * multiple)
    c = (side - 1) / 2
    out = []
    for P in pts:
        ox, oy = np.round(c - P.mean(0)).astype(int)
        canvas = np.zeros((side, side), bool)
        canvas[P[:, 1].astype(int) + oy, P[:, 0].astype(int) + ox] = True
        C = np.eye(3)
        C[:2, 2] = ox, oy
        out.append((canvas, C))
    (f, C_f), (m, C_m) = out
    return f, m, C_f, C_m


def _xcorr_with(F_fixed, moving, s):
    return sp_fft.irfft2(F_fixed * np.conj(sp_fft.rfft2(moving, s=s, workers=-1)), s=s, workers=-1)


def _correlation_peak(c):
    """Best (dx, dy) in an un-shifted circular cross-correlation, refined to sub-pixel by parabola fits."""
    H, W = c.shape
    i, j = np.unravel_index(np.argmax(c), c.shape)
    v = c[i, j]

    def offset(before, after):
        d = before - 2 * v + after
        return 0.5 * (before - after) / d if d < 0 else 0.0

    dy = (i - H if i >= H // 2 else i) + offset(c[(i - 1) % H, j], c[(i + 1) % H, j])
    dx = (j - W if j >= W // 2 else j) + offset(c[i, (j - 1) % W], c[i, (j + 1) % W])
    return v, dx, dy


def _search_numpy(fixed, moving, angles):
    h, w = fixed.shape
    s = (2 * h, 2 * w)  # zero-pad to twice the size so shifts don't wrap around
    c = image_center(fixed.shape)
    F = sp_fft.rfft2(fixed.astype(np.float32), s=s, workers=-1)
    area = fixed.sum()
    scores, shifts = np.zeros(len(angles)), np.zeros((len(angles), 2))
    for k, a in enumerate(angles):
        m = cv2.warpAffine(moving.astype(np.float32), rigid_matrix(a, 0, 0, c)[:2], (w, h),
                           flags=cv2.INTER_LINEAR)
        v, dx, dy = _correlation_peak(_xcorr_with(F, m, s))
        scores[k], shifts[k] = 2 * v / (area + m.sum()), (dx, dy)
    return scores, shifts


def _search_torch(fixed, moving, angles, device=None):
    """Same as _search_numpy, with all angles of a batch rotated and correlated at once on the GPU."""
    import torch
    import torch.nn.functional as F

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    h, w = fixed.shape
    H, W = 2 * h, 2 * w
    f = torch.as_tensor(np.asarray(fixed, np.float32), device=device)
    m = torch.as_tensor(np.asarray(moving, np.float32), device=device)[None, None]
    Ff = torch.fft.rfft2(f, s=(H, W))
    cx, cy = (w - 1) / 2, (h - 1) / 2
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32) - cy,
                            torch.arange(w, device=device, dtype=torch.float32) - cx, indexing="ij")
    batch = max(1, 2 ** 24 // (h * w))
    scores, shifts = [], []
    for start in range(0, len(angles), batch):
        th = torch.as_tensor(np.deg2rad(angles[start:start + batch]), dtype=torch.float32, device=device)
        cos, sin = th.cos()[:, None, None], th.sin()[:, None, None]
        # output pixel x samples the input at R^T (x - c) + c, in grid_sample's [-1, 1] coordinates
        sx = (cos * xs + sin * ys + cx) * 2 / (w - 1) - 1
        sy = (-sin * xs + cos * ys + cy) * 2 / (h - 1) - 1
        mr = F.grid_sample(m.expand(len(th), -1, -1, -1), torch.stack([sx, sy], -1),
                           mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0]
        c = torch.fft.irfft2(Ff * torch.fft.rfft2(mr, s=(H, W)).conj(), s=(H, W))

        v, idx = c.reshape(len(th), -1).max(1)
        i, j = idx // W, idx % W
        b = torch.arange(len(th), device=device)

        def offset(before, after):
            d = before - 2 * v + after
            return torch.where(d < 0, 0.5 * (before - after) / torch.where(d < 0, d, -1.0), 0.0)

        dy = torch.where(i >= H // 2, i - H, i) + offset(c[b, (i - 1) % H, j], c[b, (i + 1) % H, j])
        dx = torch.where(j >= W // 2, j - W, j) + offset(c[b, i, (j - 1) % W], c[b, i, (j + 1) % W])
        scores.append(2 * v / (f.sum() + mr.sum((1, 2))))
        shifts.append(torch.stack([dx, dy], 1))
    return torch.cat(scores).cpu().numpy().astype(float), torch.cat(shifts).cpu().numpy().astype(float)


SEARCH_BACKENDS = {"numpy": _search_numpy, "torch": _search_torch}


def resolve_backend(backend: str = "auto") -> str:
    """"auto" -> "torch" when a CUDA GPU is available, else "numpy"."""
    if backend != "auto":
        return backend
    try:
        import torch
        return "torch" if torch.cuda.is_available() else "numpy"
    except ImportError:
        return "numpy"


def rotation_search(fixed, moving, angles, backend: str = "numpy"):
    """For each angle, rotate moving about the image centre, then find the best shift by FFT cross-correlation.

    Returns scores (Dice at the best shift) and shifts (N, 2) as (dx, dy), so the
    candidate for angle k is rigid_matrix(angles[k], *shifts[k], image_center(fixed.shape)).
    """
    return SEARCH_BACKENDS[resolve_backend(backend)](fixed, moving, np.asarray(angles, dtype=float))


def fft_align(fixed: np.ndarray, moving: np.ndarray, coarse_size: int = 128, coarse_step: float = 3.0,
              fine_range: float = 3.0, fine_step: float = 0.25, backend: str = "numpy",
              ambiguity_deg: float = 20.0):
    """Rigid alignment of two bool masks by exhaustive rotation search with FFT cross-correlation.

    Returns T (moving -> fixed, in the input pixel coordinates) and an info dict.
    info["score"] is the Dice at the solution. info["margin"] is the best coarse
    score minus the best score at least ambiguity_deg away; a small margin means
    another rotation fits nearly as well (ambiguous or unreliable).
    """
    backend = resolve_backend(backend)
    f, m, C_f, C_m = center_pair(fixed, moving)
    side = f.shape[0]
    small_side = max(8, min(side, coarse_size))
    small = [cv2.resize(x.astype(np.float32), (small_side, small_side), interpolation=cv2.INTER_AREA)
             for x in (f, m)]
    coarse_angles = np.arange(-180, 180, coarse_step)
    coarse_scores, _ = rotation_search(*small, coarse_angles, backend)
    k = coarse_scores.argmax()
    a0 = coarse_angles[k]
    far = np.abs((coarse_angles - a0 + 180) % 360 - 180) >= ambiguity_deg

    fine_angles = a0 + np.arange(-fine_range, fine_range + 1e-9, fine_step)
    fine_scores, _ = rotation_search(f, m, fine_angles, backend)
    k = fine_scores.argmax()
    angle = fine_angles[k]
    if 0 < k < len(fine_angles) - 1:
        before, v, after = fine_scores[k - 1:k + 2]
        d = before - 2 * v + after
        if d < 0:
            angle += fine_step * 0.5 * (before - after) / d
    score, shift = rotation_search(f, m, [angle], backend)

    T_canvas = rigid_matrix(angle, *shift[0], image_center(f.shape))
    T = np.linalg.inv(C_f) @ T_canvas @ C_m
    info = {
        "coarse_angles": coarse_angles, "coarse_scores": coarse_scores,
        "fine_angles": fine_angles, "fine_scores": fine_scores,
        "angle": float(angle), "score": float(score[0]),
        "margin": float(coarse_scores.max() - coarse_scores[far].max()) if far.any() else 0.0,
        "canvas_side": side, "coarse_side": small_side,
    }
    return T, info


# ── Entry point used by registration.register_pair ────────────────────────────

def fft_rigid_register(
    fixed_rgb: np.ndarray,
    moving_rgb: np.ndarray,
    coarse_size: int = 256,
    coarse_step: float = 3.0,
    fine_range: float = 3.0,
    fine_step: float = 0.25,
    backend: str = "auto",
    allow_flips: bool = False,
    flip_margin: float = 0.02,
    mask_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    """Rigid transform aligning moving_rgb to fixed_rgb, from their tissue masks.

    Args:
        fixed_rgb, moving_rgb: uint8 RGB H&E images. Their shapes may differ, and
                               neither is resampled before fitting.
        coarse_size:  Canvas side (px) of the full-circle coarse search.
        coarse_step:  Coarse angle step (degrees).
        fine_range:   Half-width (degrees) of the full-resolution fine search.
        fine_step:    Fine angle step (degrees).
        backend:      "auto" (torch on GPU if available), "torch", or "numpy".
        allow_flips:  Also fit the mirrored moving section. Use the mirrored fit
                      if its Dice beats the normal fit by more than flip_margin
                      (a section mounted face down).
        mask_kwargs:  Keyword arguments for auto_tissue_mask (e.g. offset, close_px).

    Returns:
        (T, info). T is a 3x3 matrix mapping moving pixels to fixed pixels.
        The info dict has these keys:
        - score: mask Dice at the solution.
        - margin: rotation ambiguity margin (small = unreliable).
        - angle_deg: rotation of the solution.
        - flipped: whether the mirrored fit was used.
        - dice_if_flipped: Dice of the mirrored fit, when allow_flips is set.
        - backend, seconds, fixed_mask, moving_mask.
    """
    t0 = time.perf_counter()
    mask_kwargs = mask_kwargs or {}
    fixed_mask = auto_tissue_mask(fixed_rgb, **mask_kwargs)
    moving_mask = auto_tissue_mask(moving_rgb, **mask_kwargs)
    backend = resolve_backend(backend)
    info = {"backend": backend, "fixed_mask": fixed_mask, "moving_mask": moving_mask, "flipped": False}

    if not fixed_mask.any() or not moving_mask.any():
        info.update(score=0.0, margin=0.0, angle_deg=0.0, seconds=time.perf_counter() - t0)
        return np.eye(3), info

    kw = dict(coarse_size=coarse_size, coarse_step=coarse_step, fine_range=fine_range,
              fine_step=fine_step, backend=backend)
    T, fit = fft_align(fixed_mask, moving_mask, **kw)
    info.update(score=fit["score"], margin=fit["margin"])

    if allow_flips:
        T_f, fit_f = fft_align(fixed_mask, np.ascontiguousarray(moving_mask[:, ::-1]), **kw)
        info["dice_if_flipped"] = fit_f["score"]
        if fit_f["score"] > fit["score"] + flip_margin:
            T = T_f @ flip_matrix(moving_mask.shape[1])
            info.update(score=fit_f["score"], margin=fit_f["margin"], flipped=True)

    info["angle_deg"] = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
    info["seconds"] = time.perf_counter() - t0
    return T, info
