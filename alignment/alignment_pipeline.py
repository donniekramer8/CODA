"""
Serial-section alignment pipeline.

Given a folder of .ndpi H&E images from serially cut tissue, align them
to one another starting from the middle file and working outward.
"""

import re
import os
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import cv2
import tifffile
from PIL import Image

from registration import (
    read_ndpi_at_target_level,
    get_level_dimensions,
    get_scale_factor,
    tissue_mask,
    compute_iou,
    register_pair,
    _bg_color,
)


def _read_level(path: str, level: int) -> np.ndarray:
    """Read a slide at the resolution corresponding to `level`, ensuring a
    consistent effective downsample even if the slide's pyramid doesn't have
    a proper level at that depth."""
    arr, _, _ = read_ndpi_at_target_level(path, target_level=level)
    return arr


# File discovery and sorting

def _extract_trailing_number(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    # Prefer slide-N pattern; fall back to last number in the name
    m = re.search(r'slide-(\d+)', stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s*$', stem)
    if m:
        return int(m.group(1))
    return 0

def discover_slides(folder: str, extension: str = ".ndpi") -> List[str]:
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extension)],
        key=_extract_trailing_number,
    )
    return files


# Result container

@dataclass
class AlignmentResult:
    """Stores per-slide alignment metadata."""
    slide_path: str
    index: int                  # position in sorted file list
    reference_index: int        # which slide it was aligned to (-1 = anchor)
    iou: float                  # tissue IoU after registration
    skipped: bool = False       # was this slide skipped (damaged)?
    affine_matrix: Optional[np.ndarray] = None
    notes: str = ""


# Main pipeline

def align_serial_sections(
    slide_folder: str,
    output_folder: str,
    level: int = 3,
    iou_threshold: float = 0.9,
    max_skip: int = 3,
    g_thresh: int = 170,
    affine_method: str = "ORB",
    do_elastic: bool = True,
    tile_size: int = 400,
    buffer: int = 200,
    grid_spacing: int = 150,
    tissue_cutoff: float = 0.15,
    elastic_passes: int = 3,
    rbf_smoothing: float = 0.1,
    keep_if_improves: bool = True,
    extension: str = ".ndpi",
    save_aligned: bool = True,
    save_masks: bool = False,
    padding: int = 0,
    verbose: bool = True,
) -> List[AlignmentResult]:
    """Align a folder of serial .ndpi sections middle-out.

    Algorithm:
        1. Discover and sort slides by trailing number in filename.
        2. Pick the middle slide as anchor (no transform needed).
        3. Align forward (middle+1 → end): each slide is registered to the
           previous *successfully aligned* slide.
        4. Align backward (middle-1 → start): same logic, going left.
        5. If a registration yields IoU < iou_threshold, skip ahead up to
           max_skip slides looking for a better match. Skipped slides are
           marked as damaged.

    Args:
        slide_folder:        Path to folder containing .ndpi files.
        output_folder:       Where to write aligned images and metadata.
        level:               Pyramid level to read for registration (higher = smaller).
        iou_threshold:       Minimum tissue IoU to accept a registration.
        max_skip:            Max consecutive slides to skip when IoU is too low.
        g_thresh:            Green-channel threshold for tissue masking.
        affine_method:       "ECC", "ORB", or "combined".
        do_elastic:          Whether to run elastic registration after affine.
        elastic_grid_spacing: B-spline grid spacing.
        elastic_iterations:  Max B-spline optimizer iterations.
        extension:           Slide file extension.
        save_aligned:        Save aligned images as PNG.
        save_masks:          Also save tissue masks.
        padding:             Pixels of white border added to anchor (and all moving images)
                       before registration. Prevents edge-cropping of aligned outputs.
        verbose:             Print progress.

    Returns:
        List of AlignmentResult, one per slide.
    """
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Organised subfolders
    aligned_dir    = out_dir / "aligned"
    masks_dir      = out_dir / "masks"
    transforms_dir = out_dir / "transforms"
    affine_dir     = transforms_dir / "affine"
    elastic_dir    = transforms_dir / "elastic"
    aligned_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    transforms_dir.mkdir(exist_ok=True)
    affine_dir.mkdir(exist_ok=True)
    elastic_dir.mkdir(exist_ok=True)

    slides = discover_slides(slide_folder, extension)
    n = len(slides)
    if n == 0:
        raise FileNotFoundError(f"No {extension} files found in {slide_folder}")

    if verbose:
        print(f"Found {n} slides in {slide_folder}")

    mid = n // 2
    if verbose:
        print(f"\nAnchor slide (middle): [{mid}] {Path(slides[mid]).name}")

    results: List[AlignmentResult] = [None] * n

    # Helper: pad an image with mode-colour border
    def _pad_image(img: np.ndarray) -> np.ndarray:
        if padding <= 0:
            return img
        fill = _bg_color(img)
        return cv2.copyMakeBorder(
            img, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=fill,
        )

    # Resume: load already-completed transforms
    def _is_done(idx: int) -> bool:
        return (affine_dir / f"{Path(slides[idx]).stem}.pkl").exists()

    def _load_result(idx: int) -> AlignmentResult:
        stem = Path(slides[idx]).stem
        t = load_transform(str(affine_dir), stem)
        return AlignmentResult(
            slide_path=slides[idx],
            index=idx,
            reference_index=t["reference_index"],
            iou=0.0,
            skipped=("skipped" in t.get("notes", "")),
            affine_matrix=t["affine_matrix"],
            notes=t.get("notes", "") + " (resumed)",
        )

    def _load_aligned_image(idx: int) -> Optional[np.ndarray]:
        stem = Path(slides[idx]).stem
        png_path = aligned_dir / f"{stem}.png"
        if png_path.exists():
            try:
                return np.array(Image.open(str(png_path)))
            except Exception:
                pass
        return None

    # Restore already-done results
    n_resumed = 0
    for i in range(n):
        if _is_done(i):
            results[i] = _load_result(i)
            n_resumed += 1
    if verbose and n_resumed:
        print(f"Resuming: {n_resumed}/{n} slides already processed - skipping those.")

    # Anchor
    anchor_img = None
    if _is_done(mid):
        if verbose:
            print(f"Anchor [{mid}] already done - loading from disk.")
        cached = _load_aligned_image(mid)
        anchor_img = cached if cached is not None else _pad_image(_read_level(slides[mid], level))
    else:
        anchor_raw = _read_level(slides[mid], level)
        anchor_img = _pad_image(anchor_raw)
        if verbose and padding > 0:
            print(f"  Anchor padded: {anchor_raw.shape[:2]} -> {anchor_img.shape[:2]} (padding={padding}px)")
        anchor_stem = Path(slides[mid]).stem
        if save_aligned:
            _save_image(anchor_img, aligned_dir / f"{anchor_stem}.png")
        if save_masks:
            _save_image(
                tissue_mask(anchor_img, g_thresh).astype(np.uint8) * 255,
                masks_dir / f"{anchor_stem}.png",
            )
        _save_transform(None, None, affine_dir, elastic_dir, anchor_stem,
                        slide_index=mid, reference_index=-1, level=level,
                        notes=f"anchor padding={padding}")
        results[mid] = AlignmentResult(
            slide_path=slides[mid], index=mid, reference_index=-1,
            iou=1.0, notes=f"anchor padding={padding}",
        )

    if results[mid] is None:
        results[mid] = AlignmentResult(
            slide_path=slides[mid], index=mid, reference_index=-1,
            iou=1.0, notes="anchor (resumed)",
        )

    def _align_direction(indices: List[int]):
        ref_idx = mid
        ref_img = anchor_img.copy()
        aligned_cache: dict = {mid: anchor_img.copy()}

        for idx in indices:
            if _is_done(idx) and not ("skipped" in (results[idx].notes if results[idx] else "")):
                cached = _load_aligned_image(idx)
                if cached is not None:
                    aligned_cache[idx] = cached

        for i_pos, idx in enumerate(indices):
            if _is_done(idx):
                cached = aligned_cache.get(idx)
                if cached is not None:
                    ref_img = cached
                    ref_idx = idx
                if verbose:
                    print(f"[{idx}] {Path(slides[idx]).name} — already done, skipping.")
                continue

            for prev_idx in reversed(indices[:i_pos]):
                if prev_idx in aligned_cache:
                    ref_idx = prev_idx
                    ref_img = aligned_cache[prev_idx]
                    break

            if verbose:
                print(f"\nAligning [{idx}] {Path(slides[idx]).name} "
                      f"-> reference [{ref_idx}]")

            # Pad moving image to same canvas as anchor
            moving_img = _pad_image(_read_level(slides[idx], level))
            slide_stem = Path(slides[idx]).stem

            aligned, aligned_mask, iou, M, displacement = register_pair(
                fixed=ref_img,
                moving=moving_img,
                g_thresh=g_thresh,
                affine_method=affine_method,
                do_elastic=do_elastic,
                tile_size=tile_size,
                buffer=buffer,
                grid_spacing=grid_spacing,
                tissue_cutoff=tissue_cutoff,
                iou_threshold=0.0,
                elastic_passes=elastic_passes,
                rbf_smoothing=rbf_smoothing,
                keep_if_improves=keep_if_improves,
                verbose=verbose,
            )

            if iou >= iou_threshold:
                if verbose:
                    print(f"  OK accepted (IoU={iou:.4f})")
                results[idx] = AlignmentResult(
                    slide_path=slides[idx], index=idx,
                    reference_index=ref_idx, iou=iou, affine_matrix=M,
                )
                if save_aligned:
                    _save_image(aligned, aligned_dir / f"{slide_stem}.png")
                if save_masks:
                    _save_image(aligned_mask.astype(np.uint8) * 255,
                                masks_dir / f"{slide_stem}.png")
                _save_transform(M, displacement, affine_dir, elastic_dir, slide_stem,
                                slide_index=idx, reference_index=ref_idx, level=level)
                aligned_cache[idx] = aligned
                ref_img = aligned
                ref_idx = idx
            else:
                if verbose:
                    print(f"  FAIL below threshold {iou_threshold} (IoU={iou:.4f}), probing alternate references for this slide...")

                # Probe other references (affine only) to find the one that best
                # aligns THIS slide. Each candidate keeps its fixed image so the
                # winner can be re-registered at full quality and saved.
                behind_candidates = [
                    indices[j] for j in range(i_pos - 1, max(i_pos - max_skip - 1, -1), -1)
                    if indices[j] in aligned_cache
                ]
                ahead_candidates = _get_skip_candidates(idx, indices, max_skip)

                # Seed with the result we already computed against the current reference.
                candidate_results = [(iou, ref_idx, ref_img)]

                for cand_idx in behind_candidates:
                    if cand_idx == ref_idx:
                        continue
                    if verbose:
                        print(f"  Probing behind [{cand_idx}] (affine only) ...", end=" ", flush=True)
                    _, _, cand_iou, _, _ = register_pair(
                        fixed=aligned_cache[cand_idx], moving=moving_img,
                        g_thresh=g_thresh, affine_method=affine_method,
                        do_elastic=False, tile_size=tile_size, buffer=buffer,
                        grid_spacing=grid_spacing, tissue_cutoff=tissue_cutoff,
                        iou_threshold=0.0, verbose=False,
                    )
                    if verbose:
                        print(f"IoU={cand_iou:.4f}")
                    candidate_results.append((cand_iou, cand_idx, aligned_cache[cand_idx]))

                for skip_idx in ahead_candidates:
                    # An ahead slide can only be a reference if it has already been
                    # aligned; otherwise it is not yet in a usable coordinate frame.
                    if skip_idx not in aligned_cache:
                        continue
                    if verbose:
                        print(f"  Probing ahead [{skip_idx}] (affine only) ...", end=" ", flush=True)
                    _, _, skip_iou, _, _ = register_pair(
                        fixed=aligned_cache[skip_idx], moving=moving_img,
                        g_thresh=g_thresh, affine_method=affine_method,
                        do_elastic=False, tile_size=tile_size, buffer=buffer,
                        grid_spacing=grid_spacing, tissue_cutoff=tissue_cutoff,
                        iou_threshold=0.0, verbose=False,
                    )
                    if verbose:
                        print(f"IoU={skip_iou:.4f}")
                    candidate_results.append((skip_iou, skip_idx, aligned_cache[skip_idx]))

                # Pick the reference that best aligns THIS slide.
                candidate_results.sort(key=lambda x: x[0], reverse=True)
                best_iou, best_ref_idx, best_ref_img = candidate_results[0]

                if best_ref_idx == ref_idx:
                    # Current reference was already best — reuse what we computed.
                    final_aligned, final_mask, final_iou = aligned, aligned_mask, iou
                    final_M, final_disp = M, displacement
                else:
                    if verbose:
                        print(f"  Better reference [{best_ref_idx}] (affine IoU={best_iou:.4f}); "
                              f"re-registering at full quality...")
                    final_aligned, final_mask, final_iou, final_M, final_disp = register_pair(
                        fixed=best_ref_img, moving=moving_img,
                        g_thresh=g_thresh, affine_method=affine_method,
                        do_elastic=do_elastic, tile_size=tile_size, buffer=buffer,
                        grid_spacing=grid_spacing, tissue_cutoff=tissue_cutoff,
                        iou_threshold=0.0, elastic_passes=elastic_passes,
                        rbf_smoothing=rbf_smoothing, keep_if_improves=keep_if_improves,
                        verbose=verbose,
                    )

                note = "" if final_iou >= iou_threshold else f"low IoU ({final_iou:.4f}), best available"
                if verbose:
                    accepted = "OK" if final_iou >= iou_threshold else "WARN best available"
                    print(f"  {accepted} accepted (IoU={final_iou:.4f}, ref=[{best_ref_idx}])")

                # Always save idx — every slide gets an output image.
                results[idx] = AlignmentResult(
                    slide_path=slides[idx], index=idx,
                    reference_index=best_ref_idx, iou=final_iou, affine_matrix=final_M,
                    notes=note,
                )
                if save_aligned:
                    _save_image(final_aligned, aligned_dir / f"{slide_stem}.png")
                if save_masks:
                    _save_image(final_mask.astype(np.uint8) * 255,
                                masks_dir / f"{slide_stem}.png")
                _save_transform(final_M, final_disp, affine_dir, elastic_dir, slide_stem,
                                slide_index=idx, reference_index=best_ref_idx, level=level,
                                notes=note)
                aligned_cache[idx] = final_aligned
                ref_img = final_aligned
                ref_idx = idx

    forward_indices = list(range(mid + 1, n))
    if verbose and forward_indices:
        print(f"\n{'='*60}")
        print(f"FORWARD PASS: indices {forward_indices[0]}..{forward_indices[-1]}")
        print(f"{'='*60}")
    _align_direction(forward_indices)

    backward_indices = list(range(mid - 1, -1, -1))
    if verbose and backward_indices:
        print(f"\n{'='*60}")
        print(f"BACKWARD PASS: indices {backward_indices[0]}..{backward_indices[-1]}")
        print(f"{'='*60}")
    _align_direction(backward_indices)

    for i in range(n):
        if results[i] is None:
            results[i] = AlignmentResult(
                slide_path=slides[i], index=i, reference_index=-1,
                iou=0.0, skipped=True, notes="not processed",
            )

    _save_metadata(results, out_dir / "alignment_results.json")
    if verbose:
        print(f"\nDone.")
        print(f"  Aligned images : {aligned_dir}")
        print(f"  Tissue masks   : {masks_dir}")
        print(f"  Transforms     : {transforms_dir}")
        print(f"  Metadata       : {out_dir / 'alignment_results.json'}")
        _print_summary(results)

    return results


# Skip helpers

def _get_skip_candidates(
    current_idx: int,
    ordered_indices: List[int],
    max_skip: int,
) -> List[int]:
    """Return the next max_skip indices after current_idx in ordered_indices."""
    try:
        pos = ordered_indices.index(current_idx)
    except ValueError:
        return []
    candidates = ordered_indices[pos + 1: pos + 1 + max_skip]
    return candidates


# other helpers

def _save_image(img: np.ndarray, path: Path):
    """Save image as TIFF or PNG."""
    path = Path(path)
    if path.suffix.lower() == ".png":
        Image.fromarray(img).save(str(path))
    else:
        tifffile.imwrite(str(path), img, compression="zlib")


def _save_metadata(results: List[AlignmentResult], path: Path):
    """Save alignment results as JSON."""
    records = []
    for r in results:
        d = {
            "slide_path": r.slide_path,
            "index": r.index,
            "reference_index": r.reference_index,
            "iou": r.iou,
            "skipped": r.skipped,
            "notes": r.notes,
        }
        if r.affine_matrix is not None:
            d["affine_matrix"] = r.affine_matrix.tolist()
        records.append(d)

    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def _print_summary(results: List[AlignmentResult]):
    """Print a compact summary table."""
    print(f"\n{'Idx':<5} {'IoU':<8} {'Ref':<5} {'Skip':<6} {'Notes'}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.index):
        skip_str = "YES" if r.skipped else ""
        print(f"{r.index:<5} {r.iou:<8.4f} {r.reference_index:<5} "
              f"{skip_str:<6} {r.notes}")


# Transform

def _save_transform(
    affine_matrix: Optional[np.ndarray],
    displacement: Optional[np.ndarray],
    affine_dir: Path,
    elastic_dir: Path,
    slide_stem: str,
    slide_index: int = -1,
    reference_index: int = -1,
    level: int = -1,
    notes: str = "",
):
    """Save affine matrix as {slide_stem}.pkl in affine_dir;
    save displacement field as {slide_stem}.npy in elastic_dir.
    """
    affine_dir  = Path(affine_dir)
    elastic_dir = Path(elastic_dir)

    record = {
        "slide_index": slide_index,
        "reference_index": reference_index,
        "level": level,
        "affine_matrix": affine_matrix,
        "displacement_file": None,
        "notes": notes,
    }

    if displacement is not None:
        npy_path = elastic_dir / f"{slide_stem}.npy"
        try:
            np.save(str(npy_path), displacement)
            record["displacement_file"] = str(npy_path)
        except Exception as e:
            record["notes"] += f" | displacement save failed: {e}"

    pkl_path = affine_dir / f"{slide_stem}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_transform(affine_dir: str, slide_stem: str) -> dict:
    """Load a saved transform for a given slide stem.

    Looks for:
        {affine_dir}/{slide_stem}.pkl
        {affine_dir}/../elastic/{slide_stem}.npy  (if displacement_file not set)

    Returns dict with keys:
        slide_index, reference_index, level, affine_matrix,
        displacement_file, elastic_transform, notes.
    """
    affine_dir = Path(affine_dir)
    pkl_path   = affine_dir / f"{slide_stem}.pkl"

    with open(pkl_path, "rb") as f:
        record = pickle.load(f)

    npy_file = record.get("displacement_file")
    # Fallback: check elastic/ sibling folder
    if not npy_file:
        elastic_dir  = affine_dir.parent / "elastic"
        npy_fallback = elastic_dir / f"{slide_stem}.npy"
        if npy_fallback.exists():
            npy_file = str(npy_fallback)

    if npy_file and Path(npy_file).exists():
        try:
            record["elastic_transform"] = np.load(npy_file)
        except Exception:
            record["elastic_transform"] = None
    else:
        record["elastic_transform"] = None

    return record



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Align serial H&E sections from .ndpi files."
    )
    parser.add_argument("slide_folder", help="Folder containing .ndpi files.")
    parser.add_argument("output_folder", help="Where to save aligned images.")
    parser.add_argument("--level", type=int, default=3,
                        help="Pyramid level for registration (default: 3).")
    parser.add_argument("--iou-threshold", type=float, default=0.9,
                        help="Min IoU to accept alignment (default: 0.9).")
    parser.add_argument("--max-skip", type=int, default=3,
                        help="Max slides to skip for damaged sections (default: 3).")
    parser.add_argument("--no-elastic", action="store_true",
                        help="Skip elastic registration (affine only).")
    parser.add_argument("--affine-method", default="combined",
                        choices=["ECC", "ORB", "combined"],
                        help="Affine estimation method (default: combined).")
    parser.add_argument("--g-thresh", type=int, default=170,
                        help="Green-channel threshold for tissue mask (default: 170).")
    parser.add_argument("--extension", default=".ndpi",
                        help="Slide file extension (default: .ndpi).")

    args = parser.parse_args()

    align_serial_sections(
        slide_folder=args.slide_folder,
        output_folder=args.output_folder,
        level=args.level,
        iou_threshold=args.iou_threshold,
        max_skip=args.max_skip,
        g_thresh=args.g_thresh,
        affine_method=args.affine_method,
        do_elastic=not args.no_elastic,
        extension=args.extension,
    )
