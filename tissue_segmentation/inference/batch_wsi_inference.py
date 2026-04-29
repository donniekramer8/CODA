import sys
sys.path.append("../src")

import argparse
import os
from pathlib import Path

import torch

import config
from wsi_inference import process_single_wsi

VALID_EXTENSIONS = {".ndpi"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch WSI inference — runs on every .ndpi in a folder."
    )
    parser.add_argument("--ndpi_dir",   required=True,
                        help="Folder containing .ndpi files.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pth checkpoint.")
    parser.add_argument("--out_dir",    required=True,
                        help="Output directory for label maps.")
    parser.add_argument("--level",      type=int, default=1,
                        help="OpenSlide pyramid level (default: 1).")
    parser.add_argument("--tile_size",  type=int, default=config.TILE_SIZE)
    parser.add_argument("--overlap",    type=int, default=64,
                        help="Tile overlap in pixels (default: 64).")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--g_thresh",   type=int, default=190,
                        help="Green-channel tissue threshold (default: 190).")
    parser.add_argument("--save_scale", type=int, default=4,
                        help="Downscale factor for saved label maps (default: 4).")
    parser.add_argument("--no_colormap", action="store_true",
                        help="Skip saving colourmap PNGs (save label maps only).")
    return parser.parse_args()


def collect_ndpi_files(folder: str) -> list[str]:
    """Recursively find all files with a valid extension in folder."""
    found = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if Path(f).suffix.lower() in VALID_EXTENSIONS:
                found.append(os.path.join(root, f))
    return found


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ndpi_files = collect_ndpi_files(args.ndpi_dir)
    if not ndpi_files:
        print(f"No .ndpi files found in: {args.ndpi_dir}")
        return

    print(f"Found {len(ndpi_files)} .ndpi file(s) in '{args.ndpi_dir}'")
    os.makedirs(args.out_dir, exist_ok=True)

    failed = []
    for i, ndpi_path in enumerate(ndpi_files, 1):
        print(f"\n[{i}/{len(ndpi_files)}] {ndpi_path}")

        stem = Path(ndpi_path).stem
        expected_output = Path(args.out_dir) / f"{stem}.png"
        if expected_output.exists():
            print(f"  Skipping — output already exists: {expected_output}")
            continue

        try:
            process_single_wsi(
                ndpi_path=ndpi_path,
                checkpoint_path=args.checkpoint,
                out_dir=args.out_dir,
                device=device,
                level=args.level,
                tile_size=args.tile_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
                g_thresh=args.g_thresh,
                save_scale=args.save_scale,
                save_colormap=not args.no_colormap,
            )
        except Exception as e:
            print(f"  ERROR processing {ndpi_path}: {e}")
            failed.append(ndpi_path)

    print(f"\n{'='*60}")
    print(f"Batch complete. {len(ndpi_files) - len(failed)}/{len(ndpi_files)} succeeded.")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()

# python /home/donald/Desktop/shelter_server_donald/code/CODA_my_version/tissue_segmentation/inference/batch_wsi_inference.py \
#     --ndpi        /home/donald/Desktop/shelter_server_donald/data/PDAC_Dimitri/S21-27914_4N \
#     --checkpoint  /home/donald/Desktop/shelter_server_donald/code/CODA_my_version/tissue_segmentation/src/checkpoints/best_model.pth \
#     --out         /home/donald/Desktop/shelter_server_donald/data/PDAC_Dimitri/S21-27914_4N/segmentation_3_2_26 \
#     --level       1 \
#     --save_scale  8 \
#     --no_colormap
