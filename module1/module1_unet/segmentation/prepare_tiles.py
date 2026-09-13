"""
prepare_tiles.py — One-time preprocessing: reads each scene's full GeoTIFF +
mask ONCE, tiles them in-memory, and writes each tile pair to local disk as
a torch .pt file, along with a lightweight manifest describing them.

Why this exists:
    The training Dataset was opening every scene's GeoTIFF twice per tile
    (once for the image, once for the mask) via rasterio, 64 tiles/scene.
    Even with windowed reads, if the source TIFFs aren't internally tiled
    (BLOCKXSIZE/BLOCKYSIZE/TILED=YES), GDAL still has to decode full strips
    per "windowed" read — so the real I/O cost per epoch could be close to
    decoding the whole 200-scene dataset dozens of times over, on top of
    per-open overhead (worse again if files sit on a Drive/FUSE mount).

    This script pays that decode cost exactly once per scene, then training
    just does torch.load() off local disk per tile — no GDAL in the hot loop
    at all.

This does NOT change the model, loss, or the train/val split *semantics* —
run it separately on a train-scene manifest and a val-scene manifest (see
the README note at the bottom) so the scene-level split you already computed
via train_val_split() is preserved and tiles from the same scene never end
up split across train and val.

Usage:
    python prepare_tiles.py \
        --manifest data/train_scenes_manifest.json \
        --out-dir data/pretiled_train \
        --patch-size 256 \
        --bands VV VH \
        --max-scenes 28          # omit to use ALL scenes in the manifest

    python prepare_tiles.py \
        --manifest data/val_scenes_manifest.json \
        --out-dir data/pretiled_val \
        --patch-size 256 \
        --bands VV VH \
        --max-scenes 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
# These are reused as-is from the existing dataset module, so tiling/NaN
# rejection logic stays identical to what training already validates against.
from src.dataset import (
    DatasetValidationError,
    _load_geotiff,
    _load_mask,
    _tile_indices,
    load_manifest_entries,
)


def prepare_tiles(
    manifest_path: str,
    out_dir: str,
    patch_size: int,
    expected_bands: list[str],
    max_scenes: int | None,
    seed: int,
) -> None:
    entries = load_manifest_entries(manifest_path)

    if max_scenes is not None and max_scenes < len(entries):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(entries), size=max_scenes, replace=False)
        entries = [entries[i] for i in sorted(idx.tolist())]
        print(f"Subsetting {manifest_path} to {max_scenes} scenes (seed={seed}).")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tiles_dir = out_path / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    new_manifest = []
    total_tiles = 0
    total_oil_tiles = 0
    skipped_nan_tiles = 0

    for i, entry in enumerate(entries):
        scene_id = entry["scene_id"]
        if entry["bands"] != expected_bands:
            raise DatasetValidationError(
                f"scene_id={scene_id} has bands {entry['bands']}, expected {expected_bands}"
            )

        # Read the FULL scene exactly once — this is the expensive GDAL
        # decode, and after this script it never happens again.
        image, _meta = _load_geotiff(entry["patch_path"])
        mask = None
        if "mask_path" in entry:
            mask = _load_mask(entry["mask_path"])

        if image.shape[0] != len(expected_bands):
            raise DatasetValidationError(
                f"scene_id={scene_id}: {image.shape[0]} bands, expected {len(expected_bands)}"
            )

        h, w = image.shape[1], image.shape[2]
        if h < patch_size or w < patch_size:
            print(f"Skipping {scene_id}: {h}x{w} smaller than patch_size={patch_size}")
            continue

        for y, x in _tile_indices(h, w, patch_size, patch_size):
            img_tile = image[:, y : y + patch_size, x : x + patch_size]
            if not np.isfinite(img_tile).all():
                # Same NaN/Inf rejection the original loader applied per-tile,
                # just done once here instead of on every epoch.
                skipped_nan_tiles += 1
                continue

            tile_id = f"{scene_id}_{y}_{x}"
            record = {
                "scene_id": scene_id,
                "tile_id": tile_id,
                "image_path": str(tiles_dir / f"{tile_id}_img.pt"),
            }
            torch.save(torch.from_numpy(img_tile.copy()).float(), record["image_path"])

            if mask is not None:
                mask_tile = mask[y : y + patch_size, x : x + patch_size]
                record["mask_path"] = str(tiles_dir / f"{tile_id}_mask.pt")
                torch.save(
                    torch.from_numpy(mask_tile.copy()).float().unsqueeze(0),
                    record["mask_path"],
                )
                if mask_tile.sum() > 0:
                    total_oil_tiles += 1

            new_manifest.append(record)
            total_tiles += 1

        if (i + 1) % 5 == 0 or i == len(entries) - 1:
            print(f"Processed {i + 1}/{len(entries)} scenes, {total_tiles} tiles so far.")

    manifest_out = out_path / "manifest.json"
    with manifest_out.open("w") as f:
        json.dump({"entries": new_manifest}, f)

    print(f"\nDone. {total_tiles} tiles written to {tiles_dir}")
    if skipped_nan_tiles:
        print(f"Skipped {skipped_nan_tiles} tiles for NaN/Inf pixels (spec §11 rule preserved).")
    if total_tiles:
        print(f"Positive (oil) tile fraction: {total_oil_tiles / total_tiles:.3f}")
    print(f"New manifest: {manifest_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="Path to a scene-level manifest (train OR val split, not the full 200-scene one)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--bands", nargs="+", default=["VV", "VH"])
    p.add_argument("--max-scenes", type=int, default=None, help="Cap scene count for a fast MVP subset")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare_tiles(
        args.manifest, args.out_dir, args.patch_size, args.bands, args.max_scenes, args.seed
    )
