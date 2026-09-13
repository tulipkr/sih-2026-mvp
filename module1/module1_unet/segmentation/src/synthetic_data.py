"""
Synthetic data generator (spec §34): "Build a synthetic mock-patch generator
... so Tulip's training/inference code can be developed and tested
independently."

This mimics the manifest/GeoTIFF FORMAT exactly, not the physics — synthetic
"SAR" values are structured noise with an injected low-value blob standing
in for oil. This must only be used for pipeline verification (per this
session's explicit instruction) — it never substitutes for real training data
and never feeds into the real-data code path unless you point manifest_path
at it yourself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_geotiff(path: Path, array: np.ndarray, transform, crs: str, dtype: str):
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    count = array.shape[0] if array.ndim == 3 else 1
    height, width = array.shape[-2], array.shape[-1]

    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype=dtype, crs=crs, transform=transform,
    ) as dst:
        if array.ndim == 3:
            dst.write(array)
        else:
            dst.write(array, 1)


def generate_synthetic_dataset(
    output_dir: str | Path,
    n_samples: int = 8,
    patch_size: int = 256,
    seed: int = 42,
    no_oil_fraction: float = 0.25,
) -> Path:
    """
    Creates output_dir/patches/<id>.tif, output_dir/masks/<id>.tif, and
    output_dir/manifest.json with n_samples entries, matching spec §7's
    manifest schema exactly. Returns the manifest path.
    """
    from rasterio.transform import Affine

    rng = np.random.RandomState(seed)
    output_dir = Path(output_dir)
    patches_dir = output_dir / "patches"
    masks_dir = output_dir / "masks"
    patches_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    base_lon, base_lat = 80.30, 13.20  # arbitrary synthetic origin, not a real location
    pixel_deg = 0.0001
    n_no_oil = max(1, int(n_samples * no_oil_fraction)) if n_samples > 3 else 0

    for i in range(n_samples):
        scene_id = f"synthetic_scene_{i:03d}"
        is_no_oil = i < n_no_oil

        vv = rng.normal(loc=-15.0, scale=3.0, size=(patch_size, patch_size)).astype(np.float32)
        vh = rng.normal(loc=-20.0, scale=3.0, size=(patch_size, patch_size)).astype(np.float32)
        mask = np.zeros((patch_size, patch_size), dtype=np.uint8)

        if not is_no_oil:
            cy, cx = rng.randint(patch_size // 4, 3 * patch_size // 4, size=2)
            max_radius = max(2, patch_size // 4)
            min_radius = min(15, max_radius - 1)
            radius = rng.randint(min_radius, max_radius)            
            yy, xx = np.ogrid[:patch_size, :patch_size]
            blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            vv[blob] -= 10.0  # push well below a typical fallback threshold
            vh[blob] -= 6.0
            mask[blob] = 1

        image = np.stack([vv, vh], axis=0)  # band order: VV, VH — matches config.data.bands

        lon_offset = i * patch_size * pixel_deg
        transform = Affine(pixel_deg, 0.0, base_lon + lon_offset, 0.0, -pixel_deg, base_lat)
        crs = "EPSG:4326"

        patch_path = patches_dir / f"{scene_id}.tif"
        mask_path = masks_dir / f"{scene_id}.tif"
        _write_geotiff(patch_path, image, transform, crs, dtype="float32")
        _write_geotiff(mask_path, mask, transform, crs, dtype="uint8")

        manifest_entries.append({
            "scene_id": scene_id,
            "patch_path": str(patch_path),
            "mask_path": str(mask_path),
            "acquisition_timestamp_utc": f"2026-01-{(i % 28) + 1:02d}T03:00:00Z",
            "crs": crs,
            "transform": list(transform)[:6],
            "bands": ["VV", "VH"],
        })

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump({"entries": manifest_entries}, f, indent=2)

    return manifest_path


def generate_lookalike_set(
    output_dir: str | Path, n_samples: int = 4, patch_size: int = 256, seed: int = 99
) -> Path:
    """
    Dedicated no-oil ("look-alike") set for the false-positive check (spec
    §3/§29) — every sample here has zero oil pixels by construction.
    """
    return generate_synthetic_dataset(
        output_dir, n_samples=n_samples, patch_size=patch_size, seed=seed, no_oil_fraction=1.0
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic pipeline-verification data.")
    parser.add_argument("--output_dir", default="data/synthetic")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookalike_dir", default="data/test_lookalike")
    parser.add_argument("--lookalike_samples", type=int, default=4)
    args = parser.parse_args()

    manifest_path = generate_synthetic_dataset(
        args.output_dir, args.n_samples, args.patch_size, args.seed
    )
    print(f"Synthetic manifest written to {manifest_path}")

    lookalike_path = generate_lookalike_set(
        args.lookalike_dir, args.lookalike_samples, args.patch_size, seed=args.seed + 1
    )
    print(f"Look-alike (no-oil) manifest written to {lookalike_path}")
