"""
Module 2 Stage A: raw Sentinel-1 -> calibrated, speckle-filtered,
land-masked, (optionally) normalized 256x256 VV/VH patches + manifest.json
matching Module 1's schema exactly (spec §7/§9).

Two input modes, auto-detected by what's passed:
  1. Real SAFE product (--safe_path): full pipeline — locates VV/VH,
     radiometrically calibrates from the product's own LUT, extracts the
     real acquisition timestamp, resolves georeferencing (direct or via
     GCPs).
  2. Direct VV/VH GeoTIFFs (--vv_path/--vh_path): legacy/test path for
     already-calibrated input (spec §35's documented pre-calibrated-Sigma0
     shortcut) — kept for backward compatibility with existing tests and for
     providers that ship pre-calibrated Sigma0 GeoTIFFs directly.

Never hardcodes a scene-specific path, coordinate, or timestamp anywhere in
this file — every scene-specific value comes from the SAFE product itself
(mode 1) or from caller-supplied arguments (mode 2).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.calibration import CalibrationError, calibrate_band
from src.georeferencing import GeoreferencingError, resolve_geotransform
from src.land_mask import LandMaskError, apply_land_mask
from src.safe_reader import SafeProductError, locate_safe_product

logger = logging.getLogger(__name__)


class StageAError(ValueError):
    pass


def lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """5x5 window Lee speckle filter."""
    mean = uniform_filter(img, size)
    mean_sq = uniform_filter(img ** 2, size)
    var = np.maximum(0, mean_sq - mean ** 2)
    overall_var = np.var(img)
    weight = var / (var + overall_var + 1e-8)
    return mean + weight * (img - mean)


def _is_already_db(sample_array: np.ndarray) -> bool:
    """Legacy-mode heuristic only (mode 2, pre-calibrated input of unknown
    linear/dB state). In SAFE mode (mode 1) this is never used — calibration
    always starts from real DN, so the linear-vs-dB question doesn't arise."""
    valid = sample_array[np.isfinite(sample_array)]
    if len(valid) == 0:
        return False
    return bool(np.percentile(valid, 5) < 0.0)


def _read_dn_with_georeferencing(path: Path):
    with rasterio.open(path) as src:
        transform, crs, is_gcp_approx = resolve_geotransform(src)
        dn = src.read(1).astype(np.float64)
        height, width = src.height, src.width
    return dn, transform, crs, is_gcp_approx, height, width


def _prepare_from_safe(safe_path: Path, work_dir: Path):
    """Mode 1: locate + calibrate a real SAFE product. Returns
    (vv_sigma0, vh_sigma0, transform, crs, scene_id, acquisition_timestamp_utc)."""
    product = locate_safe_product(safe_path, work_dir)
    logger.info(f"Located SAFE product {product.product_id}: "
                f"VV={product.vv_measurement_path.name}, VH={product.vh_measurement_path.name}")

    vv_dn, vv_transform, vv_crs, vv_gcp, height, width = _read_dn_with_georeferencing(
        product.vv_measurement_path
    )
    vh_dn, vh_transform, vh_crs, vh_gcp, vh_height, vh_width = _read_dn_with_georeferencing(
        product.vh_measurement_path
    )
    if (vh_height, vh_width) != (height, width):
        raise StageAError(
            f"VV measurement is {height}x{width} but VH measurement is "
            f"{vh_height}x{vh_width} — cannot pair mismatched-dimension bands "
            f"from the same product (spec §16)."
        )

    logger.info("Applying real radiometric calibration from the product's own LUT "
                "(annotation/calibration/*.xml) — no placeholder constant.")
    vv_sigma0 = calibrate_band(vv_dn, product.vv_calibration_path)
    vh_sigma0 = calibrate_band(vh_dn, product.vh_calibration_path)

    return (
        vv_sigma0, vh_sigma0, vv_transform, vv_crs,
        product.product_id, product.acquisition_start_utc,
    )


def _prepare_from_direct_files(vv_path: Path, vh_path: Path):
    """Mode 2: legacy/test path — already-calibrated (or already-dB) VV/VH
    GeoTIFFs, auto-detecting linear-vs-dB via a heuristic (documented
    limitation: only reliable for genuinely pre-calibrated input, per
    spec §35's shortcut — never use this mode on raw, uncalibrated DN)."""
    with rasterio.open(vv_path) as src_vv:
        transform, crs, _ = resolve_geotransform(src_vv)
        height, width = src_vv.height, src_vv.width
        sample_win = Window(0, 0, min(512, width), min(512, height))
        sample_vv = src_vv.read(1, window=sample_win).astype(np.float32)
        input_is_db = _is_already_db(sample_vv)
        vv = src_vv.read(1).astype(np.float64)
    with rasterio.open(vh_path) as src_vh:
        if (src_vh.height, src_vh.width) != (height, width):
            raise StageAError(
                f"VV is {height}x{width} but VH is {src_vh.height}x{src_vh.width} — "
                f"cannot pair mismatched-dimension bands (spec §16)."
            )
        vh = src_vh.read(1).astype(np.float64)

    if input_is_db:
        logger.info(">> Detected input imagery is ALREADY in dB (mode 2 legacy heuristic). "
                    "Treating as pre-calibrated per spec §35's documented shortcut.")
        vv_db, vh_db = vv, vh
    else:
        logger.info(">> Detected LINEAR values (mode 2 legacy heuristic) — assuming "
                    "pre-calibrated Sigma0, converting to dB. If this is actually raw, "
                    "uncalibrated DN, these values will be physically wrong — use "
                    "--safe_path with a real SAFE product instead of this legacy path "
                    "for genuine raw Sentinel-1 data.")
        eps = 1e-6
        vv_db = 10.0 * np.log10(np.maximum(vv, eps))
        vh_db = 10.0 * np.log10(np.maximum(vh, eps))

    return vv_db, vh_db, transform, crs


def run_stage_a(
    output_dir,
    safe_path=None,
    vv_path=None,
    vh_path=None,
    scene_id=None,
    timestamp_utc=None,
    patch_size=256,
    config=None,
):
    """
    Args:
        output_dir: where preprocessed_patches/ + manifest.json go.
        safe_path: path to a .SAFE dir or .zip (mode 1 — real product).
        vv_path/vh_path: direct GeoTIFF paths (mode 2 — legacy/pre-calibrated).
        scene_id/timestamp_utc: REQUIRED overrides for mode 2 only (mode 1
            derives both from the real product and ignores these if passed).
        config: the stage_a config block (see configs/config.yaml). Uses
            spec-default values if not supplied.
    """
    config = config or {}
    speckle_cfg = config.get("speckle_filter", {"apply": True, "window": 5})
    norm_cfg = config.get("normalization", {})
    land_cfg = config.get("land_mask", {"source_path": None})
    scene_is_coastal = config.get("scene_is_coastal", True)

    out_root = Path(output_dir)
    patches_dir = out_root / "preprocessed_patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    if safe_path:
        vv_sigma0, vh_sigma0, transform, crs, real_scene_id, real_timestamp = _prepare_from_safe(
            Path(safe_path), out_root / "_safe_extract"
        )
        scene_id = real_scene_id
        timestamp_utc = real_timestamp
        eps = 1e-6
        vv_db = 10.0 * np.log10(np.maximum(vv_sigma0, eps))
        vh_db = 10.0 * np.log10(np.maximum(vh_sigma0, eps))
    elif vv_path and vh_path:
        if not os.path.exists(vv_path) or not os.path.exists(vh_path):
            raise FileNotFoundError(f"Missing input files: {vv_path} or {vh_path}")
        if not scene_id or not timestamp_utc:
            raise StageAError(
                "Mode 2 (direct vv_path/vh_path) requires explicit scene_id and "
                "timestamp_utc — this mode has no product metadata to derive them "
                "from, and this pipeline never fabricates either (spec §20/§39)."
            )
        vv_db, vh_db, transform, crs = _prepare_from_direct_files(Path(vv_path), Path(vh_path))
    else:
        raise StageAError("Must supply either safe_path, or both vv_path and vh_path.")

    if crs is None or transform is None or transform.is_identity:
        raise StageAError(
            "CRS or affine transform missing/unresolvable after calibration — "
            "cannot proceed to tiling without georeferencing (spec §16)."
        )

    height, width = vv_db.shape

    if speckle_cfg.get("apply", True):
        window = speckle_cfg.get("window", 5)
        logger.info(f"Applying Lee speckle filter ({window}x{window}) before land masking.")
        vv_db = lee_filter(vv_db, size=window)
        vh_db = lee_filter(vh_db, size=window)

    vv_db, vh_db, land_fraction = apply_land_mask(
        vv_db, vh_db, transform, crs, land_cfg, scene_is_coastal=scene_is_coastal
    )
    logger.info(f"Land mask coverage: {land_fraction:.2%} of scene.")

    apply_db_clipping = norm_cfg.get("apply_db_clipping", False)
    apply_normalization = norm_cfg.get("apply_normalization", False)
    db_clip_min = norm_cfg.get("db_clip_min", -30)
    db_clip_max = norm_cfg.get("db_clip_max", 0)
    if apply_db_clipping:
        vv_db = np.clip(vv_db, db_clip_min, db_clip_max)
        vh_db = np.clip(vh_db, db_clip_min, db_clip_max)
    if apply_normalization:
        denom = db_clip_max - db_clip_min
        vv_db = (vv_db - db_clip_min) / denom
        vh_db = (vh_db - db_clip_min) / denom

    profile = {
        "driver": "GTiff", "count": 2, "dtype": rasterio.float32,
        "width": patch_size, "height": patch_size, "crs": crs, "nodata": None,
    }

    tile_idx = 0
    skipped_partial_nan = 0
    for row in range(0, height - patch_size + 1, patch_size):
        for col in range(0, width - patch_size + 1, patch_size):
            vv_tile = vv_db[row:row + patch_size, col:col + patch_size].astype(np.float32)
            vh_tile = vh_db[row:row + patch_size, col:col + patch_size].astype(np.float32)

            nan_frac = max(np.isnan(vv_tile).mean(), np.isnan(vh_tile).mean())
            if nan_frac == 1.0:
                continue  # entirely land/nodata tile — not an error, just skip
            if nan_frac > 0.0:
                # Partial NaN (some land pixels, some ocean; or a genuine
                # calibration edge artifact) — spec §16 requires rejecting
                # affected tiles explicitly and logging, not silently
                # shipping partially-NaN patches downstream.
                logger.warning(
                    f"scene_id={scene_id} tile row={row} col={col}: "
                    f"{nan_frac:.1%} NaN pixels — rejecting this tile rather "
                    f"than passing partial NaN to Module 1 (spec §16)."
                )
                skipped_partial_nan += 1
                continue

            tile_win = Window(col, row, patch_size, patch_size)
            tile_transform = rasterio.windows.transform(tile_win, transform)

            patch_id = f"{scene_id}_{tile_idx:04d}"
            patch_file = patches_dir / f"{patch_id}.tif"
            tile_profile = dict(profile, transform=tile_transform)
            with rasterio.open(patch_file, "w", **tile_profile) as dst:
                dst.write(vv_tile, 1)
                dst.write(vh_tile, 2)

            manifest.append({
                "scene_id": scene_id,
                "patch_id": patch_id,
                "patch_path": str(patch_file).replace("\\", "/"),
                "acquisition_timestamp_utc": timestamp_utc,
                "crs": str(crs) if not isinstance(crs, str) else crs,
                "transform": list(tile_transform)[:6],
                "bands": ["VV", "VH"],
            })
            tile_idx += 1

    if tile_idx == 0:
        raise StageAError(
            f"scene_id={scene_id}: produced zero usable tiles (all were either "
            f"fully outside the scene, fully land-masked, or rejected for partial "
            f"NaN — {skipped_partial_nan} tiles skipped for partial NaN). Nothing "
            f"to hand to Module 1."
        )

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Stage A complete for scene_id={scene_id}: {tile_idx} patches written "
                f"({skipped_partial_nan} skipped for partial NaN). Manifest: {manifest_path}")
    print(f"Stage A complete! Generated {tile_idx} patches in {patches_dir}")
    print(f"Manifest written to {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Module 2 Stage A: SAR preprocessing.")
    parser.add_argument("--safe_path", default=None, help="Path to a .SAFE dir or .zip (real product).")
    parser.add_argument("--vv_path", default=None, help="Direct VV GeoTIFF (legacy/pre-calibrated mode).")
    parser.add_argument("--vh_path", default=None, help="Direct VH GeoTIFF (legacy/pre-calibrated mode).")
    parser.add_argument("--scene_id", default=None, help="Required for --vv_path/--vh_path mode.")
    parser.add_argument("--timestamp_utc", default=None, help="Required for --vv_path/--vh_path mode.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults used if omitted).")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            full_cfg = yaml.safe_load(f)
        cfg = full_cfg.get("stage_a", {})

    run_stage_a(
        output_dir=args.output_dir,
        safe_path=args.safe_path,
        vv_path=args.vv_path,
        vh_path=args.vh_path,
        scene_id=args.scene_id,
        timestamp_utc=args.timestamp_utc,
        patch_size=cfg.get("patch_size", 256),
        config=cfg,
    )
