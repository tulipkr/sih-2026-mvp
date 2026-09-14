"""
Module 2 Stage A: raw Sentinel-1 -> calibrated, speckle-filtered,
land-masked, (optionally) normalized 256x256 VV/VH patches + manifest.json
matching Module 1's schema exactly (spec §7/§9).

MEMORY MODEL (important): a real Sentinel-1 IW GRD scene is ~25,000x17,000
px (~430M pixels). Nothing in this file ever reads, calibrates, filters, or
masks the whole scene into memory at once — every stage operates on one
haloed tile window at a time (see _process_windowed). At any moment, the
largest array alive is roughly (patch_size + speckle halo)^2, independent of
scene size. See src/calibration.py's build_sigma_naught_lut_window and
src/land_mask.py's mask_window for the two other places a naive
full-resolution version would be the obvious (wrong) thing to write.

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
import rasterio.windows
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.calibration import CalibrationError, calibrate_band, calibrate_window, parse_calibration_lut
from src.georeferencing import GeoreferencingError, resolve_geotransform
from src.land_mask import LandMaskError, apply_land_mask, load_land_mask_source, mask_window
from src.safe_reader import SafeProductError, locate_safe_product

logger = logging.getLogger(__name__)


class StageAError(ValueError):
    pass


def lee_filter(img: np.ndarray, size: int = 5) -> np.ndarray:
    """5x5 window Lee speckle filter. Called per-tile (on a haloed window),
    never on a full scene — see _process_windowed's halo handling."""
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


def _clip_window(row: int, col: int, patch_size: int, halo: int, height: int, width: int) -> Window:
    """The haloed read window for one tile, clipped to the scene's actual
    bounds (only matters at the very first/last row or column of tiles)."""
    row0 = max(0, row - halo)
    col0 = max(0, col - halo)
    row1 = min(height, row + patch_size + halo)
    col1 = min(width, col + patch_size + halo)
    return Window(col0, row0, col1 - col0, row1 - row0)


def _process_windowed(
    src_vv, src_vh, transform, crs, scene_id, timestamp_utc,
    calibrate_fn,       # callable(dn_window, band, row_start, col_start) -> sigma0_window (float32), or None
    already_db: bool,   # only used when calibrate_fn is None (mode 2)
    patch_size: int,
    speckle_cfg: dict, land_cfg: dict, scene_is_coastal: bool, norm_cfg: dict,
    patches_dir: Path,
):
    """The one real tiling loop both modes share. At any point in time, the
    only sizeable arrays alive are the current haloed window's VV/VH arrays
    (patch_size + halo)^2 — never the full scene. `src_vv`/`src_vh` must
    already be open rasterio datasets; this function only ever calls
    `.read(1, window=...)` on them, never a bare `.read(1)`.
    """
    height, width = src_vv.height, src_vv.width

    apply_speckle = speckle_cfg.get("apply", True)
    speckle_window = speckle_cfg.get("window", 5)
    halo = (speckle_window // 2) if apply_speckle else 0

    # Loaded/reprojected ONCE for the whole run, not once per tile — see
    # land_mask.py's module docstring for why re-reading a coastline source
    # per tile would be its own performance problem on a several-thousand-
    # tile scene.
    land_gdf = load_land_mask_source(land_cfg, crs, scene_is_coastal=scene_is_coastal)

    apply_db_clipping = norm_cfg.get("apply_db_clipping", False)
    apply_normalization = norm_cfg.get("apply_normalization", False)
    db_clip_min = norm_cfg.get("db_clip_min", -30)
    db_clip_max = norm_cfg.get("db_clip_max", 0)

    profile = {
        "driver": "GTiff", "count": 2, "dtype": rasterio.float32,
        "width": patch_size, "height": patch_size, "crs": crs, "nodata": None,
    }

    eps = 1e-6
    manifest = []
    tile_idx = 0
    skipped_partial_nan = 0

    for row in range(0, height - patch_size + 1, patch_size):
        for col in range(0, width - patch_size + 1, patch_size):
            halo_win = _clip_window(row, col, patch_size, halo, height, width)
            halo_row0, halo_col0 = int(halo_win.row_off), int(halo_win.col_off)

            # --- Bounded read: this tile's haloed window ONLY ---------------
            vv_dn = src_vv.read(1, window=halo_win).astype(np.float32)
            vh_dn = src_vh.read(1, window=halo_win).astype(np.float32)

            # --- Calibration, bounded to this window -------------------------
            if calibrate_fn is not None:
                vv_sigma0 = calibrate_fn(vv_dn, "vv", halo_row0, halo_col0)
                vh_sigma0 = calibrate_fn(vh_dn, "vh", halo_row0, halo_col0)
                vv_win = 10.0 * np.log10(np.maximum(vv_sigma0, eps))
                vh_win = 10.0 * np.log10(np.maximum(vh_sigma0, eps))
            elif already_db:
                vv_win, vh_win = vv_dn, vh_dn
            else:
                vv_win = 10.0 * np.log10(np.maximum(vv_dn, eps))
                vh_win = 10.0 * np.log10(np.maximum(vh_dn, eps))

            # --- Speckle filter on the haloed window, THEN crop the halo off -
            # (needs neighbor pixels beyond the core tile for correct 5x5
            # behavior at tile edges — that's what the halo is for)
            if apply_speckle:
                vv_win = lee_filter(vv_win, size=speckle_window)
                vh_win = lee_filter(vh_win, size=speckle_window)

            halo_transform = rasterio.windows.transform(halo_win, transform)
            vv_win, vh_win, _land_fraction = mask_window(vv_win, vh_win, halo_transform, land_gdf)

            core_row0 = row - halo_row0
            core_col0 = col - halo_col0
            vv_tile = vv_win[core_row0:core_row0 + patch_size, core_col0:core_col0 + patch_size]
            vh_tile = vh_win[core_row0:core_row0 + patch_size, core_col0:core_col0 + patch_size]

            if apply_db_clipping:
                vv_tile = np.clip(vv_tile, db_clip_min, db_clip_max)
                vh_tile = np.clip(vh_tile, db_clip_min, db_clip_max)
            if apply_normalization:
                denom = db_clip_max - db_clip_min
                vv_tile = (vv_tile - db_clip_min) / denom
                vh_tile = (vh_tile - db_clip_min) / denom

            vv_tile = vv_tile.astype(np.float32)
            vh_tile = vh_tile.astype(np.float32)

            nan_frac = max(np.isnan(vv_tile).mean(), np.isnan(vh_tile).mean())
            if nan_frac == 1.0:
                continue  # entirely land/nodata tile — not an error, just skip
            if nan_frac > 0.0:
                logger.warning(
                    f"scene_id={scene_id} tile row={row} col={col}: "
                    f"{nan_frac:.1%} NaN pixels — rejecting this tile rather "
                    f"than passing partial NaN to Module 1 (spec §16)."
                )
                skipped_partial_nan += 1
                continue

            core_win = Window(col, row, patch_size, patch_size)
            tile_transform = rasterio.windows.transform(core_win, transform)

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

    return manifest, tile_idx, skipped_partial_nan


def _run_safe_windowed(safe_path: Path, work_dir: Path, patches_dir: Path, patch_size: int, config: dict):
    product = locate_safe_product(safe_path, work_dir)
    logger.info(f"Located SAFE product {product.product_id}: "
                f"VV={product.vv_measurement_path.name}, VH={product.vh_measurement_path.name}")

    speckle_cfg = config.get("speckle_filter", {"apply": True, "window": 5})
    norm_cfg = config.get("normalization", {})
    land_cfg = config.get("land_mask", {"source_path": None})
    scene_is_coastal = config.get("scene_is_coastal", True)

    with rasterio.open(product.vv_measurement_path) as src_vv, \
         rasterio.open(product.vh_measurement_path) as src_vh:

        transform, crs, _is_gcp_approx = resolve_geotransform(src_vv)
        if (src_vh.height, src_vh.width) != (src_vv.height, src_vv.width):
            raise StageAError(
                f"VV measurement is {src_vv.height}x{src_vv.width} but VH measurement "
                f"is {src_vh.height}x{src_vh.width} — cannot pair mismatched-dimension "
                f"bands from the same product (spec §16)."
            )
        if crs is None or transform is None or transform.is_identity:
            raise StageAError(
                "CRS or affine transform missing/unresolvable — cannot proceed "
                "to tiling without georeferencing (spec §16)."
            )

        logger.info("Applying real radiometric calibration from the product's own LUT "
                    "(annotation/calibration/*.xml) — no placeholder constant. Calibration "
                    "vectors are parsed once here; the LUT is only ever expanded to one "
                    "tile's extent at a time, never full scene resolution.")
        vv_lines, vv_pixels, vv_sigma_grid = parse_calibration_lut(product.vv_calibration_path)
        vh_lines, vh_pixels, vh_sigma_grid = parse_calibration_lut(product.vh_calibration_path)

        def calibrate_fn(dn_window, band, row_start, col_start):
            if band == "vv":
                return calibrate_window(dn_window, vv_lines, vv_pixels, vv_sigma_grid, row_start, col_start)
            return calibrate_window(dn_window, vh_lines, vh_pixels, vh_sigma_grid, row_start, col_start)

        manifest, tile_idx, skipped = _process_windowed(
            src_vv, src_vh, transform, crs, product.product_id, product.acquisition_start_utc,
            calibrate_fn=calibrate_fn, already_db=False, patch_size=patch_size,
            speckle_cfg=speckle_cfg, land_cfg=land_cfg, scene_is_coastal=scene_is_coastal,
            norm_cfg=norm_cfg, patches_dir=patches_dir,
        )

    return manifest, tile_idx, skipped, product.product_id


def _run_direct_windowed(
    vv_path: Path, vh_path: Path, scene_id: str, timestamp_utc: str,
    patches_dir: Path, patch_size: int, config: dict,
):
    speckle_cfg = config.get("speckle_filter", {"apply": True, "window": 5})
    norm_cfg = config.get("normalization", {})
    land_cfg = config.get("land_mask", {"source_path": None})
    scene_is_coastal = config.get("scene_is_coastal", True)

    with rasterio.open(vv_path) as src_vv, rasterio.open(vh_path) as src_vh:
        transform, crs, _is_gcp_approx = resolve_geotransform(src_vv)
        if (src_vh.height, src_vh.width) != (src_vv.height, src_vv.width):
            raise StageAError(
                f"VV is {src_vv.height}x{src_vv.width} but VH is "
                f"{src_vh.height}x{src_vh.width} — cannot pair mismatched-dimension bands (spec §16)."
            )
        if crs is None or transform is None or transform.is_identity:
            raise StageAError(
                "CRS or affine transform missing/unresolvable — cannot proceed "
                "to tiling without georeferencing (spec §16)."
            )

        sample_win = Window(0, 0, min(512, src_vv.width), min(512, src_vv.height))
        sample_vv = src_vv.read(1, window=sample_win).astype(np.float32)
        already_db = _is_already_db(sample_vv)
        if already_db:
            logger.info(">> Detected input imagery is ALREADY in dB (mode 2 legacy heuristic, "
                        "sampled from a 512x512 window, not the full scene). Treating as "
                        "pre-calibrated per spec §35's documented shortcut.")
        else:
            logger.info(">> Detected LINEAR values (mode 2 legacy heuristic) — assuming "
                        "pre-calibrated Sigma0, converting to dB per tile. If this is "
                        "actually raw, uncalibrated DN, these values will be physically "
                        "wrong — use --safe_path with a real SAFE product instead of this "
                        "legacy path for genuine raw Sentinel-1 data.")

        manifest, tile_idx, skipped = _process_windowed(
            src_vv, src_vh, transform, crs, scene_id, timestamp_utc,
            calibrate_fn=None, already_db=already_db, patch_size=patch_size,
            speckle_cfg=speckle_cfg, land_cfg=land_cfg, scene_is_coastal=scene_is_coastal,
            norm_cfg=norm_cfg, patches_dir=patches_dir,
        )

    return manifest, tile_idx, skipped


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

    Both modes process the scene tile-by-tile with a bounded halo (see
    _process_windowed) — nothing here reads, calibrates, filters, or masks
    a full scene into memory regardless of its size.
    """
    config = config or {}
    out_root = Path(output_dir)
    patches_dir = out_root / "preprocessed_patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    if safe_path:
        manifest, tile_idx, skipped_partial_nan, real_scene_id = _run_safe_windowed(
            Path(safe_path), out_root / "_safe_extract", patches_dir, patch_size, config
        )
        scene_id = real_scene_id
    elif vv_path and vh_path:
        if not os.path.exists(vv_path) or not os.path.exists(vh_path):
            raise FileNotFoundError(f"Missing input files: {vv_path} or {vh_path}")
        if not scene_id or not timestamp_utc:
            raise StageAError(
                "Mode 2 (direct vv_path/vh_path) requires explicit scene_id and "
                "timestamp_utc — this mode has no product metadata to derive them "
                "from, and this pipeline never fabricates either (spec §20/§39)."
            )
        manifest, tile_idx, skipped_partial_nan = _run_direct_windowed(
            Path(vv_path), Path(vh_path), scene_id, timestamp_utc, patches_dir, patch_size, config
        )
    else:
        raise StageAError("Must supply either safe_path, or both vv_path and vh_path.")

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
