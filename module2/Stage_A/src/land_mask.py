"""
Land/ocean masking (Module 2 spec §2/§4/§10/§12/§15/§16/§17).

The spec is explicit and repeated: skipping this for a coastal scene is not
acceptable, and a bundled coarse fallback should be used if no external
source is configured — but a *fabricated* coastline (guessed geometry) is
strictly worse than refusing to run, since it produces false confidence
rather than an honest error. This module ships the real masking mechanics
(given any real coastline polygon source) and a documented, honest gap:
no coastline dataset is bundled here (none could be sourced in this
environment — no network access), so a genuinely missing source on a scene
flagged coastal is a **blocking error**, matching spec §15's explicit
instruction, rather than a silent skip or an invented placeholder polygon.

Before a real run: set `land_mask_source_path` in config.yaml to a real
coastline/ocean polygon file (e.g. Natural Earth 1:10m land or ocean
polygons, downloaded once and cached locally per spec §21/§34's guidance to
pre-cache everything needed for the demo scenes ahead of time).

Split into a one-time load (`load_land_mask_source`) and a cheap per-window
mask (`mask_window`) so large-scene windowed processing reads/reprojects the
coastline source exactly once per run, not once per tile — a real scene can
have several thousand tiles, and re-reading a shapefile/GeoJSON from disk
that many times would be its own (separate) performance problem.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class LandMaskError(ValueError):
    pass


def load_land_mask_source(config: dict[str, Any], crs, scene_is_coastal: bool = True):
    """Load and reproject the coastline source ONCE per Stage A run. Returns
    a GeoDataFrame already in `crs` (so mask_window never has to reproject
    per-tile), or None if masking isn't applicable (non-coastal scene, no
    source configured).

    Raises:
        LandMaskError: missing source on a coastal scene (spec §15 — this is
        a blocking error, not a warning), or an unreadable/misaligned source.
    """
    source_path = config.get("source_path")

    if not source_path:
        if scene_is_coastal:
            raise LandMaskError(
                "land_mask.source_path is not configured, and this scene is "
                "marked coastal (scene_is_coastal=True). Spec §15: missing "
                "land mask for a coastal scene is a BLOCKING error, not a "
                "soft warning — skipping masking risks reading land pixels "
                "as spill. Set land_mask.source_path in config.yaml to a "
                "real coastline/ocean polygon file (e.g. a Natural Earth "
                "1:10m land or ocean layer downloaded ahead of time) before "
                "running on a real coastal demo scene."
            )
        return None

    source_path = Path(source_path)
    if not source_path.exists():
        raise LandMaskError(f"land_mask.source_path does not exist: {source_path.resolve()}")

    import geopandas as gpd

    try:
        land_gdf = gpd.read_file(source_path)
    except Exception as e:
        raise LandMaskError(f"Could not read land mask source {source_path}: {e}") from e

    if land_gdf.crs is None:
        raise LandMaskError(
            f"Land mask source {source_path} has no CRS — cannot reliably "
            f"align it against the scene without one."
        )
    if str(land_gdf.crs) != str(crs):
        land_gdf = land_gdf.to_crs(crs)

    return land_gdf


def mask_window(vv: np.ndarray, vh: np.ndarray, transform, land_gdf) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply land masking to ONE bounded window/tile, using an
    already-loaded, already-reprojected `land_gdf` (see
    load_land_mask_source — call that once per run, this once per tile).
    Never builds anything larger than this window's own (H, W) shape."""
    if land_gdf is None:
        return vv, vh, 0.0

    import rasterio.features
    import rasterio.transform

    height, width = vv.shape
    window_bounds = rasterio.transform.array_bounds(height, width, transform)
    local_gdf = land_gdf.cx[window_bounds[0]:window_bounds[2], window_bounds[1]:window_bounds[3]]

    if local_gdf.empty:
        # Genuinely no land in this window — valid (open-ocean tile), not an error.
        return vv, vh, 0.0

    land_raster_mask = rasterio.features.geometry_mask(
        local_gdf.geometry, out_shape=(height, width), transform=transform, invert=True,
    )  # True where LAND — bounded to (height, width) of this window, never the full scene

    land_fraction = float(land_raster_mask.mean())

    masked_vv = vv.copy()
    masked_vh = vh.copy()
    masked_vv[land_raster_mask] = np.nan
    masked_vh[land_raster_mask] = np.nan

    return masked_vv, masked_vh, land_fraction


def apply_land_mask(
    vv: np.ndarray, vh: np.ndarray, transform, crs, config: dict[str, Any], scene_is_coastal: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Backward-compatible single-shot entrypoint: loads the source AND
    masks in one call. Fine for a single whole-array call (existing tests,
    genuinely small inputs) — real large-scene windowed processing should
    call load_land_mask_source() once and mask_window() per tile instead,
    so a multi-thousand-tile scene doesn't re-read/reproject the coastline
    source from disk on every tile.
    """
    land_gdf = load_land_mask_source(config, crs, scene_is_coastal=scene_is_coastal)
    return mask_window(vv, vh, transform, land_gdf)
