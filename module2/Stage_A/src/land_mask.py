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
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class LandMaskError(ValueError):
    pass


def apply_land_mask(
    vv: np.ndarray,
    vh: np.ndarray,
    transform,
    crs: str,
    config: dict[str, Any],
    scene_is_coastal: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Zero/NaN-masks land pixels in both bands using a real coastline source.

    Args:
        vv, vh: (H, W) arrays for this tile/scene, in the raster's own pixel grid.
        transform: rasterio Affine for this array.
        crs: the array's CRS (string, e.g. "EPSG:4326").
        config: the `land_mask` config block — must have `source_path` (or None).
        scene_is_coastal: if True (the default — Huntington Beach and most
            real oil-spill demo scenes ARE coastal per spec §15), a missing
            source is a hard error. Set False only for genuinely open-ocean
            scenes where land masking is not applicable.

    Returns:
        (masked_vv, masked_vh, land_fraction) — land_fraction is logged by
        the caller per spec §26 ("land-mask coverage %").

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
        return vv, vh, 0.0

    source_path = Path(source_path)
    if not source_path.exists():
        raise LandMaskError(f"land_mask.source_path does not exist: {source_path.resolve()}")

    import geopandas as gpd
    import rasterio.features
    import rasterio.transform

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

    height, width = vv.shape
    scene_bounds = rasterio.transform.array_bounds(height, width, transform)
    land_gdf = land_gdf.cx[scene_bounds[0]:scene_bounds[2], scene_bounds[1]:scene_bounds[3]]

    if land_gdf.empty:
        # Genuinely no land in this scene's bounding box — valid (open ocean
        # crop of a larger coastline dataset), not an error.
        return vv, vh, 0.0

    land_raster_mask = rasterio.features.geometry_mask(
        land_gdf.geometry, out_shape=(height, width), transform=transform, invert=True,
    )  # True where LAND

    land_fraction = float(land_raster_mask.mean())

    masked_vv = vv.copy()
    masked_vh = vh.copy()
    masked_vv[land_raster_mask] = np.nan
    masked_vh[land_raster_mask] = np.nan

    return masked_vv, masked_vh, land_fraction
