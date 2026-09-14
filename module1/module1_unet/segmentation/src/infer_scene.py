"""
Scene-level inference orchestrator (NEW — bridges the Module 1 <-> Module 2
interface gap identified in the cross-module audit).

Ishita's Stage A manifest gives every tile of one scene the SAME scene_id
(only patch_path/patch_id differ). infer.py's run_inference() answers "run
inference on one patch"; Module 2 Stage B needs one whole-scene mask.tif +
prob_map.tif + inference_result.json.

This module does NOT change the model, training code, or infer.py's existing
single-patch behavior — it composes the same helpers (load_and_validate_patch,
resolve_inference_backend, run_tile_inference) infer.py already exposes, runs
them once per tile of a scene, and stitches the results using each tile's own
affine transform to recover its pixel offset within the scene. No separate
"scene transform" needs to be published by Ishita — it's reconstructed purely
from the collection of per-tile transforms already in the manifest.

Usage:
    python -m src.infer_scene --config configs/config.yaml --scene_id S1B_..._3EE4
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.dataset import load_manifest_entries
from src.infer import (
    _find_entries_for_scene,
    _write_geotiff,
    load_and_validate_patch,
    resolve_inference_backend,
    run_tile_inference,
)
from src.schema import validate_inference_result
from src.utils import load_config, setup_logging, write_json

logger = logging.getLogger(__name__)


def _stitch(tiles: list[dict], patch_size: int) -> dict:
    """Reconstruct scene-level canvas position for every tile from its own
    affine transform. Tiles missing crs/transform are excluded from the
    mosaic (their pixels are left as "no detection" in the stitched output)
    but flag the whole scene as geolocation_incomplete, since we can't be
    sure the rest of the mosaic's placement is trustworthy either."""
    # NOTE: filtered by a boolean condition, not `in`/`not in` — tile dicts
    # contain numpy arrays, and `dict == dict` on those raises ValueError
    # ("truth value of an array is ambiguous").
    geo_tiles = [t for t in tiles if t["crs"] is not None and t["transform"] is not None]
    missing_geo = [t for t in tiles if t["crs"] is None or t["transform"] is None]

    if not geo_tiles:
        return {
            "crs": None,
            "transform": None,
            "geolocation_incomplete": True,
            "stitched_probs": None,
            "stitched_mask": None,
            "num_tiles_missing_geolocation": len(tiles),
        }

    crs_values = {t["crs"] for t in geo_tiles}
    if len(crs_values) > 1:
        raise ValueError(
            f"Tiles of this scene report different CRS values {crs_values} — "
            f"cannot stitch inconsistent georeferencing. This is a Stage A bug, "
            f"not something to silently resolve here."
        )

    pixel_sizes = {(round(t["transform"][0], 9), round(t["transform"][4], 9)) for t in geo_tiles}
    if len(pixel_sizes) > 1:
        raise ValueError(
            f"Tiles of this scene report different pixel sizes {pixel_sizes} — "
            f"cannot stitch. This is a Stage A tiling bug, not something to "
            f"silently resolve here."
        )
    a, e = next(iter(pixel_sizes))

    scene_min_c = min(t["transform"][2] for t in geo_tiles)
    scene_max_f = max(t["transform"][5] for t in geo_tiles)

    placements = []
    max_row = max_col = 0
    for t in geo_tiles:
        c, f = t["transform"][2], t["transform"][5]
        col_off = round((c - scene_min_c) / a)
        row_off = round((scene_max_f - f) / abs(e))
        h, w = t["binary_mask"].shape
        placements.append((t, row_off, col_off, h, w))
        max_row = max(max_row, row_off + h)
        max_col = max(max_col, col_off + w)

    stitched_probs = np.zeros((max_row, max_col), dtype=np.float32)
    stitched_mask = np.zeros((max_row, max_col), dtype=np.uint8)
    for t, row_off, col_off, h, w in placements:
        stitched_probs[row_off:row_off + h, col_off:col_off + w] = t["probs"]
        stitched_mask[row_off:row_off + h, col_off:col_off + w] = t["binary_mask"]

    b, d = 0.0, 0.0
    transform = [a, b, scene_min_c, d, e, scene_max_f]

    if missing_geo:
        logger.warning(
            f"{len(missing_geo)} of {len(tiles)} tiles in this scene are missing "
            f"CRS/transform and were excluded from the mosaic (their area reads "
            f"as 'no detection' in the stitched output, not a real absence). "
            f"Scene marked geolocation_incomplete."
        )

    return {
        "crs": next(iter(crs_values)),
        "transform": transform,
        "geolocation_incomplete": bool(missing_geo),
        "stitched_probs": stitched_probs,
        "stitched_mask": stitched_mask,
        "num_tiles_missing_geolocation": len(missing_geo),
    }


def run_scene_inference(config_path: str, scene_id: str, checkpoint_path: str | None = None) -> dict:
    config = load_config(config_path)
    setup_logging(config["logging"]["log_dir"], config["logging"].get("level", "INFO"))

    entries = load_manifest_entries(config["data"]["inference_manifest_path"])
    tile_entries = _find_entries_for_scene(entries, scene_id)
    logger.info(f"scene_id={scene_id}: found {len(tile_entries)} tile(s) in manifest.")

    # Resolve model-vs-fallback exactly once for the whole scene — a scene's
    # tiles must never be processed under a mix of the two.
    fallback_used, model, device, checkpoint_path = resolve_inference_backend(config, checkpoint_path)

    tiles = []
    acquisition_timestamps = set()
    score_type = model_version = None
    for entry in tile_entries:
        patch_label = entry.get("patch_id") or entry["patch_path"]
        label = f"{scene_id}/{patch_label}"
        image, crs, transform, tile_geo_incomplete = load_and_validate_patch(entry, config, label)
        probs, binary_mask, score_type, model_version = run_tile_inference(
            image, config, fallback_used, model, device
        )
        tiles.append({
            "entry": entry, "crs": crs, "transform": transform,
            "geolocation_incomplete": tile_geo_incomplete,
            "probs": probs, "binary_mask": binary_mask,
        })
        acquisition_timestamps.add(entry.get("acquisition_timestamp_utc"))

    if len(acquisition_timestamps) > 1:
        logger.warning(
            f"scene_id={scene_id}: tiles disagree on acquisition_timestamp_utc "
            f"{acquisition_timestamps} — using the first tile's value. This "
            f"usually means Stage A wrote inconsistent metadata across tiles "
            f"of the same scene; worth checking, not silently averaging."
        )
    acquisition_timestamp_utc = tile_entries[0].get("acquisition_timestamp_utc")

    stitched = _stitch(tiles, config["data"]["patch_size"])

    output_dir = Path(config["inference"]["output_dir"]) / "scenes" / scene_id
    prob_map_path = output_dir / "prob_map.tif"
    mask_path = output_dir / "mask.tif"

    if stitched["stitched_probs"] is not None:
        _write_geotiff(prob_map_path, stitched["stitched_probs"], stitched["transform"],
                        stitched["crs"], dtype="float32")
        _write_geotiff(mask_path, stitched["stitched_mask"], stitched["transform"],
                        stitched["crs"], dtype="uint8")
        positive_pixel_fraction = float(stitched["stitched_mask"].mean())
        mean_score_in_positive_region = (
            float(stitched["stitched_probs"][stitched["stitched_mask"] == 1].mean())
            if positive_pixel_fraction > 0 else None
        )
    else:
        # No tile in this scene has usable georeferencing at all — write empty
        # placeholder rasters (still real files, not fabricated data: all-zero,
        # 1x1) so mask_path/prob_map_path always point at something that exists.
        _write_geotiff(prob_map_path, np.zeros((1, 1), dtype=np.float32), None, None, dtype="float32")
        _write_geotiff(mask_path, np.zeros((1, 1), dtype=np.uint8), None, None, dtype="uint8")
        positive_pixel_fraction = 0.0
        mean_score_in_positive_region = None

    no_oil_detected = positive_pixel_fraction == 0.0

    result = {
        "scene_id": scene_id,
        "model_version": model_version,
        "inference_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "acquisition_timestamp_utc": acquisition_timestamp_utc,
        "crs": stitched["crs"],
        "transform": stitched["transform"],
        "geolocation_incomplete": stitched["geolocation_incomplete"],
        "threshold_used": float(config["inference"]["threshold"]),
        "positive_pixel_fraction": positive_pixel_fraction,
        "mean_score_in_positive_region": mean_score_in_positive_region,
        "no_oil_detected": no_oil_detected,
        "score_type": score_type,
        "fallback_used": fallback_used,
        "prob_map_path": str(prob_map_path),
        "mask_path": str(mask_path),
        # Extra fields beyond schema §9 — additive only, never required by
        # validate_inference_result, purely diagnostic for this stitched path.
        "num_tiles_stitched": len(tiles),
        "num_tiles_missing_geolocation": stitched["num_tiles_missing_geolocation"],
    }

    validate_inference_result(result)

    result_path = Path(config["inference"]["output_dir"]) / f"{scene_id}_inference_result.json"
    write_json(result_path, result)
    logger.info(
        f"Scene-level inference complete for {scene_id}: {len(tiles)} tile(s) stitched, "
        f"fallback_used={fallback_used}, geolocation_incomplete={stitched['geolocation_incomplete']}. "
        f"Result written to {result_path}"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on every tile of one scene and stitch.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--scene_id", required=True)
    args = parser.parse_args()
    result = run_scene_inference(args.config, args.scene_id, args.checkpoint)
    print(result)
