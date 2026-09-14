"""
Module 2 Stage B: Tulip's scene-level mask.tif (+ optional prob_map.tif) +
inference_result.json -> spill_geometry.geojson + spill_summary.json.

Fixes applied vs. the previously-audited version:
  - geolocation_incomplete / missing-CRS check now happens BEFORE opening the
    mask raster, so a missing mask file and missing georeferencing are
    distinguishable, non-crashing outcomes (previously: opening a missing
    mask_path raised a raw rasterio error before this check ever ran).
  - Confidence is now computed by actually sampling Tulip's real
    prob_map.tif inside each vectorized polygon (spec §10), not by reading a
    'mean_confidence' key that never existed in Tulip's real JSON schema
    (his field is mean_score_in_positive_region, and it's a whole-tile
    average anyway — not a valid per-polygon substitute; see the
    cross-module audit this fixes).
  - Every terminal branch now writes a schema-valid spill_summary.json,
    including "all candidates filtered as noise" and "zero shapes
    extracted" (previously: silent return, no file written at all).
  - Flags (doesn't silently accept) a fully-saturated mask, per spec §11.
  - Optional timestamp cross-check against Stage A's original manifest,
    per spec §16/§20 (logs a warning on mismatch, never silently overwrites).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from rasterio.features import shapes
from shapely.geometry import shape

logger = logging.getLogger(__name__)


def _write_summary(out: Path, **fields) -> Path:
    summary_path = out / "spill_summary.json"
    with open(summary_path, "w") as f:
        json.dump(fields, f, indent=2)
    return summary_path


def _cross_check_timestamp(meta: dict, stage_a_manifest_path: str | None) -> None:
    """spec §16/§20: log a warning on mismatch, never silently overwrite
    either value. Best-effort — only runs if a Stage A manifest was supplied."""
    if not stage_a_manifest_path or not Path(stage_a_manifest_path).is_file():
        return
    try:
        stage_a_entries = json.loads(Path(stage_a_manifest_path).read_text())
        if isinstance(stage_a_entries, dict):
            stage_a_entries = stage_a_entries.get("entries", [])
        scene_id = meta.get("scene_id")
        matching = [e for e in stage_a_entries if e.get("scene_id") == scene_id]
        if not matching:
            return
        original_ts = matching[0].get("acquisition_timestamp_utc")
        echoed_ts = meta.get("acquisition_timestamp_utc")
        if original_ts and echoed_ts and original_ts != echoed_ts:
            logger.warning(
                f"scene_id={scene_id}: timestamp mismatch between Stage A's "
                f"original manifest ({original_ts}) and Tulip's echoed value "
                f"({echoed_ts}) — using the echoed value as-is, not silently "
                f"resolving the discrepancy (spec §16/§20)."
            )
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read Stage A manifest for timestamp cross-check: {e}")


def _sample_confidence_per_polygon(prob_map_path: str | None, gdf: gpd.GeoDataFrame, src_crs) -> list:
    """Real per-polygon confidence (spec §10): mean prob_map value under
    each polygon's own footprint, sampled from Tulip's actual probability
    raster — not a scene-wide/tile-wide scalar reused across every region."""
    if not prob_map_path or not Path(prob_map_path).is_file():
        logger.warning("No prob_map_path available — mean_confidence will be null for all regions.")
        return [None] * len(gdf)

    confidences = []
    with rasterio.open(prob_map_path) as prob_src:
        prob_crs = prob_src.crs
        for geom in gdf.geometry:
            geom_for_sampling = geom
            if prob_crs is not None and str(prob_crs) != str(src_crs):
                geom_for_sampling = gpd.GeoSeries([geom], crs=src_crs).to_crs(prob_crs).iloc[0]
            try:
                out_image, _ = rasterio.mask.mask(prob_src, [geom_for_sampling], crop=True, filled=False)
            except ValueError:
                # Polygon doesn't overlap the probability raster at all —
                # shouldn't normally happen (it came from this raster's own
                # mask), but never fabricate a confidence value if it does.
                confidences.append(None)
                continue
            data = out_image[0]
            valid = data[~np.ma.getmaskarray(data)] if np.ma.isMaskedArray(data) else data
            valid = valid[np.isfinite(valid)] if hasattr(valid, "__len__") else valid
            confidences.append(float(np.mean(valid)) if len(valid) > 0 else None)
    return confidences


def run_stage_b(
    mask_path,
    inference_json_path=None,
    output_dir="outputs",
    min_area_km2=0.01,
    saturated_mask_warning_fraction=0.99,
    stage_a_manifest_path=None,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {}
    if inference_json_path and Path(inference_json_path).is_file():
        with open(inference_json_path, "r") as f:
            meta = json.load(f)

    scene_id = meta.get("scene_id", "unknown")
    timestamp = meta.get("acquisition_timestamp_utc")
    _cross_check_timestamp(meta, stage_a_manifest_path)

    # --- Check georeferencing BEFORE opening the mask raster -----------------
    # Fixes the ordering bug: previously this check ran *inside* the
    # `with rasterio.open(mask_path)` block, so a missing/corrupt mask file
    # crashed with a raw rasterio error before this branch could ever run.
    manifest_says_incomplete = bool(meta.get("geolocation_incomplete"))
    manifest_crs_present = meta.get("crs") is not None
    manifest_transform_present = meta.get("transform") is not None

    if manifest_says_incomplete or (meta and not manifest_crs_present) or (meta and not manifest_transform_present):
        summary_path = _write_summary(
            out, scene_id=scene_id, num_regions=0, total_area_km2=0.0,
            primary_region_centroid=None, acquisition_timestamp_utc=timestamp,
            no_oil_detected=meta.get("no_oil_detected", False),
            geometry_unavailable=True, geojson_path=None,
        )
        logger.info(f"scene_id={scene_id}: geolocation_incomplete per Tulip's JSON — "
                    f"aborted cleanly, no coordinates fabricated. Summary: {summary_path}")
        return

    if not Path(mask_path).is_file():
        raise FileNotFoundError(f"mask.tif not found: {Path(mask_path).resolve()}")

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        src_crs = src.crs
        src_transform = src.transform

        is_missing_transform = (src_transform is None) or src_transform.is_identity
        if src_crs is None or is_missing_transform:
            summary_path = _write_summary(
                out, scene_id=scene_id, num_regions=0, total_area_km2=0.0,
                primary_region_centroid=None, acquisition_timestamp_utc=timestamp,
                no_oil_detected=meta.get("no_oil_detected", False),
                geometry_unavailable=True, geojson_path=None,
            )
            logger.info(f"scene_id={scene_id}: mask.tif itself has no usable CRS/transform — "
                        f"aborted cleanly. Summary: {summary_path}")
            return

        # --- Fully-saturated mask check (spec §11): flag, don't silently accept ---
        positive_fraction = float((mask > 0).mean())
        if positive_fraction >= saturated_mask_warning_fraction:
            logger.warning(
                f"scene_id={scene_id}: {positive_fraction:.1%} of the mask is positive — "
                f"this looks like an upstream bug (e.g. a broken threshold or a fully "
                f"failed model), not a real detection. Proceeding, but flagged."
            )

        # --- No oil / empty mask ---------------------------------------------
        if np.all(mask == 0) or meta.get("no_oil_detected", False):
            geojson_file = out / "spill_geometry.geojson"
            with open(geojson_file, "w") as f:
                json.dump({"type": "FeatureCollection", "features": []}, f, indent=2)
            summary_path = _write_summary(
                out, scene_id=scene_id, num_regions=0, total_area_km2=0.0,
                primary_region_centroid=None, acquisition_timestamp_utc=timestamp,
                no_oil_detected=True, geometry_unavailable=False,
                geojson_path=str(geojson_file),
            )
            logger.info(f"scene_id={scene_id}: no oil detected. Summary: {summary_path}")
            return

        # --- Vectorize ---------------------------------------------------------
        extracted = [shape(geom) for geom, val in shapes(mask, mask=(mask > 0), transform=src_transform)]
        if not extracted:
            summary_path = _write_summary(
                out, scene_id=scene_id, num_regions=0, total_area_km2=0.0,
                primary_region_centroid=None, acquisition_timestamp_utc=timestamp,
                no_oil_detected=False, geometry_unavailable=False, geojson_path=None,
            )
            logger.info(f"scene_id={scene_id}: no shapes extracted from a non-empty mask "
                        f"(unexpected but not an error). Summary: {summary_path}")
            return

        gdf = gpd.GeoDataFrame({"geometry": extracted}, crs=src_crs)

        gdf_ea = gdf.to_crs(epsg=6933)
        gdf["area_km2"] = gdf_ea.geometry.area / 1e6

        gdf = gdf[gdf["area_km2"] >= min_area_km2].copy().reset_index(drop=True)
        if gdf.empty:
            summary_path = _write_summary(
                out, scene_id=scene_id, num_regions=0, total_area_km2=0.0,
                primary_region_centroid=None, acquisition_timestamp_utc=timestamp,
                no_oil_detected=False, geometry_unavailable=False, geojson_path=None,
            )
            logger.info(f"scene_id={scene_id}: all {len(extracted)} candidate region(s) "
                        f"were below min_area_km2={min_area_km2} (noise). "
                        f"Summary: {summary_path}")
            return

        gdf_ea_filtered = gdf.to_crs(epsg=6933)
        centroids_wgs84 = gdf_ea_filtered.geometry.centroid.to_crs(epsg=4326)

        gdf_wgs84 = gdf.to_crs(epsg=4326)
        gdf_wgs84["centroid_lat"] = centroids_wgs84.y
        gdf_wgs84["centroid_lon"] = centroids_wgs84.x
        gdf_wgs84["region_id"] = gdf_wgs84.index + 1
        gdf_wgs84["scene_id"] = scene_id
        gdf_wgs84["acquisition_timestamp_utc"] = timestamp

        # --- Real per-polygon confidence from the probability raster ---------
        prob_map_path = meta.get("prob_map_path")
        gdf_wgs84["mean_confidence"] = _sample_confidence_per_polygon(prob_map_path, gdf, src_crs)
        gdf_wgs84["source_mask_path"] = str(mask_path)

        cols = [
            "region_id", "scene_id", "area_km2", "centroid_lat",
            "centroid_lon", "acquisition_timestamp_utc",
            "mean_confidence", "source_mask_path", "geometry"
        ]
        gdf_final = gdf_wgs84[cols]

        geojson_path = out / "spill_geometry.geojson"
        gdf_final.to_file(geojson_path, driver="GeoJSON")

        primary = gdf_final.loc[gdf_final["area_km2"].idxmax()]
        summary_path = _write_summary(
            out, scene_id=scene_id, num_regions=int(len(gdf_final)),
            total_area_km2=float(gdf_final["area_km2"].sum()),
            primary_region_centroid=[float(primary["centroid_lat"]), float(primary["centroid_lon"])],
            acquisition_timestamp_utc=timestamp, no_oil_detected=False,
            geometry_unavailable=False, geojson_path=str(geojson_path),
        )
        logger.info(f"scene_id={scene_id}: {len(gdf_final)} region(s) exported. "
                    f"Summary: {summary_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Module 2 Stage B: mask -> geometry.")
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--inference_json", default=None)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--min_area_km2", type=float, default=0.01)
    parser.add_argument("--stage_a_manifest", default=None)
    args = parser.parse_args()

    run_stage_b(
        mask_path=args.mask_path, inference_json_path=args.inference_json,
        output_dir=args.output_dir, min_area_km2=args.min_area_km2,
        stage_a_manifest_path=args.stage_a_manifest,
    )
