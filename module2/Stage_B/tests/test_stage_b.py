import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from affine import Affine

from postprocess_pipeline import run_stage_b


def _write_mask(path, mask, transform, crs="EPSG:3857"):
    with rasterio.open(
        path, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1], count=1,
        dtype=rasterio.uint8, crs=crs, transform=transform,
    ) as dst:
        dst.write(mask, 1)


def _write_prob_map(path, probs, transform, crs="EPSG:3857"):
    with rasterio.open(
        path, "w", driver="GTiff", height=probs.shape[0], width=probs.shape[1], count=1,
        dtype=rasterio.float32, crs=crs, transform=transform,
    ) as dst:
        dst.write(probs.astype(np.float32), 1)


# --- 1. Valid input / area computation --------------------------------------
def test_area_computation_and_vectorize(tmp_path):
    mask_path = tmp_path / "test_mask.tif"
    json_path = tmp_path / "test_inference.json"
    out_dir = tmp_path / "output"

    transform = Affine(10.0, 0, 0, 0, -10.0, 0)  # 10m pixels -> 100x100 px = 1.0 km^2
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 1
    _write_mask(mask_path, mask, transform)

    meta = {
        "scene_id": "TEST_SCENE", "crs": "EPSG:3857", "transform": list(transform)[:6],
        "acquisition_timestamp_utc": "2023-12-04T00:30:00Z", "mask_path": str(mask_path),
        "no_oil_detected": False, "geolocation_incomplete": False,
    }
    json_path.write_text(json.dumps(meta))

    run_stage_b(str(mask_path), str(json_path), str(out_dir), min_area_km2=0.01)

    with open(out_dir / "spill_summary.json") as f:
        summary = json.load(f)
    assert summary["num_regions"] == 1
    assert pytest.approx(summary["total_area_km2"], rel=1e-2) == 1.0


# --- 12. Probability-map-based confidence calculation (the interface fix) --
def test_confidence_sampled_per_polygon_from_prob_map(tmp_path):
    """Two disjoint regions with DIFFERENT probability levels must get
    DIFFERENT mean_confidence values — proves this isn't a shared scalar
    copied from a (nonexistent) 'mean_confidence' JSON field."""
    mask_path = tmp_path / "mask.tif"
    prob_path = tmp_path / "prob_map.tif"
    json_path = tmp_path / "inference.json"
    out_dir = tmp_path / "output"

    transform = Affine(10.0, 0, 0, 0, -10.0, 0)
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[20:80, 20:80] = 1      # region A: high confidence
    mask[200:260, 200:260] = 1  # region B: low confidence

    probs = np.zeros((300, 300), dtype=np.float32)
    probs[20:80, 20:80] = 0.95
    probs[200:260, 200:260] = 0.55

    _write_mask(mask_path, mask, transform)
    _write_prob_map(prob_path, probs, transform)

    meta = {
        "scene_id": "CONF_TEST", "crs": "EPSG:3857", "transform": list(transform)[:6],
        "acquisition_timestamp_utc": "2023-12-04T00:30:00Z",
        "no_oil_detected": False, "geolocation_incomplete": False,
        "prob_map_path": str(prob_path),
        "mean_score_in_positive_region": 0.75,  # Tulip's REAL field — must NOT be used directly
    }
    json_path.write_text(json.dumps(meta))

    run_stage_b(str(mask_path), str(json_path), str(out_dir), min_area_km2=0.001)

    features = json.loads((out_dir / "spill_geometry.geojson").read_text())["features"]
    confidences = sorted(f["properties"]["mean_confidence"] for f in features)
    assert len(confidences) == 2
    assert confidences[0] == pytest.approx(0.55, abs=0.02)
    assert confidences[1] == pytest.approx(0.95, abs=0.02)
    assert confidences[0] != confidences[1]  # not a shared scalar


def test_confidence_null_when_no_prob_map_available(tmp_path):
    mask_path = tmp_path / "mask.tif"
    json_path = tmp_path / "inference.json"
    out_dir = tmp_path / "output"
    transform = Affine(10.0, 0, 0, 0, -10.0, 0)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:50, 10:50] = 1
    _write_mask(mask_path, mask, transform)
    meta = {"scene_id": "NO_PROB", "crs": "EPSG:3857", "transform": list(transform)[:6],
            "geolocation_incomplete": False, "no_oil_detected": False}
    json_path.write_text(json.dumps(meta))

    run_stage_b(str(mask_path), str(json_path), str(out_dir), min_area_km2=0.001)
    features = json.loads((out_dir / "spill_geometry.geojson").read_text())["features"]
    assert features[0]["properties"]["mean_confidence"] is None


# --- min-area filter — now also checks the summary is always written -------
def test_min_area_filter_still_writes_summary(tmp_path):
    mask_path = tmp_path / "noise_mask.tif"
    json_path = tmp_path / "test_inference.json"
    out_dir = tmp_path / "output"

    transform = Affine(10.0, 0, 0, 0, -10.0, 0)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10, 10] = 1  # single pixel = 0.0001 km^2
    _write_mask(mask_path, mask, transform)

    meta = {"scene_id": "NOISE_SCENE", "crs": "EPSG:3857", "transform": list(transform)[:6],
            "acquisition_timestamp_utc": "2023-12-04T00:30:00Z",
            "no_oil_detected": False, "geolocation_incomplete": False}
    json_path.write_text(json.dumps(meta))

    run_stage_b(str(mask_path), str(json_path), str(out_dir), min_area_km2=0.01)

    assert not (out_dir / "spill_geometry.geojson").exists()
    # FIX vs. previously-audited version: a summary must still exist, num_regions=0.
    summary = json.loads((out_dir / "spill_summary.json").read_text())
    assert summary["num_regions"] == 0
    assert summary["geometry_unavailable"] is False


# --- 5/13. Missing/corrupt georeferencing & missing metadata ---------------
def test_missing_georeferencing_no_crash_no_fabrication(tmp_path):
    """FIX: geolocation_incomplete is checked from meta BEFORE the mask file
    is ever opened — the mask file need not even exist for this branch."""
    json_path = tmp_path / "meta_no_crs.json"
    out_dir = tmp_path / "output"
    meta = {"scene_id": "NO_GEO", "crs": None, "transform": None,
            "acquisition_timestamp_utc": "2023-12-04T00:30:00Z", "geolocation_incomplete": True}
    json_path.write_text(json.dumps(meta))

    run_stage_b(str(tmp_path / "nonexistent_mask.tif"), str(json_path), str(out_dir))

    summary = json.loads((out_dir / "spill_summary.json").read_text())
    assert summary["geometry_unavailable"] is True


def test_missing_mask_file_raises_explicit_error(tmp_path):
    """Distinct from the above: georeferencing IS present in meta, but the
    mask file itself doesn't exist -> a clean, explicit error, not a
    geometry_unavailable summary and not a raw rasterio traceback."""
    json_path = tmp_path / "meta.json"
    out_dir = tmp_path / "output"
    meta = {"scene_id": "X", "crs": "EPSG:3857", "transform": [10, 0, 0, 0, -10, 0],
            "geolocation_incomplete": False}
    json_path.write_text(json.dumps(meta))

    with pytest.raises(FileNotFoundError):
        run_stage_b(str(tmp_path / "does_not_exist.tif"), str(json_path), str(out_dir))


# --- 9/10. No oil detected / completely empty mask --------------------------
def test_no_oil_detected_flag_short_circuits(tmp_path):
    mask_path = tmp_path / "mask.tif"
    json_path = tmp_path / "meta.json"
    out_dir = tmp_path / "output"
    transform = Affine(10.0, 0, 0, 0, -10.0, 0)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:10, 5:10] = 1  # would be real oil if not for the flag below
    _write_mask(mask_path, mask, transform)
    meta = {"scene_id": "FLAG_TEST", "crs": "EPSG:3857", "transform": list(transform)[:6],
            "no_oil_detected": True, "geolocation_incomplete": False}
    json_path.write_text(json.dumps(meta))

    run_stage_b(str(mask_path), str(json_path), str(out_dir))
    summary = json.loads((out_dir / "spill_summary.json").read_text())
    assert summary["no_oil_detected"] is True
    features = json.loads((out_dir / "spill_geometry.geojson").read_text())["features"]
    assert features == []


def test_completely_empty_mask_without_flag(tmp_path):
    mask_path = tmp_path / "mask.tif"
    json_path = tmp_path / "meta.json"
    out_dir = tmp_path / "output"
    transform = Affine(10.0, 0, 0, 0, -10.0, 0)
    mask = np.zeros((50, 50), dtype=np.uint8)  # all-zero, no flag needed
    _write_mask(mask_path, mask, transform)
    meta = {"scene_id": "EMPTY", "crs": "EPSG:3857", "transform": list(transform)[:6],
            "no_oil_detected": False, "geolocation_incomplete": False}
    json_path.write_text(json.dumps(meta))

    run_stage_b(str(mask_path), str(json_path), str(out_dir))
    summary = json.loads((out_dir / "spill_summary.json").read_text())
    assert summary["no_oil_detected"] is True
    assert summary["num_regions"] == 0


# --- 11. Multiple detected regions ------------------------------------------
def test_multiple_disjoint_regions_become_separate_features(tmp_path):
    mask_path = tmp_path / "mask.tif"
    json_path = tmp_path / "meta.json"
    out_dir = tmp_path / "output"
    transform = Affine(10.0, 0, 0, 0, -10.0, 0)
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[20:80, 20:80] = 1
    mask[150:220, 150:220] = 1
    mask[260:290, 260:290] = 1
    _write_mask(mask_path, mask, transform)
    meta = {"scene_id": "MULTI", "crs": "EPSG:3857", "transform": list(transform)[:6],
            "no_oil_detected": False, "geolocation_incomplete": False}
    json_path.write_text(json.dumps(meta))

    run_stage_b(str(mask_path), str(json_path), str(out_dir), min_area_km2=0.001)
    summary = json.loads((out_dir / "spill_summary.json").read_text())
    assert summary["num_regions"] == 3
    features = json.loads((out_dir / "spill_geometry.geojson").read_text())["features"]
    assert len(features) == 3
    # primary region = largest area
    biggest = max(features, key=lambda f: f["properties"]["area_km2"])
    assert summary["primary_region_centroid"] == [
        biggest["properties"]["centroid_lat"], biggest["properties"]["centroid_lon"]
    ]


# --- Fully-saturated mask warning (spec §11) --------------------------------
def test_fully_saturated_mask_still_processes_but_warns(tmp_path, caplog):
    mask_path = tmp_path / "mask.tif"
    json_path = tmp_path / "meta.json"
    out_dir = tmp_path / "output"
    transform = Affine(10.0, 0, 0, 0, -10.0, 0)
    mask = np.ones((50, 50), dtype=np.uint8)  # 100% positive
    _write_mask(mask_path, mask, transform)
    meta = {"scene_id": "SATURATED", "crs": "EPSG:3857", "transform": list(transform)[:6],
            "no_oil_detected": False, "geolocation_incomplete": False}
    json_path.write_text(json.dumps(meta))

    import logging
    with caplog.at_level(logging.WARNING):
        run_stage_b(str(mask_path), str(json_path), str(out_dir), min_area_km2=0.0001)
    assert any("upstream bug" in r.message for r in caplog.records)
