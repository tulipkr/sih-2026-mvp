"""
Integration-level edge-case tests. The previous version of this file only
asserted trivial facts about a locally-constructed dict — it never called
run_pipeline at all, so it provided no real coverage of the short-circuit
behavior spec §16/§29 actually require. Rewritten to exercise the real
pipeline end-to-end for each case.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.source_estimation_pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_short_circuit_no_oil(tmp_path):
    summary = _write(tmp_path / "summary.json", {
        "scene_id": "NO_OIL_SCENE", "no_oil_detected": True,
        "geometry_unavailable": False, "num_regions": 0,
    })
    # geometry may legitimately be an empty FeatureCollection in this case —
    # the pipeline must short-circuit before ever needing a real region.
    geometry = _write(tmp_path / "geometry.geojson", {"type": "FeatureCollection", "features": []})
    out = tmp_path / "source_estimate.json"

    result = run_pipeline(summary, geometry, config_path=ROOT / "configs" / "config.yaml", output_path=out)

    assert result["backtracking_valid"] is False
    assert result["invalid_reason"] == "no_spill_detected"
    assert result["no_oil_detected"] is True
    assert result["drift_trajectory"] == []
    assert result["fallback_used"] is False
    saved = json.loads(out.read_text())
    assert saved == result


def test_short_circuit_missing_geometry(tmp_path):
    summary = _write(tmp_path / "summary.json", {
        "scene_id": "GEO_UNAVAILABLE_SCENE", "no_oil_detected": False,
        "geometry_unavailable": True, "num_regions": 0,
    })
    geometry = _write(tmp_path / "geometry.geojson", {"type": "FeatureCollection", "features": []})
    out = tmp_path / "source_estimate.json"

    result = run_pipeline(summary, geometry, config_path=ROOT / "configs" / "config.yaml", output_path=out)

    assert result["backtracking_valid"] is False
    assert result["invalid_reason"] == "missing_georeferencing"
    assert result["geometry_unavailable"] is True
    assert result["drift_trajectory"] == []


def test_multiple_regions_only_primary_is_backtracked(tmp_path):
    """spec §5/§16: only the primary (largest-area) region is backtracked;
    other regions must remain untouched in the input, not silently dropped
    or merged — this just confirms Module 3 correctly SELECTS the primary
    one rather than e.g. the first one in the list."""
    summary = _write(tmp_path / "summary.json", {
        "scene_id": "MULTI_REGION_SCENE", "no_oil_detected": False,
        "geometry_unavailable": False, "num_regions": 2,
    })
    geometry = _write(tmp_path / "geometry.geojson", {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [80.40, 13.20]},
             "properties": {"region_id": 2, "centroid_lat": 13.20, "centroid_lon": 80.40,
                             "area_km2": 0.1, "acquisition_timestamp_utc": "2026-01-14T00:30:00Z"}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [80.32, 13.15]},
             "properties": {"region_id": 1, "centroid_lat": 13.15, "centroid_lon": 80.32,
                             "area_km2": 4.82, "acquisition_timestamp_utc": "2026-01-14T00:30:00Z"}},
        ],
    })
    out = tmp_path / "source_estimate.json"

    result = run_pipeline(
        summary, geometry, ROOT / "data_cache" / "era5" / "era5_demo.nc",
        ROOT / "data_cache" / "glorys" / "glorys_demo.nc",
        ROOT / "configs" / "config.yaml", out,
    )
    assert result["region_id"] == 1  # the larger-area region, not the first-listed one
    assert result["spill_centroid"] == [13.15, 80.32]


def test_malformed_summary_missing_field_raises_explicit_error(tmp_path):
    """spec §15: malformed input from Ishita (missing required field) ->
    explicit error naming the missing field, not a silent default."""
    summary = _write(tmp_path / "summary.json", {"scene_id": "X", "no_oil_detected": False})
    geometry = _write(tmp_path / "geometry.geojson", {"type": "FeatureCollection", "features": []})
    with pytest.raises(ValueError, match="geometry_unavailable"):
        run_pipeline(summary, geometry, config_path=ROOT / "configs" / "config.yaml",
                     output_path=tmp_path / "out.json")
