import json
from pathlib import Path

import pandas as pd
import pytest

from src.ranking_pipeline import run_pipeline
from src.synthetic_ais_generator import generate_synthetic_ais

MODULE_ROOT = Path(__file__).resolve().parent.parent


def _write_source_estimate(path, **overrides):
    base = {
        "scene_id": "TEST_SCENE", "region_id": 1,
        "backtracking_valid": True, "no_oil_detected": False,
        "probable_source_region": {
            "type": "Point", "coordinates": [80.2707, 13.0827], "radius_km": 5.0
        },
        "probable_source_time_window": {
            "start": "2026-09-01T10:00:00Z", "end": "2026-09-01T14:00:00Z"
        },
        "drift_trajectory": [
            {"timestamp": "2026-09-01T14:00:00Z", "lat": 13.20, "lon": 80.40},
            {"timestamp": "2026-09-01T10:00:00Z", "lat": 13.0827, "lon": 80.2707},
        ],
    }
    base.update(overrides)
    path.write_text(json.dumps(base))
    return base


REQUIRED_TOP_LEVEL_KEYS = {
    "scene_id", "region_id", "ais_data_source", "search_bbox", "search_time_window",
    "total_vessels_in_window", "ais_coverage_note", "ranked_candidates", "disclaimer",
    "backtracking_valid", "no_oil_detected",
}
REQUIRED_CANDIDATE_KEYS = {
    "rank", "mmsi", "vessel_name", "vessel_type", "closest_approach_km",
    "closest_approach_time_utc", "temporal_compatible", "trajectory_drift_compatibility_score",
    "ais_completeness_score", "composite_score", "score_type", "evidence_summary",
}


def _assert_schema_valid(output_data):
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(output_data.keys())
    assert output_data["disclaimer"]  # hardcoded, tested part of every output (spec §9/§40)
    for candidate in output_data["ranked_candidates"]:
        assert REQUIRED_CANDIDATE_KEYS.issubset(candidate.keys())
        assert candidate["score_type"] == "heuristic_composite_score"
        assert isinstance(candidate["mmsi"], str)


# --- 1. Normal successful analysis --------------------------------------
def test_full_pipeline_synthetic_seeded_vessel_ranks_near_top(tmp_path):
    """spec §29 test_full_pipeline_synthetic: synthetic AIS + synthetic
    source_estimate.json -> schema-valid ranked output, seeded compatible
    vessel near the top."""
    source_estimate_path = tmp_path / "source_estimate.json"
    _write_source_estimate(
        source_estimate_path,
        probable_source_time_window={"start": "2026-09-01T10:00:00Z", "end": "2026-09-01T13:00:00Z"},
    )
    ais_path = tmp_path / "synthetic_ais.csv"
    _, seeded_mmsi = generate_synthetic_ais(
        output_path=str(ais_path), source_lat=13.0827, source_lon=80.2707,
        start_time="2026-09-01T10:00:00Z", drift_direction_degrees=45, num_vessels=15,
    )

    ranked, output_data = run_pipeline(
        source_estimate_path=source_estimate_path, ais_input_path=ais_path,
        ais_source_format="synthetic", config_path=MODULE_ROOT / "config" / "config.yaml",
        output_dir=tmp_path / "out",
    )

    _assert_schema_valid(output_data)
    assert output_data["ais_data_source"] == "synthetic"
    assert len(output_data["ranked_candidates"]) > 0
    top_mmsis = [c["mmsi"] for c in output_data["ranked_candidates"][:3]]
    assert seeded_mmsi in top_mmsis
    assert (tmp_path / "out" / "normalized_ais_tracks.geojson").exists()


# --- 2. MarineCadastre real-format smoke test ---------------------------
def test_full_pipeline_marinecadastre_sample_runs_without_schema_errors(tmp_path):
    """spec §29 test_full_pipeline_marinecadastre_sample: runs without
    schema errors against real-format MarineCadastre data. No 'right
    answer' expected (this fixture is hand-built, not a real incident) --
    only that ingestion mechanics work end-to-end unmodified."""
    source_estimate_path = tmp_path / "source_estimate.json"
    _write_source_estimate(
        source_estimate_path,
        probable_source_region={"type": "Point", "coordinates": [80.28, 13.10], "radius_km": 10.0},
        probable_source_time_window={"start": "2026-09-01T09:00:00Z", "end": "2026-09-01T16:00:00Z"},
    )
    mc_sample = MODULE_ROOT / "data" / "marinecadastre_sample" / "sample.csv"

    ranked, output_data = run_pipeline(
        source_estimate_path=source_estimate_path, ais_input_path=mc_sample,
        config_path=MODULE_ROOT / "config" / "config.yaml", output_dir=tmp_path / "out",
    )
    _assert_schema_valid(output_data)
    assert output_data["ais_data_source"] == "marinecadastre"


def test_same_pipeline_code_handles_both_sources_unmodified(tmp_path):
    """spec §16: incident/data source swapped later -> pipeline code must
    not need to change, only the input file. Runs the literal same
    run_pipeline() against both sources in one test to verify this."""
    source_estimate_path = tmp_path / "source_estimate.json"
    _write_source_estimate(source_estimate_path)

    synthetic_path = tmp_path / "synthetic.csv"
    generate_synthetic_ais(str(synthetic_path), 13.0827, 80.2707, "2026-09-01T10:00:00Z")
    mc_sample = MODULE_ROOT / "data" / "marinecadastre_sample" / "sample.csv"

    for ais_path in (synthetic_path, mc_sample):
        ranked, output_data = run_pipeline(
            source_estimate_path=source_estimate_path, ais_input_path=ais_path,
            config_path=MODULE_ROOT / "config" / "config.yaml", output_dir=tmp_path / f"out_{ais_path.stem}",
        )
        _assert_schema_valid(output_data)  # same assertions hold for both, same code path


# --- 3. Short-circuit ----------------------------------------------------
def test_short_circuit_invalid_backtracking(tmp_path):
    source_estimate_path = tmp_path / "source_estimate.json"
    _write_source_estimate(source_estimate_path, backtracking_valid=False)

    ranked, output_data = run_pipeline(
        source_estimate_path=source_estimate_path,
        config_path=MODULE_ROOT / "config" / "config.yaml", output_dir=tmp_path / "out",
    )
    assert ranked.empty
    assert output_data["ranked_candidates"] == []
    assert output_data["backtracking_valid"] is False
    assert "backtracking_valid" in output_data["ais_coverage_note"]
    assert (tmp_path / "out" / "candidate_ranking.json").exists()  # output IS written, not just an empty return


def test_short_circuit_no_oil_detected(tmp_path):
    source_estimate_path = tmp_path / "source_estimate.json"
    _write_source_estimate(source_estimate_path, no_oil_detected=True)

    ranked, output_data = run_pipeline(
        source_estimate_path=source_estimate_path,
        config_path=MODULE_ROOT / "config" / "config.yaml", output_dir=tmp_path / "out",
    )
    assert ranked.empty
    assert output_data["no_oil_detected"] is True
    assert (tmp_path / "out" / "candidate_ranking.json").exists()


# --- 4. Zero candidates / no vessels found -------------------------------
def test_zero_candidates_when_no_vessels_in_window(tmp_path):
    """spec §15/§16: empty search window -> valid empty-candidate output,
    not an error."""
    source_estimate_path = tmp_path / "source_estimate.json"
    _write_source_estimate(
        source_estimate_path,
        probable_source_region={"type": "Point", "coordinates": [10.0, 60.0], "radius_km": 1.0},  # nowhere near any AIS data
    )
    ais_path = tmp_path / "synthetic.csv"
    generate_synthetic_ais(str(ais_path), 13.0827, 80.2707, "2026-09-01T10:00:00Z")

    ranked, output_data = run_pipeline(
        source_estimate_path=source_estimate_path, ais_input_path=ais_path,
        config_path=MODULE_ROOT / "config" / "config.yaml", output_dir=tmp_path / "out",
    )
    _assert_schema_valid(output_data)
    assert output_data["ranked_candidates"] == []
    assert output_data["total_vessels_in_window"] == 0
    assert "AIS-dark" in output_data["ais_coverage_note"] or "dark" in output_data["ais_coverage_note"]
    geojson = json.loads((tmp_path / "out" / "normalized_ais_tracks.geojson").read_text())
    assert geojson["features"] == []


# --- 5. Vessel outside radius / outside time window ----------------------
def test_vessel_outside_search_radius_is_excluded(tmp_path):
    source_estimate_path = tmp_path / "source_estimate.json"
    _write_source_estimate(
        source_estimate_path,
        probable_source_region={"type": "Point", "coordinates": [80.2707, 13.0827], "radius_km": 1.0},
    )
    ais_path = tmp_path / "ais.csv"
    pd.DataFrame({
        "MMSI": [111111111], "vessel_name": ["FAR VESSEL"],
        "timestamp": ["2026-09-01T11:00:00Z"],
        "latitude": [20.0], "longitude": [85.0],  # hundreds of km away
        "SOG": [10.0], "COG": [90.0], "heading": [90.0],
    }).to_csv(ais_path, index=False)

    ranked, output_data = run_pipeline(
        source_estimate_path=source_estimate_path, ais_input_path=ais_path,
        config_path=MODULE_ROOT / "config" / "config.yaml", output_dir=tmp_path / "out",
    )
    assert output_data["ranked_candidates"] == []
    assert output_data["total_vessels_in_window"] == 0


def test_vessel_outside_time_window_is_excluded(tmp_path):
    source_estimate_path = tmp_path / "source_estimate.json"
    _write_source_estimate(
        source_estimate_path,
        probable_source_time_window={"start": "2026-09-01T10:00:00Z", "end": "2026-09-01T11:00:00Z"},
    )
    ais_path = tmp_path / "ais.csv"
    pd.DataFrame({
        "MMSI": [222222222], "vessel_name": ["WRONG TIME VESSEL"],
        "timestamp": ["2026-01-01T00:00:00Z"],  # months away, well beyond any margin
        "latitude": [13.0827], "longitude": [80.2707],
        "SOG": [10.0], "COG": [90.0], "heading": [90.0],
    }).to_csv(ais_path, index=False)

    ranked, output_data = run_pipeline(
        source_estimate_path=source_estimate_path, ais_input_path=ais_path,
        config_path=MODULE_ROOT / "config" / "config.yaml", output_dir=tmp_path / "out",
    )
    assert output_data["ranked_candidates"] == []


# --- 6. Malformed / missing AIS data --------------------------------------
def test_malformed_ais_rows_are_skipped_not_fatal(tmp_path):
    """spec §11: malformed individual records are logged/skipped, never
    fail the entire ingestion."""
    source_estimate_path = tmp_path / "source_estimate.json"
    _write_source_estimate(source_estimate_path)
    ais_path = tmp_path / "ais.csv"
    pd.DataFrame({
        "MMSI": [367123456, "not_a_valid_mmsi", 367123457],
        "vessel_name": ["Good1", "Bad", "Good2"],
        "timestamp": ["2026-09-01T11:00:00Z", "not-a-timestamp", "2026-09-01T11:05:00Z"],
        "latitude": [13.09, 999.0, 13.08],  # row 2 has an out-of-range latitude too
        "longitude": [80.27, 80.27, 80.27],
        "SOG": [9.0, 9.0, 9.0], "COG": [45.0, 45.0, 45.0], "heading": [45.0, 45.0, 45.0],
    }).to_csv(ais_path, index=False)

    ranked, output_data = run_pipeline(
        source_estimate_path=source_estimate_path, ais_input_path=ais_path,
        config_path=MODULE_ROOT / "config" / "config.yaml", output_dir=tmp_path / "out",
    )
    _assert_schema_valid(output_data)  # doesn't crash; malformed row just isn't ranked
    mmsis = [c["mmsi"] for c in output_data["ranked_candidates"]]
    assert "367123456" in mmsis
    assert "367123457" in mmsis


def test_missing_ais_file_raises_explicit_error(tmp_path):
    source_estimate_path = tmp_path / "source_estimate.json"
    _write_source_estimate(source_estimate_path)
    with pytest.raises(FileNotFoundError, match="AIS source file not found"):
        run_pipeline(
            source_estimate_path=source_estimate_path, ais_input_path=tmp_path / "does_not_exist.csv",
            config_path=MODULE_ROOT / "config" / "config.yaml", output_dir=tmp_path / "out",
        )


# --- 7. Invalid upstream input ---------------------------------------------
def test_invalid_source_estimate_missing_field_raises(tmp_path):
    source_estimate_path = tmp_path / "source_estimate.json"
    source_estimate_path.write_text(json.dumps({"scene_id": "X", "backtracking_valid": True}))
    with pytest.raises(ValueError, match="Missing required field"):
        run_pipeline(source_estimate_path=source_estimate_path,
                     config_path=MODULE_ROOT / "config" / "config.yaml", output_dir=tmp_path / "out")


# --- 8. Coordinate handling (GeoJSON Point vs flat, real schema) ---------
def test_geojson_point_region_is_correctly_parsed(tmp_path):
    """Confirms the fix for the critical schema bug: Shazmeen's real
    probable_source_region is a GeoJSON Point {"coordinates": [lon, lat]},
    not a flat {"latitude", "longitude"} dict."""
    source_estimate_path = tmp_path / "source_estimate.json"
    _write_source_estimate(
        source_estimate_path,
        probable_source_region={"type": "Point", "coordinates": [80.2707, 13.0827], "radius_km": 2.0},
    )
    ais_path = tmp_path / "ais.csv"
    pd.DataFrame({
        "MMSI": [367123456], "vessel_name": ["ON TARGET"],
        "timestamp": ["2026-09-01T11:00:00Z"],
        "latitude": [13.0827], "longitude": [80.2707],  # exactly at the (correctly parsed) region point
        "SOG": [9.0], "COG": [45.0], "heading": [45.0],
    }).to_csv(ais_path, index=False)

    ranked, output_data = run_pipeline(
        source_estimate_path=source_estimate_path, ais_input_path=ais_path,
        config_path=MODULE_ROOT / "config" / "config.yaml", output_dir=tmp_path / "out",
    )
    assert len(output_data["ranked_candidates"]) == 1
    assert output_data["ranked_candidates"][0]["closest_approach_km"] == pytest.approx(0.0, abs=1e-3)
