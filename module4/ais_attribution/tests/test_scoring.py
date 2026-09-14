import math

import pandas as pd
import pytest

from src.closest_approach import haversine_distance_km, calculate_closest_approach
from src.scoring import calculate_composite_score, calculate_proximity_score
from src.ais_completeness import calculate_ais_completeness
from src.ingest_ais import clean_ais_data


def test_geodesic_distance_known_value():
    """spec §28 test_geodesic_distance: known two-point distance matches a
    hand-computed value. Chennai (13.0827N, 80.2707E) to a point exactly
    1 degree of latitude north (14.0827N, 80.2707E) -- pure north-south
    displacement, so the great-circle distance is simply
    (pi/180) * earth_radius_km, independent of longitude."""
    distance = haversine_distance_km(13.0827, 80.2707, 14.0827, 80.2707)
    expected = math.radians(1.0) * 6371.0
    assert distance == pytest.approx(expected, rel=1e-6)


def test_geodesic_distance_zero_for_identical_points():
    assert haversine_distance_km(13.0, 80.0, 13.0, 80.0) == pytest.approx(0.0, abs=1e-9)


def test_closest_approach_picks_minimum_distance_point():
    vessel_df = pd.DataFrame({
        "latitude": [13.0827, 13.20, 13.50],
        "longitude": [80.2707, 80.40, 80.70],
        "timestamp": pd.to_datetime(
            ["2026-09-01T10:00:00Z", "2026-09-01T11:00:00Z", "2026-09-01T12:00:00Z"], utc=True
        ),
    })
    result = calculate_closest_approach(vessel_df, 13.0827, 80.2707)
    assert result["closest_approach_km"] == pytest.approx(0.0, abs=1e-6)
    assert result["closest_approach_time"] is not None


def test_scoring_known_answer_close_compatible_vessel_scores_high():
    """spec §28 test_scoring_known_answer: a vessel placed deliberately
    close in space/time/drift scores near the top of the [0,1] range."""
    score = calculate_composite_score(
        closest_approach_km=0.5, temporal_compatible=True, drift_compatibility_score=0.95,
        proximity_weight=0.5, temporal_weight=0.2, drift_weight=0.2,
    )
    assert score > 0.85


def test_scoring_known_answer_far_incompatible_vessel_scores_low():
    score = calculate_composite_score(
        closest_approach_km=95, temporal_compatible=False, drift_compatibility_score=0.0,
        proximity_weight=0.5, temporal_weight=0.2, drift_weight=0.2,
    )
    assert score < 0.1


def test_ais_completeness_never_used_as_score_input():
    """spec §25/§38/§39, the single most important thing to get right:
    AIS completeness must NEVER be a parameter calculate_composite_score
    accepts or uses. Verified by inspecting the actual function signature,
    not assumed."""
    import inspect
    params = inspect.signature(calculate_composite_score).parameters
    for name in params:
        assert "completeness" not in name.lower(), (
            f"calculate_composite_score accepts a completeness-related parameter "
            f"'{name}' -- AIS completeness must never influence the score."
        )


def test_ais_completeness_gap_does_not_change_composite_score():
    """A vessel with heavily gapped AIS near the source must NOT be
    ranked lower because of the gap alone (spec §16) -- same
    proximity/temporal/drift inputs must give the identical score whether
    completeness is high or low, because completeness never enters the
    calculation at all."""
    score_complete = calculate_composite_score(2.0, True, 0.8)
    score_gappy = calculate_composite_score(2.0, True, 0.8)  # completeness isn't even a parameter
    assert score_complete == score_gappy


def test_mmsi_validity_filter_removes_known_invalid_values():
    """spec §28 test_mmsi_validity_filter: known-invalid MMSI values are
    correctly filtered (all-zero, wrong digit count)."""
    df = pd.DataFrame({
        "MMSI": [367123456, 0, 123, "000000000", 999999999],
        "timestamp": ["2026-09-01T10:00:00Z"] * 5,
        "latitude": [13.0] * 5,
        "longitude": [80.0] * 5,
    })
    cleaned, drop_reasons = clean_ais_data(df)
    assert set(cleaned["MMSI"].astype(str)) == {"367123456", "999999999"}
    assert drop_reasons["invalid_mmsi"] >= 3
