import pytest
from src.utils import validate_source_estimate


def make_valid_estimate(**overrides):
    """Base valid estimate; overrides let each test tweak one field."""
    base = {
        "region_id": 1,
        "backtracking_valid": True,
        "no_oil_detected": False,
        "probable_source_region": {
            "type": "Point",
            "coordinates": [80.2707, 13.0827],
            "radius_km": None
        },
        "probable_source_time_window": {
            "start": "2026-09-01T10:00:00Z",
            "end": "2026-09-01T14:00:00Z"
        }
    }
    base.update(overrides)
    return base


def test_valid_source_estimate():
    result = validate_source_estimate(make_valid_estimate())
    assert result["valid"] is True
    assert result["short_circuit"] is False
    assert result["reason"] is None


def test_missing_required_field_raises():
    estimate = make_valid_estimate()
    del estimate["probable_source_region"]
    with pytest.raises(ValueError, match="Missing required field"):
        validate_source_estimate(estimate)


def test_invalid_latitude_raises():
    estimate = make_valid_estimate(
        probable_source_region={"latitude": 999, "longitude": 80.27}
    )
    with pytest.raises(ValueError, match="Invalid latitude"):
        validate_source_estimate(estimate)


def test_invalid_longitude_raises():
    estimate = make_valid_estimate(
        probable_source_region={"latitude": 13.08, "longitude": -999}
    )
    with pytest.raises(ValueError, match="Invalid longitude"):
        validate_source_estimate(estimate)


def test_time_window_start_after_end_raises():
    estimate = make_valid_estimate(
        probable_source_time_window={
            "start": "2026-09-01T14:00:00Z",
            "end": "2026-09-01T10:00:00Z"
        }
    )
    with pytest.raises(ValueError, match="start must be before end"):
        validate_source_estimate(estimate)


def test_short_circuit_on_invalid_backtracking():
    estimate = make_valid_estimate(backtracking_valid=False)
    result = validate_source_estimate(estimate)
    assert result["short_circuit"] is True
    assert "backtracking_valid" in result["reason"]


def test_short_circuit_on_no_oil_detected():
    estimate = make_valid_estimate(no_oil_detected=True)
    result = validate_source_estimate(estimate)
    assert result["short_circuit"] is True
    assert "no_oil_detected" in result["reason"]


def test_drift_trajectory_is_optional():
    estimate = make_valid_estimate()
    assert "drift_trajectory" not in estimate
    result = validate_source_estimate(estimate)
    assert result["valid"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])