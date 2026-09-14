import math
from datetime import datetime, timezone

import pytest

from src.backward_advection import step_backward, backward_trajectory, combine_velocity

WGS84_EQUATORIAL_RADIUS_M = 6378137.0  # semi-major axis; exact for movement along the equator


def test_wind_current_vector_combination():
    """spec §28: verifies u/v components combine correctly without a
    sign/convention flip. ERA5 u10/v10 and GLORYS uo/vo are both physical
    velocity-vector components (direction things actually move TOWARD),
    so combination must be plain vector addition scaled by wind_drift_factor
    — never routed through a direction/bearing intermediate representation."""
    u, v = combine_velocity(u_current=0.2, v_current=0.1, u_wind=3.0, v_wind=1.0, wind_drift_factor=0.03)
    assert u == pytest.approx(0.2 + 0.03 * 3.0)
    assert v == pytest.approx(0.1 + 0.03 * 1.0)

    # Sign must be preserved: opposing wind should partially cancel current,
    # not add to it — a flipped sign convention would get this backwards.
    u_opposed, v_opposed = combine_velocity(u_current=0.2, v_current=0.1, u_wind=-3.0, v_wind=-1.0, wind_drift_factor=0.03)
    assert u_opposed < u
    assert v_opposed < v
    assert u_opposed == pytest.approx(0.2 - 0.03 * 3.0)


def test_advection_step_zero_velocity():
    assert step_backward(13.0, 80.0, 0.0, 0.0, 1) == (13.0, 80.0)


def test_advection_step_eastward_at_equator_matches_hand_computed_geodesic():
    """spec §28 test_advection_step: known velocity + timestep -> expected
    displacement, checked against a hand-computed GEODESIC offset (not
    naive degree math). At the equator, moving purely eastward stays on a
    circle whose radius is exactly the WGS84 semi-major axis, so the
    expected longitude shift can be hand-computed precisely without a
    geodesy library: dlon = distance_m / R_equator (radians).

    velocity is (u=1.0 m/s east, v=0.0), dt=1h -> distance = 3600 m.
    BACKWARD stepping from eastward velocity must move the point WEST.
    """
    lat, lon = step_backward(0.0, 0.0, 1.0, 0.0, 1)
    distance_m = 1.0 * 1 * 3600.0
    expected_delta_lon_deg = math.degrees(distance_m / WGS84_EQUATORIAL_RADIUS_M)

    assert lat == pytest.approx(0.0, abs=1e-6)
    assert lon == pytest.approx(-expected_delta_lon_deg, abs=1e-4)  # negative: stepped WEST, backward from eastward flow


def test_advection_step_northward_backward_moves_south():
    """Backward stepping from a purely northward velocity must move the
    point south, not north — this is exactly the class of sign/direction
    bug spec §38/§39 name as the most common failure mode."""
    lat, lon = step_backward(0.0, 0.0, 0.0, 1.0, 1)
    assert lat < 0.0
    assert abs(lon) < 1e-6
    # sanity-bound the magnitude against the same simple-sphere approximation
    # (meridional radius differs slightly from equatorial at higher latitudes,
    # but at the equator they're close enough for a loose bound, not an exact match)
    distance_m = 1.0 * 1 * 3600.0
    expected_delta_lat_deg = math.degrees(distance_m / WGS84_EQUATORIAL_RADIUS_M)
    assert abs(lat) == pytest.approx(expected_delta_lat_deg, rel=0.01)


def test_trajectory_records_steps():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out = backward_trajectory(13, 80, t, 2, 1, lambda *_: (1, 0))
    assert len(out) == 3


def test_trajectory_timestamps_step_strictly_backward():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    out = backward_trajectory(13, 80, t, 3, 1, lambda *_: (1, 0))
    timestamps = [p["timestamp"] for p in out]
    assert timestamps[0] == "2026-01-01T00:00:00Z"
    assert timestamps == sorted(timestamps, reverse=True)  # each step is earlier than the last

