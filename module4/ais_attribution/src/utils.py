from datetime import datetime


def extract_region_lat_lon(region):
    """
    Extract (lat, lon) from Shazmeen's probable_source_region.

    Real schema (Module 3's actual output, confirmed against her code):
        {"type": "Point", "coordinates": [lon, lat], "radius_km": float}
    (GeoJSON convention: [longitude, latitude] order.)

    Also defensively accepts a flat {"latitude": ..., "longitude": ...}
    shape, in case an earlier/different upstream format is ever supplied —
    this is a fallback, not the expected primary shape.
    """
    if not isinstance(region, dict):
        raise ValueError(f"probable_source_region must be an object, got {type(region).__name__}")

    if "coordinates" in region:
        coordinates = region["coordinates"]
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
            raise ValueError(
                f"probable_source_region.coordinates must be [lon, lat], got {coordinates!r}"
            )
        lon, lat = coordinates[0], coordinates[1]
    elif "latitude" in region and "longitude" in region:
        lat, lon = region["latitude"], region["longitude"]
    else:
        raise ValueError(
            "probable_source_region has neither GeoJSON 'coordinates' [lon, lat] "
            f"nor flat 'latitude'/'longitude' keys. Got keys: {sorted(region.keys())}"
        )

    if lat is None or lon is None:
        raise ValueError("probable_source_region missing latitude/longitude value")
    lat, lon = float(lat), float(lon)
    if not -90 <= lat <= 90:
        raise ValueError(f"Invalid latitude: {lat}")
    if not -180 <= lon <= 180:
        raise ValueError(f"Invalid longitude: {lon}")
    return lat, lon


def extract_region_radius_km(region):
    """radius_km is optional (Module 3's fixed_radius_heuristic and
    perturbation_ensemble uncertainty methods both populate it, but don't
    assume it's always present)."""
    if isinstance(region, dict) and region.get("radius_km") is not None:
        return float(region["radius_km"])
    return None


def extract_drift_direction_degrees(source_estimate):
    """
    Derive a single representative drift bearing (degrees, 0-360, compass
    convention) from Shazmeen's real drift_trajectory — a LIST of
    {"timestamp", "lat", "lon"} points, not a scalar direction field.
    Returns None if drift_trajectory is missing/empty/too short to derive
    a bearing from (spec §16: must degrade gracefully to null, not error).
    """
    import math

    trajectory = source_estimate.get("drift_trajectory")
    if not trajectory or not isinstance(trajectory, list) or len(trajectory) < 2:
        return None

    try:
        # Shazmeen's drift_trajectory is a BACKWARD trajectory: point [0] is
        # the latest position (at/near acquisition time), point [-1] is the
        # earliest (closest to the probable source), per her actual
        # backward_trajectory() implementation. The physical drift
        # direction -- the direction water/oil was actually moving forward
        # in time, which is what a compatible vessel's COG should roughly
        # align with -- runs from the EARLIEST point to the LATEST one,
        # i.e. the reverse of the array's own order.
        earliest, latest = trajectory[-1], trajectory[0]
        lat1, lon1 = float(earliest["lat"]), float(earliest["lon"])
        lat2, lon2 = float(latest["lat"]), float(latest["lon"])
    except (KeyError, TypeError, ValueError):
        return None

    if lat1 == lat2 and lon1 == lon2:
        return None  # no net displacement -> no meaningful direction

    # Initial bearing (compass degrees, 0=N, 90=E) from the first to the
    # last drift point -- the overall direction of the estimated drift.
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    delta_lon_r = math.radians(lon2 - lon1)
    x = math.sin(delta_lon_r) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(delta_lon_r)
    bearing = (math.degrees(math.atan2(x, y)) + 360) % 360
    return float(bearing)


def validate_source_estimate(source_estimate):
    """
    Validates Shazmeen's source_estimate.json against the agreed schema.
    Returns a dict: {"valid": bool, "short_circuit": bool, "reason": str or None}
    Raises ValueError only for structurally malformed input (missing/bad fields).
    """
    required_fields = [
        "region_id",
        "backtracking_valid",
        "no_oil_detected",
        "probable_source_region",
        "probable_source_time_window"
    ]
    for field in required_fields:
        if field not in source_estimate:
            raise ValueError(f"Missing required field: {field}")

    # Validates shape/ranges via extract_region_lat_lon; short-circuit
    # inputs may legitimately carry a placeholder/degenerate region, so
    # only enforce this when we're not about to short-circuit anyway.
    will_short_circuit = (
        source_estimate["backtracking_valid"] is False
        or source_estimate["no_oil_detected"] is True
    )
    if not will_short_circuit:
        extract_region_lat_lon(source_estimate["probable_source_region"])

    time_window = source_estimate["probable_source_time_window"]
    start_str = time_window.get("start")
    end_str = time_window.get("end")

    if not start_str or not end_str:
        raise ValueError("probable_source_time_window missing start/end")

    try:
        start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("probable_source_time_window has invalid ISO8601 timestamps")

    if start >= end:
        raise ValueError("probable_source_time_window: start must be before end")

    if source_estimate["backtracking_valid"] is False:
        return {"valid": True, "short_circuit": True, "reason": "backtracking_valid is false"}

    if source_estimate["no_oil_detected"] is True:
        return {"valid": True, "short_circuit": True, "reason": "no_oil_detected is true"}

    return {"valid": True, "short_circuit": False, "reason": None}
