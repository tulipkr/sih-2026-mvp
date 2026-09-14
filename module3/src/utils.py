import json
from datetime import datetime, timezone

REQUIRED_REGION_FIELDS = ["region_id", "centroid_lat", "centroid_lon", "area_km2", "acquisition_timestamp_utc"]
REQUIRED_SUMMARY_FIELDS = ["scene_id", "no_oil_detected", "geometry_unavailable", "num_regions"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_geojson(path):
    return load_json(path)


def parse_utc(value):
    if not isinstance(value, str):
        raise ValueError("acquisition_timestamp_utc must be a string")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("acquisition_timestamp_utc must include UTC timezone")
    return dt.astimezone(timezone.utc)


def validate_summary(summary):
    missing = [k for k in REQUIRED_SUMMARY_FIELDS if k not in summary]
    if missing:
        raise ValueError(f"spill_summary.json missing required fields: {missing}")
    if not isinstance(summary["no_oil_detected"], bool) or not isinstance(summary["geometry_unavailable"], bool):
        raise ValueError("scene flags must be boolean")
    if not isinstance(summary["num_regions"], int) or summary["num_regions"] < 0:
        raise ValueError("num_regions must be a non-negative int")


def validate_region(feature):
    props = feature.get("properties", {})
    missing = [k for k in REQUIRED_REGION_FIELDS if k not in props]
    if missing:
        raise ValueError(f"spill_geometry feature missing required fields: {missing}")
    lat, lon, area = float(props["centroid_lat"]), float(props["centroid_lon"]), float(props["area_km2"])
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("centroid coordinates out of range")
    if area <= 0:
        raise ValueError("area_km2 must be positive")
    parse_utc(props["acquisition_timestamp_utc"])
    return props


def choose_primary_region(features):
    if not features:
        raise ValueError("No spill regions found")
    return max(features, key=lambda f: float(f["properties"]["area_km2"]))
