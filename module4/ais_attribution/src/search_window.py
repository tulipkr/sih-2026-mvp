from datetime import datetime, timedelta
from pathlib import Path
import math
import yaml

from .utils import extract_region_lat_lon, extract_region_radius_km

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(config_path=None):
    """config_path defaults to this module's own config/config.yaml,
    resolved relative to this file's location (not the current working
    directory) -- so this works regardless of where the caller's process
    happens to be run from."""
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def calculate_search_window(source_estimate, config):
    region = source_estimate["probable_source_region"]
    latitude, longitude = extract_region_lat_lon(region)

    # Total search buffer = the configured fixed margin PLUS Shazmeen's own
    # positional uncertainty radius for this region, when she supplied one
    # -- a fixed margin alone would ignore how uncertain the estimate
    # actually is.
    margin_km = config["search_margin_km"] + (extract_region_radius_km(region) or 0.0)
    time_margin_hours = config["search_time_margin_hours"]

    lat_margin = margin_km / 111.0
    lon_margin = margin_km / (111.0 * math.cos(math.radians(latitude)))

    min_latitude = latitude - lat_margin
    max_latitude = latitude + lat_margin
    min_longitude = longitude - lon_margin
    max_longitude = longitude + lon_margin

    time_window = source_estimate["probable_source_time_window"]
    window_start = datetime.fromisoformat(time_window["start"].replace("Z", "+00:00"))
    window_end = datetime.fromisoformat(time_window["end"].replace("Z", "+00:00"))

    start_time = window_start - timedelta(hours=time_margin_hours)
    end_time = window_end + timedelta(hours=time_margin_hours)

    return {
        "bbox": {
            "min_latitude": min_latitude,
            "max_latitude": max_latitude,
            "min_longitude": min_longitude,
            "max_longitude": max_longitude
        },
        "time_window": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        }
    }
