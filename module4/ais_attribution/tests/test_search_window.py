from src.search_window import calculate_search_window


def make_config(**overrides):
    base = {
        "search_margin_km": 20,
        "search_time_margin_hours": 6,
    }
    base.update(overrides)
    return base


def test_search_window_bbox_known_values():
    source_estimate = {
        "region_id": 1,
        "backtracking_valid": True,
        "no_oil_detected": False,
        "probable_source_region": {
            "type": "Point", "coordinates": [80.2707, 13.0827], "radius_km": None
        },
        "probable_source_time_window": {
            "start": "2026-09-01T10:00:00Z",
            "end": "2026-09-01T14:00:00Z"
        }
    }

    config = make_config()
    result = calculate_search_window(source_estimate, config)

    expected_lat_margin = 20 / 111.0
    assert abs(result["bbox"]["min_latitude"] - (13.0827 - expected_lat_margin)) < 1e-4
    assert abs(result["bbox"]["max_latitude"] - (13.0827 + expected_lat_margin)) < 1e-4

    import math
    expected_lon_margin = 20 / (111.0 * math.cos(math.radians(13.0827)))
    assert abs(result["bbox"]["min_longitude"] - (80.2707 - expected_lon_margin)) < 1e-4
    assert abs(result["bbox"]["max_longitude"] - (80.2707 + expected_lon_margin)) < 1e-4


def test_search_window_time_margins_both_edges():
    """
    This is the key regression test: the margin must be applied to the
    WINDOW'S start and end independently, not to a single midpoint timestamp.
    """
    source_estimate = {
        "region_id": 1,
        "backtracking_valid": True,
        "no_oil_detected": False,
        "probable_source_region": {"type": "Point", "coordinates": [80.2707, 13.0827], "radius_km": None},
        "probable_source_time_window": {
            "start": "2026-09-01T10:00:00Z",
            "end": "2026-09-01T14:00:00Z"
        }
    }

    config = make_config(search_time_margin_hours=6)
    result = calculate_search_window(source_estimate, config)

    assert result["time_window"]["start"] == "2026-09-01T04:00:00+00:00"
    assert result["time_window"]["end"] == "2026-09-01T20:00:00+00:00"


def test_search_window_returns_expected_keys():
    source_estimate = {
        "region_id": 1,
        "backtracking_valid": True,
        "no_oil_detected": False,
        "probable_source_region": {"type": "Point", "coordinates": [80.2707, 13.0827], "radius_km": None},
        "probable_source_time_window": {
            "start": "2026-09-01T10:00:00Z",
            "end": "2026-09-01T14:00:00Z"
        }
    }
    result = calculate_search_window(source_estimate, make_config())
    assert set(result.keys()) == {"bbox", "time_window"}
    assert set(result["bbox"].keys()) == {"min_latitude", "max_latitude", "min_longitude", "max_longitude"}


if __name__ == "__main__":
    test_search_window_bbox_known_values()
    test_search_window_time_margins_both_edges()
    test_search_window_returns_expected_keys()
    print("Search window tests passed!")