import logging
import pandas as pd

logger = logging.getLogger(__name__)


def filter_by_bbox(df, bbox, lat_col="latitude", lon_col="longitude"):
    """
    Keep only AIS records inside the calculated search bounding box.
    Records with NaN lat/lon (should already be filtered out by cleaning,
    but defended here too) are dropped, not silently mismatched.
    """
    original_count = len(df)

    filtered_df = df[
        (df[lat_col] >= bbox["min_latitude"])
        & (df[lat_col] <= bbox["max_latitude"])
        & (df[lon_col] >= bbox["min_longitude"])
        & (df[lon_col] <= bbox["max_longitude"])
    ].copy()

    removed = original_count - len(filtered_df)
    logger.info(f"Spatial filter: {original_count} → {len(filtered_df)} records "
                f"({removed} outside bbox)")

    return filtered_df


if __name__ == "__main__":
    from search_window import calculate_search_window, load_config
    from normalize_schema import normalize_ais_schema
    

    logging.basicConfig(level=logging.INFO)

    # Use a real source_estimate + config, not the data's own min/max —
    # otherwise this demo proves nothing.
    source_estimate = {
        "region_id": 1,
        "backtracking_valid": True,
        "no_oil_detected": False,
        "probable_source_region": {"type": "Point", "coordinates": [80.2707, 13.0827], "radius_km": None},
        "probable_source_time_window": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-01T04:00:00Z"
        }
    }
    config = load_config("config/config.yaml")
    window = calculate_search_window(source_estimate, config)

    ais_df = pd.read_csv("data/processed/normalized_ais.csv")

    filtered = filter_by_bbox(ais_df, window["bbox"])

    print("Spatial filtering completed.")
    print("Original records:", len(ais_df))
    print("Records inside search region:", len(filtered))