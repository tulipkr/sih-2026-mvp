import logging
import pandas as pd

logger = logging.getLogger(__name__)


def filter_by_time(df, time_window, timestamp_col="timestamp"):
    """
    Keep only AIS records inside the calculated time window.
    """

    original_count = len(df)

    start_time = pd.to_datetime(time_window["start"], utc=True)
    end_time = pd.to_datetime(time_window["end"], utc=True)

    timestamps = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")

    filtered_df = df[
        (timestamps >= start_time)
        & (timestamps <= end_time)
    ].copy()

    removed = original_count - len(filtered_df)

    logger.info(
        f"Temporal filter: {original_count} → {len(filtered_df)} records "
        f"({removed} outside time window)"
    )

    return filtered_df


if __name__ == "__main__":
    from search_window import calculate_search_window, load_config

    logging.basicConfig(level=logging.INFO)

    source_estimate = {
        "region_id": 1,
        "backtracking_valid": True,
        "no_oil_detected": False,
        "probable_source_region": {
            "type": "Point", "coordinates": [80.2707, 13.0827], "radius_km": None
        },
        "probable_source_time_window": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-01T04:00:00Z"
        }
    }

    config = load_config("config/config.yaml")
    window = calculate_search_window(source_estimate, config)

    ais_df = pd.read_csv("data/processed/normalized_ais.csv")

    filtered = filter_by_time(
        ais_df,
        window["time_window"]
    )

    print("Temporal filtering completed.")
    print("Original records:", len(ais_df))
    print("Records inside time window:", len(filtered))