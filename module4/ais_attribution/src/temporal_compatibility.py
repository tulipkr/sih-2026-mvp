import logging
import pandas as pd

logger = logging.getLogger(__name__)


def check_temporal_compatibility(
    vessel_df,
    source_time_window,
    margin_hours=6,
    timestamp_col="timestamp"
):
    """
    Check whether a vessel has an AIS position within
    the probable source time window plus the configured
    time margin.

    Returns:
        Dictionary containing:
        - temporal_compatible
        - closest_approach_time
        - time_difference_hours
    """

    if vessel_df.empty:
        return {
            "temporal_compatible": False,
            "closest_approach_time": None,
            "time_difference_hours": None
        }

    df = vessel_df.copy()

    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(subset=[timestamp_col])

    if df.empty:
        return {
            "temporal_compatible": False,
            "closest_approach_time": None,
            "time_difference_hours": None
        }

    source_start = pd.to_datetime(
        source_time_window["start"],
        utc=True
    )

    source_end = pd.to_datetime(
        source_time_window["end"],
        utc=True
    )

    # Apply configured time margin
    search_start = source_start - pd.Timedelta(
        hours=margin_hours
    )

    search_end = source_end + pd.Timedelta(
        hours=margin_hours
    )

    # Find vessel records inside the allowed time window
    compatible_records = df[
        (df[timestamp_col] >= search_start)
        & (df[timestamp_col] <= search_end)
    ].copy()

    if compatible_records.empty:
        logger.info(
            "Temporal compatibility: vessel has no "
            "AIS position inside the allowed time window"
        )

        return {
            "temporal_compatible": False,
            "closest_approach_time": None,
            "time_difference_hours": None
        }

    # Find the vessel report closest to the source
    # time window itself
    def distance_from_source_window(timestamp):
        if timestamp < source_start:
            return (
                source_start - timestamp
            ).total_seconds() / 3600

        if timestamp > source_end:
            return (
                timestamp - source_end
            ).total_seconds() / 3600

        return 0.0

    compatible_records["time_difference_hours"] = (
        compatible_records[timestamp_col]
        .apply(distance_from_source_window)
    )

    closest_record = compatible_records.loc[
        compatible_records["time_difference_hours"].idxmin()
    ]

    result = {
        "temporal_compatible": True,
        "closest_approach_time": (
            closest_record[timestamp_col].isoformat()
        ),
        "time_difference_hours": float(
            closest_record["time_difference_hours"]
        )
    }

    return result


if __name__ == "__main__":

    from trajectory_reconstruction import (
        preprocess_ais_data,
        reconstruct_trajectories
    )
    from ingest_ais import clean_ais_data

    logging.basicConfig(level=logging.INFO)

    # ---------------------------------------------------------
    # 1. Load AIS data
    # ---------------------------------------------------------

    input_path = "data/synthetic/synthetic_ais.csv"

    df = pd.read_csv(input_path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    # ---------------------------------------------------------
    # 2. Clean AIS data
    # ---------------------------------------------------------

    cleaned_df, _ = clean_ais_data(df)

    # ---------------------------------------------------------
    # 3. Preprocess AIS data
    # ---------------------------------------------------------

    processed_df = preprocess_ais_data(
        cleaned_df,
        expected_interval_minutes=6
    )

    # ---------------------------------------------------------
    # 4. Reconstruct trajectories
    # ---------------------------------------------------------

    trajectories = reconstruct_trajectories(
        processed_df
    )

    # ---------------------------------------------------------
    # 5. Source time window
    # ---------------------------------------------------------

    source_time_window = {
        "start": "2025-01-01T00:00:00Z",
        "end": "2025-01-01T04:00:00Z"
    }

    # ---------------------------------------------------------
    # 6. Check temporal compatibility
    # ---------------------------------------------------------

    print("\nTemporal compatibility results:")

    for mmsi, vessel_df in trajectories.items():

        result = check_temporal_compatibility(
            vessel_df,
            source_time_window,
            margin_hours=6
        )

        print(
            f"MMSI {mmsi}: "
            f"compatible={result['temporal_compatible']} | "
            f"time_difference="
            f"{result['time_difference_hours']} hours"
        )