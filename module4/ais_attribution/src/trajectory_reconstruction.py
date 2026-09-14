import logging
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_ais_data(
    df,
    mmsi_col="MMSI",
    timestamp_col="timestamp",
    expected_interval_minutes=6,
    gap_multiplier=3
):
    """
    Preprocess cleaned AIS data.

    - Sorts records by vessel and timestamp
    - Flags large time gaps as AIS-dark periods
    - Does NOT remove or penalize vessels because of gaps

    Duplicate removal is handled by clean_ais_data in Part 9.
    """

    df = df.copy()

    gap_threshold_minutes = expected_interval_minutes * gap_multiplier

    # Make sure timestamp is datetime
    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col],
        utc=True,
        errors="coerce"
    )

    # Sort records by vessel and timestamp
    df = df.sort_values(
        by=[mmsi_col, timestamp_col]
    ).reset_index(drop=True)

    # Calculate time difference between consecutive reports
    # for each vessel
    df["time_gap_minutes"] = (
        df.groupby(mmsi_col)[timestamp_col]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    # Flag large gaps
    # IMPORTANT: informational only
    df["ais_gap_flag"] = (
        df["time_gap_minutes"] > gap_threshold_minutes
    )

    num_gaps = int(df["ais_gap_flag"].sum())

    num_vessels_with_gaps = (
        df.loc[df["ais_gap_flag"], mmsi_col].nunique()
    )

    logger.info(
        f"AIS gap detection: threshold={gap_threshold_minutes}min "
        f"({expected_interval_minutes}min interval x{gap_multiplier}), "
        f"{num_gaps} gap-flagged records across "
        f"{num_vessels_with_gaps} vessels"
    )

    return df


def reconstruct_trajectories(
    df,
    mmsi_col="MMSI",
    timestamp_col="timestamp"
):
    """
    Reconstruct one chronological trajectory for each vessel.

    Each MMSI represents one vessel.

    Returns:
        Dictionary where:
        key   = MMSI
        value = AIS records for that vessel,
                sorted chronologically
    """

    trajectories = {}

    for mmsi, vessel_df in df.groupby(mmsi_col):

        # Sort each vessel's records chronologically
        vessel_df = vessel_df.sort_values(
            timestamp_col
        ).reset_index(drop=True)

        trajectories[mmsi] = vessel_df

    logger.info(
        f"Trajectory reconstruction: {len(df)} records → "
        f"{len(trajectories)} vessel trajectories"
    )

    return trajectories


if __name__ == "__main__":

    from ingest_ais import clean_ais_data

    logging.basicConfig(level=logging.INFO)

    # ---------------------------------------------------------
    # 1. Load raw AIS data
    # ---------------------------------------------------------

    input_path = "data/synthetic/synthetic_ais.csv"

    df = pd.read_csv(input_path)

    # Convert timestamp to UTC datetime
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
    # 4. Reconstruct vessel trajectories
    # ---------------------------------------------------------

    trajectories = reconstruct_trajectories(
        processed_df
    )

    # ---------------------------------------------------------
    # 5. Display results
    # ---------------------------------------------------------

    print("\nPreprocessed AIS data:")
    print(processed_df.head())

    print(
        f"\nAIS gap records: "
        f"{processed_df['ais_gap_flag'].sum()}"
    )

    print(
        f"Total AIS records: "
        f"{len(processed_df)}"
    )

    print(
        f"Unique vessel trajectories: "
        f"{len(trajectories)}"
    )

    print("\nTrajectory summary:")

    for mmsi, vessel_df in trajectories.items():

        print(
            f"MMSI {mmsi}: "
            f"{len(vessel_df)} records | "
            f"{vessel_df['timestamp'].min()} → "
            f"{vessel_df['timestamp'].max()}"
        )