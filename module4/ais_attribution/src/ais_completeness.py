import logging
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_ais_completeness(
    vessel_df,
    expected_interval_minutes=6,
    timestamp_col="timestamp"
):
    """
    Calculate AIS completeness for one vessel.

    Completeness = actual number of reports /
                   expected number of reports

    This is contextual information only.
    It must NOT be used as a guilt penalty or
    directly reduce the composite score.
    """

    if vessel_df.empty:
        return 0.0

    timestamps = pd.to_datetime(
        vessel_df[timestamp_col],
        utc=True,
        errors="coerce"
    ).dropna()

    if len(timestamps) <= 1:
        return 1.0

    start_time = timestamps.min()
    end_time = timestamps.max()

    duration_minutes = (
        end_time - start_time
    ).total_seconds() / 60

    expected_reports = (
        duration_minutes / expected_interval_minutes
    ) + 1

    actual_reports = len(timestamps)

    completeness = (
        actual_reports / expected_reports
    )

    completeness = min(completeness, 1.0)

    return round(completeness, 3)


def calculate_completeness_for_trajectories(
    trajectories,
    expected_interval_minutes=6
):
    """
    Calculate AIS completeness for every vessel trajectory.

    Returns:
        Dictionary:
        MMSI -> completeness score
    """

    results = {}

    for mmsi, vessel_df in trajectories.items():

        score = calculate_ais_completeness(
            vessel_df,
            expected_interval_minutes
        )

        results[mmsi] = score

    logger.info(
        f"AIS completeness calculated for "
        f"{len(results)} vessels"
    )

    return results


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    from trajectory_reconstruction import (
        preprocess_ais_data,
        reconstruct_trajectories
    )
    from ingest_ais import clean_ais_data

    input_path = "data/synthetic/synthetic_ais.csv"

    df = pd.read_csv(input_path)

    cleaned_df, _ = clean_ais_data(df)

    processed_df = preprocess_ais_data(
        cleaned_df,
        expected_interval_minutes=6
    )

    trajectories = reconstruct_trajectories(
        processed_df
    )

    completeness_results = (
        calculate_completeness_for_trajectories(
            trajectories,
            expected_interval_minutes=6
        )
    )

    print("\nAIS completeness results:")

    for mmsi, score in completeness_results.items():
        print(
            f"MMSI {mmsi}: "
            f"completeness={score:.3f}"
        )