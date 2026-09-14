import logging
import math
import pandas as pd

logger = logging.getLogger(__name__)


def circular_angle_difference(angle1, angle2):
    """
    Calculate the smallest difference between two directions.

    Directions are measured in degrees from 0 to 360.
    """
    difference = abs(angle1 - angle2)
    return min(difference, 360 - difference)


def calculate_drift_compatibility(
    vessel_df,
    drift_direction_degrees,
    course_col="COG"
):
    """
    Calculate drift compatibility between a vessel's course
    and the local drift direction.

    Cosine similarity is used:

        score = cos(angle_difference)

    The result is converted to the range 0 to 1:

        1.0 = highly compatible
        0.5 = perpendicular
        0.0 = opposite direction

    If drift information is unavailable, returns None.
    """

    # No drift information available
    if drift_direction_degrees is None:
        return None

    if vessel_df.empty:
        return None

    # Remove records without course information
    valid_courses = vessel_df[course_col].dropna()

    if valid_courses.empty:
        return None

    # Calculate compatibility for every available course
    compatibility_scores = []

    for course in valid_courses:

        angle_difference = circular_angle_difference(
            float(course),
            float(drift_direction_degrees)
        )

        angle_difference_rad = math.radians(
            angle_difference
        )

        cosine_similarity = math.cos(
            angle_difference_rad
        )

        # Convert [-1, 1] to [0, 1]
        compatibility_score = (
            cosine_similarity + 1
        ) / 2

        compatibility_scores.append(
            compatibility_score
        )

    # Use the strongest compatibility observed
    # along the vessel trajectory.
    score = max(compatibility_scores)

    return float(score)


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
    # 5. Example drift direction
    # ---------------------------------------------------------
    #
    # This is ONLY a dummy value for testing.
    # Later, Shazmeen/environmental processing can provide
    # the real local drift direction.
    #
    # 90 degrees = eastward
    # ---------------------------------------------------------

    drift_direction_degrees = 90.0

    # ---------------------------------------------------------
    # 6. Calculate drift compatibility
    # ---------------------------------------------------------

    print("\nDrift compatibility results:")

    for mmsi, vessel_df in trajectories.items():

        score = calculate_drift_compatibility(
            vessel_df,
            drift_direction_degrees
        )

        print(
            f"MMSI {mmsi}: "
            f"drift_compatibility_score="
            f"{score:.3f}"
        )

    # ---------------------------------------------------------
    # 7. Test missing drift behavior
    # ---------------------------------------------------------

    missing_drift_score = calculate_drift_compatibility(
        trajectories[100000000],
        None
    )

    print(
        "\nMissing drift test:"
        f" score={missing_drift_score}"
    )