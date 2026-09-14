import logging
import math
import pandas as pd

logger = logging.getLogger(__name__)


def haversine_distance_km(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate the great-circle distance between two
    latitude/longitude points using the Haversine formula.

    Returns distance in kilometers.
    """

    earth_radius_km = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius_km * c


def calculate_closest_approach(
    vessel_df,
    source_latitude,
    source_longitude,
    lat_col="latitude",
    lon_col="longitude"
):
    """
    Find the closest AIS position of one vessel
    to the probable source location.

    Returns:
        Dictionary containing:
        - closest_approach_km
        - closest_approach_time
        - latitude
        - longitude
    """

    if vessel_df.empty:
        return {
            "closest_approach_km": None,
            "closest_approach_time": None,
            "latitude": None,
            "longitude": None
        }

    distances = []

    for _, row in vessel_df.iterrows():

        distance = haversine_distance_km(
            row[lat_col],
            row[lon_col],
            source_latitude,
            source_longitude
        )

        distances.append(distance)

    vessel_df = vessel_df.copy()
    vessel_df["distance_to_source_km"] = distances

    closest_index = vessel_df[
        "distance_to_source_km"
    ].idxmin()

    closest_row = vessel_df.loc[closest_index]

    result = {
        "closest_approach_km": float(
            closest_row["distance_to_source_km"]
        ),
        "closest_approach_time": (
            closest_row["timestamp"].isoformat()
            if pd.notna(closest_row["timestamp"])
            else None
        ),
        "latitude": float(
            closest_row[lat_col]
        ),
        "longitude": float(
            closest_row[lon_col]
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
    # 5. Probable source location
    # ---------------------------------------------------------

    source_latitude = 13.0827
    source_longitude = 80.2707

    # ---------------------------------------------------------
    # 6. Calculate closest approach for each vessel
    # ---------------------------------------------------------

    print("\nClosest approach results:")

    for mmsi, vessel_df in trajectories.items():

        result = calculate_closest_approach(
            vessel_df,
            source_latitude,
            source_longitude
        )

        print(
            f"MMSI {mmsi}: "
            f"{result['closest_approach_km']:.3f} km "
            f"at {result['closest_approach_time']}"
        )