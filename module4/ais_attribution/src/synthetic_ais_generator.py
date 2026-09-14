import pandas as pd
import numpy as np
from datetime import timedelta


def generate_synthetic_ais(
    output_path,
    source_lat,
    source_lon,
    start_time,
    drift_direction_degrees=90,
    num_vessels=20,
    num_points=30,
    seed=42
):
    """
    Generates synthetic AIS data for testing the ranking pipeline.

    Deliberately seeds ONE vessel (returned as `seeded_mmsi`) with a track
    that is close in space, within the time window, and drift-aligned with
    `drift_direction_degrees` — this is the vessel the pipeline is expected
    to rank at/near the top (§28 test_scoring_known_answer, §29
    test_full_pipeline_synthetic).

    Returns (df, seeded_mmsi) so tests can assert against the known answer.
    """
    np.random.seed(seed)
    records = []
    start_time = pd.to_datetime(start_time, utc=True)

    seeded_vessel_index = 0  # vessel 0 is the deliberately compatible one
    seeded_mmsi = str(100000000 + seeded_vessel_index)

    for vessel_id in range(num_vessels):
        mmsi = str(100000000 + vessel_id)
        vessel_name = f"Synthetic Vessel {vessel_id + 1}"
        is_seeded = (vessel_id == seeded_vessel_index)

        if is_seeded:
            # Start VERY close to the source, well within a typical search margin
            lat = source_lat + np.random.uniform(-0.02, 0.02)
            lon = source_lon + np.random.uniform(-0.02, 0.02)
        else:
            # Random vessels scattered across a broader region — some near, most not
            lat = source_lat + np.random.uniform(-1.0, 1.0)
            lon = source_lon + np.random.uniform(-1.0, 1.0)

        for point in range(num_points):
            timestamp = start_time + timedelta(minutes=6 * point)

            if is_seeded:
                # Move consistently along the drift direction (simple heading-based step)
                step_deg = 0.005
                lat += step_deg * np.cos(np.radians(drift_direction_degrees))
                lon += step_deg * np.sin(np.radians(drift_direction_degrees))
                cog = drift_direction_degrees + np.random.uniform(-5, 5)  # tight alignment
                sog = np.random.uniform(8, 12)
            else:
                lat += np.random.uniform(-0.01, 0.01)
                lon += np.random.uniform(-0.01, 0.01)
                cog = np.random.uniform(0, 360)  # random heading, no drift alignment
                sog = np.random.uniform(5, 15)

            records.append({
                "MMSI": mmsi,
                "vessel_name": vessel_name,
                "timestamp": timestamp,
                "latitude": lat,
                "longitude": lon,
                "SOG": sog,
                "COG": cog,
                "heading": cog
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    return df, seeded_mmsi


if __name__ == "__main__":
    df, seeded_mmsi = generate_synthetic_ais(
        output_path="data/synthetic/synthetic_ais.csv",
        source_lat=13.0827,
        source_lon=80.2707,
        start_time="2026-09-01T10:00:00Z",
        drift_direction_degrees=90
    )

    print(f"Synthetic AIS dataset generated successfully.")
    print(f"Seeded drift-compatible vessel MMSI: {seeded_mmsi} — expected to rank near top.")