import logging
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)


def export_ais_to_geojson(
    df,
    output_path="outputs/normalized_ais_tracks.geojson"
):
    """
    Export normalized AIS records as GeoJSON.

    Coordinate system:
        WGS84 / EPSG:4326
    """

    required_columns = [
        "MMSI",
        "timestamp",
        "latitude",
        "longitude"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "latitude",
            "longitude",
            "timestamp"
        ]
    )

    geometry = [
        Point(longitude, latitude)
        for longitude, latitude
        in zip(
            data["longitude"],
            data["latitude"]
        )
    ]

    geo_df = gpd.GeoDataFrame(
        data,
        geometry=geometry,
        crs="EPSG:4326"
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    geo_df.to_file(
        output_path,
        driver="GeoJSON"
    )

    logger.info(
        f"GeoJSON exported: "
        f"{len(geo_df)} records → {output_path}"
    )

    return output_path


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO
    )

    input_path = (
        "data/processed/normalized_ais.csv"
    )

    df = pd.read_csv(input_path)

    output_path = export_ais_to_geojson(
        df
    )

    print(
        "\nGeoJSON export completed."
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Records exported: {len(df)}"
    )

    print(
        "Coordinate system: EPSG:4326"
    )