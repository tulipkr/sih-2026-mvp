import pandas as pd


NORMALIZED_COLUMNS = [
    "MMSI",
    "vessel_name",
    "vessel_type",
    "timestamp",
    "latitude",
    "longitude",
    "SOG",
    "COG",
    "heading"
]


# --- Source-specific column mappings (adapters) ---
# Each maps SOURCE column name -> INTERNAL column name.
# Adding a new AIS source later = adding one more dict here, no pipeline changes.

MARINECADASTRE_COLUMN_MAP = {
    # Standard Marine Cadastre format
    "MMSI": "MMSI",
    "BaseDateTime": "timestamp",
    "LAT": "latitude",
    "LON": "longitude",
    "SOG": "SOG",
    "COG": "COG",
    "Heading": "heading",
    "VesselName": "vessel_name",
    "VesselType": "vessel_type",

    # Actual 2021 Marine Cadastre download format
    "mmsi": "MMSI",
    "base_date_time": "timestamp",
    "latitude": "latitude",
    "longitude": "longitude",
    "sog": "SOG",
    "cog": "COG",
    "heading": "heading",
    "vessel_name": "vessel_name",
    "vessel_type": "vessel_type",
}

SYNTHETIC_COLUMN_MAP = {
    "MMSI": "MMSI",
    "timestamp": "timestamp",
    "latitude": "latitude",
    "longitude": "longitude",
    "SOG": "SOG",
    "COG": "COG",
    "heading": "heading",
    "vessel_name": "vessel_name",
}


def detect_source_format(df):
    """
    Detects which known AIS source format a raw dataframe matches,
    based on its column names.
    """
    columns = set(c.strip() for c in df.columns)

    if "BaseDateTime" in columns and "LAT" in columns:
        return "marinecadastre"
    if "latitude" in columns and "timestamp" in columns:
        return "synthetic"

    raise ValueError(
        f"Unrecognized AIS source format — columns don't match any known "
        f"adapter (MarineCadastre or synthetic). Columns found: {sorted(columns)}"
    )


def normalize_ais_schema(df, source_format=None):
    """
    Convert AIS data from any supported raw source into the project's
    single internal schema (§12 — the central design requirement).

    source_format: 'marinecadastre' | 'synthetic' | None (auto-detect)
    """
    df = df.copy()
    df.columns = [column.strip() for column in df.columns]

    if source_format is None:
        source_format = detect_source_format(df)

    column_maps = {
        "marinecadastre": MARINECADASTRE_COLUMN_MAP,
        "synthetic": SYNTHETIC_COLUMN_MAP,
    }

    if source_format not in column_maps:
        raise ValueError(f"Unknown source_format: {source_format}")

    column_map = column_maps[source_format]

    # Rename source-specific columns to internal names
    df = df.rename(columns=column_map)

    # Check essential internal columns exist after mapping
    required_columns = ["MMSI", "timestamp", "latitude", "longitude"]
    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"After normalizing from '{source_format}', still missing "
            f"required columns: {missing_columns}"
        )

    # Add optional columns if missing — never discard a record for lacking these (§7)
    for column in ["vessel_name", "vessel_type", "SOG", "COG", "heading"]:
        if column not in df.columns:
            df[column] = pd.NA

    # Keep only the standard internal schema, in a fixed column order
    df = df[NORMALIZED_COLUMNS]

    # Normalize data types — identical regardless of source
    df["MMSI"] = df["MMSI"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["SOG"] = pd.to_numeric(df["SOG"], errors="coerce")
    df["COG"] = pd.to_numeric(df["COG"], errors="coerce")
    df["heading"] = pd.to_numeric(df["heading"], errors="coerce")

    return df


if __name__ == "__main__":
    # Prove the decoupling works: run the SAME function against TWO different
    # raw formats and confirm both land on the identical internal schema (§40).
    synthetic_df = pd.read_csv("data/synthetic/synthetic_ais.csv")
    normalized_synthetic = normalize_ais_schema(synthetic_df)

    print("Normalized synthetic AIS:")
    print(normalized_synthetic.head())
    print("\nColumns:", normalized_synthetic.columns.tolist())

    # Uncomment once a MarineCadastre sample is downloaded:
    # mc_df = pd.read_csv("data/marinecadastre_sample/sample.csv")
    # normalized_mc = normalize_ais_schema(mc_df)
    # assert list(normalized_mc.columns) == list(normalized_synthetic.columns)
    # print("\nMarineCadastre and synthetic both normalize to identical schema ✓")

    normalized_synthetic.to_csv(
        "data/processed/normalized_ais.csv",
        index=False
    )

    print("\nSaved normalized AIS dataset to:")
    print("data/processed/normalized_ais.csv")