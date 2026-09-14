import re
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def clean_ais_data(df, mmsi_col="MMSI", lat_col="latitude", lon_col="longitude",
                    timestamp_col="timestamp", sog_col="SOG"):
    """
    Clean raw AIS data (operates on a normalized/generic schema — column
    names passed as params so this stays source-agnostic per §38).

    Malformed individual records are logged and skipped — never fails
    the entire ingestion over one bad row (§11).
    """
    df = df.copy()
    original_count = len(df)
    drop_reasons = {}

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Convert timestamp to UTC datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)

    # Convert essential numeric fields
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    # --- MMSI format validation (§11: 9-digit numeric, not a known test/invalid value) ---
    def is_valid_mmsi(mmsi):
        mmsi_str = str(mmsi).strip()
        if not re.fullmatch(r"\d{9}", mmsi_str):
            return False
        if mmsi_str == "000000000":  # known invalid/test MMSI
            return False
        return True

    valid_mmsi_mask = df[mmsi_col].apply(is_valid_mmsi)
    drop_reasons["invalid_mmsi"] = int((~valid_mmsi_mask).sum())
    df = df[valid_mmsi_mask]

    # --- Drop rows missing essential fields ---
    before = len(df)
    df = df.dropna(subset=[mmsi_col, timestamp_col, lat_col, lon_col])
    drop_reasons["missing_essential_fields"] = before - len(df)

    # --- Valid lat/lon ranges ---
    before = len(df)
    df = df[(df[lat_col] >= -90) & (df[lat_col] <= 90)]
    df = df[(df[lon_col] >= -180) & (df[lon_col] <= 180)]
    drop_reasons["invalid_lat_lon_range"] = before - len(df)

    # --- SOG outlier flagging (§11: SOG > 50 knots implausible — flag, don't silently trust) ---
    if sog_col in df.columns:
        df[sog_col] = pd.to_numeric(df[sog_col], errors="coerce")
        df["sog_outlier_flag"] = df[sog_col] > 50
        outlier_count = int(df["sog_outlier_flag"].sum())
        if outlier_count > 0:
            logger.warning(f"{outlier_count} records flagged with implausible SOG (>50 knots)")
    else:
        df["sog_outlier_flag"] = False

    # Remove duplicate AIS records
    before = len(df)
    df = df.drop_duplicates()
    drop_reasons["duplicates"] = before - len(df)

    df = df.reset_index(drop=True)

    logger.info(f"AIS cleaning: {original_count} → {len(df)} records "
                f"(removed {original_count - len(df)}: {drop_reasons})")

    return df, drop_reasons


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    input_path = "data/synthetic/synthetic_ais.csv"
    df = pd.read_csv(input_path)

    cleaned_df, reasons = clean_ais_data(df)

    print("\nCleaned AIS preview:")
    print(cleaned_df.head())
    print(f"\nDrop reasons: {reasons}")