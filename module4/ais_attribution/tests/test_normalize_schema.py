import pandas as pd
import pytest

from src.normalize_schema import normalize_ais_schema, detect_source_format, NORMALIZED_COLUMNS


def test_marinecadastre_and_synthetic_map_to_identical_schema():
    """spec §28 test_normalize_schema: both MarineCadastre columns and
    synthetic generator output map to the IDENTICAL internal schema --
    the central decoupling requirement of the whole module."""
    mc_df = pd.DataFrame({
        "MMSI": [367123456], "BaseDateTime": ["2026-09-01T10:00:00"],
        "LAT": [13.09], "LON": [80.27], "SOG": [9.5], "COG": [44.0],
        "Heading": [44], "VesselName": ["MC VESSEL"], "VesselType": [70],
    })
    synthetic_df = pd.DataFrame({
        "MMSI": [100000000], "timestamp": ["2026-09-01T10:00:00Z"],
        "latitude": [13.09], "longitude": [80.27], "SOG": [9.5], "COG": [44.0],
        "heading": [44], "vessel_name": ["SYN VESSEL"],
    })

    normalized_mc = normalize_ais_schema(mc_df, source_format="marinecadastre")
    normalized_syn = normalize_ais_schema(synthetic_df, source_format="synthetic")

    assert list(normalized_mc.columns) == list(normalized_syn.columns) == NORMALIZED_COLUMNS
    assert normalized_mc["MMSI"].dtype == normalized_syn["MMSI"].dtype
    assert str(normalized_mc["timestamp"].dtype) == str(normalized_syn["timestamp"].dtype)


def test_detect_source_format_marinecadastre():
    df = pd.DataFrame({"MMSI": [1], "BaseDateTime": ["x"], "LAT": [1], "LON": [1]})
    assert detect_source_format(df) == "marinecadastre"


def test_detect_source_format_synthetic():
    df = pd.DataFrame({"MMSI": [1], "timestamp": ["x"], "latitude": [1], "longitude": [1]})
    assert detect_source_format(df) == "synthetic"


def test_detect_source_format_unrecognized_raises():
    df = pd.DataFrame({"some_column": [1]})
    with pytest.raises(ValueError, match="Unrecognized AIS source format"):
        detect_source_format(df)


def test_missing_optional_fields_do_not_break_normalization():
    """spec §7/§17: SOG/COG/heading/vessel_type missing must not discard
    the record -- normalization must still succeed and fill NA."""
    df = pd.DataFrame({
        "MMSI": [367999888], "BaseDateTime": ["2026-09-01T10:03:00"],
        "LAT": [13.10], "LON": [80.28],
    })
    normalized = normalize_ais_schema(df, source_format="marinecadastre")
    assert len(normalized) == 1
    assert normalized["SOG"].isna().all()
    assert normalized["vessel_name"].isna().all()


def test_normalize_after_missing_required_column_raises():
    df = pd.DataFrame({"MMSI": [1], "BaseDateTime": ["x"]})  # no LAT/LON
    with pytest.raises(ValueError, match="missing"):
        normalize_ais_schema(df, source_format="marinecadastre")

def test_marinecadastre_actual_2021_lowercase_format():
    """Actual Marine Cadastre 2021 download format must normalize correctly."""

    df = pd.DataFrame({
        "mmsi": [368210670],
        "base_date_time": ["2021-10-02 14:33:46"],
        "longitude": [-117.05272],
        "latitude": [35.23633],
        "sog": [10.0],
        "cog": [180.0],
        "heading": [180.0],
        "vessel_name": ["MARLIN MAGIN"],
        "vessel_type": [37],
    })

    normalized = normalize_ais_schema(
        df,
        source_format="marinecadastre"
    )

    assert list(normalized.columns) == NORMALIZED_COLUMNS
    assert normalized.loc[0, "MMSI"] == "368210670"
    assert normalized.loc[0, "timestamp"] == pd.Timestamp(
        "2021-10-02 14:33:46", tz="UTC"
    )
    assert normalized.loc[0, "latitude"] == 35.23633
    assert normalized.loc[0, "longitude"] == -117.05272