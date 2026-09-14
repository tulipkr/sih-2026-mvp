import pandas as pd

from src.ingest_ais import clean_ais_data
from src.drift_compatibility import calculate_drift_compatibility
from src.ranking_pipeline import rank_candidates


def test_invalid_mmsi_is_removed():
    df = pd.DataFrame({
        "MMSI": [
            100000000,
            123,
            100000001
        ],
        "vessel_name": [
            "Valid A",
            "Invalid",
            "Valid B"
        ],
        "timestamp": [
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:06:00Z",
            "2025-01-01T00:12:00Z"
        ],
        "latitude": [
            13.08,
            13.08,
            13.08
        ],
        "longitude": [
            80.27,
            80.27,
            80.27
        ],
        "SOG": [
            10.0,
            10.0,
            10.0
        ],
        "COG": [
            90.0,
            90.0,
            90.0
        ],
        "heading": [
            90.0,
            90.0,
            90.0
        ]
    })

    cleaned, _ = clean_ais_data(df)

    assert len(cleaned) == 2
    assert 123 not in cleaned["MMSI"].values


def test_duplicate_records_are_removed():

    df = pd.DataFrame({
        "MMSI": [
            100000000,
            100000000
        ],
        "vessel_name": [
            "Test Vessel",
            "Test Vessel"
        ],
        "timestamp": [
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:00:00Z"
        ],
        "latitude": [
            13.08,
            13.08
        ],
        "longitude": [
            80.27,
            80.27
        ],
        "SOG": [
            10.0,
            10.0
        ],
        "COG": [
            90.0,
            90.0
        ],
        "heading": [
            90.0,
            90.0
        ]
    })

    cleaned, _ = clean_ais_data(df)

    assert len(cleaned) == 1


def test_missing_drift_returns_none():

    trajectory = pd.DataFrame({
        "COG": [90.0, 90.0],
        "heading": [90.0, 90.0]
    })

    score = calculate_drift_compatibility(
        trajectory,
        drift_direction_degrees=None
    )

    assert score is None


def test_tied_candidates_are_all_kept():

    candidates = [
        {
            "MMSI": 100000000,
            "composite_score": 0.90
        },
        {
            "MMSI": 100000001,
            "composite_score": 0.80
        },
        {
            "MMSI": 100000002,
            "composite_score": 0.70
        },
        {
            "MMSI": 100000003,
            "composite_score": 0.70
        }
    ]

    ranked = rank_candidates(
        candidates,
        top_k=3
    )

    assert len(ranked) == 4
    assert ranked.iloc[2]["rank"] == 3
    assert ranked.iloc[3]["rank"] == 3


def test_empty_candidates():

    ranked = rank_candidates(
        [],
        top_k=10
    )

    assert ranked.empty


if __name__ == "__main__":

    print(
        "Running Part 29 edge-case tests..."
    )

    test_invalid_mmsi_is_removed()
    print("PASS: invalid MMSI")

    test_duplicate_records_are_removed()
    print("PASS: duplicate records")

    test_missing_drift_returns_none()
    print("PASS: missing drift")

    test_tied_candidates_are_all_kept()
    print("PASS: tied candidates")

    test_empty_candidates()
    print("PASS: empty candidates")

    print(
        "\nAll Part 29 edge-case tests passed."
    )