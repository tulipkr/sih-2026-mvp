import logging

logger = logging.getLogger(__name__)


def calculate_proximity_score(
    closest_approach_km,
    max_distance_km=100
):
    """
    Convert closest approach distance into a score from 0 to 1.

    Smaller distance = higher score.
    """

    if closest_approach_km is None:
        return 0.0

    score = 1 - (
        closest_approach_km / max_distance_km
    )

    return max(0.0, min(1.0, score))


def calculate_temporal_score(temporal_compatible):
    """
    Convert temporal compatibility into a score.

    Compatible = 1
    Not compatible = 0
    """

    return 1.0 if temporal_compatible else 0.0


def calculate_composite_score(
    closest_approach_km,
    temporal_compatible,
    drift_compatibility_score,
    proximity_weight=0.5,
    temporal_weight=0.2,
    drift_weight=0.3
):
    """
    Calculate the heuristic composite score.

    IMPORTANT:
    This is a ranking score, NOT a probability.

    AIS completeness is deliberately NOT included.
    """

    proximity_score = calculate_proximity_score(
        closest_approach_km
    )

    temporal_score = calculate_temporal_score(
        temporal_compatible
    )

    if drift_compatibility_score is None:
        drift_score = 0.0
    else:
        drift_score = drift_compatibility_score

    composite_score = (
        proximity_weight * proximity_score
        + temporal_weight * temporal_score
        + drift_weight * drift_score
    )

    return round(composite_score, 4)


def create_evidence_summary(
    closest_approach_km,
    closest_approach_time,
    temporal_compatible,
    drift_compatibility_score,
    ais_completeness_score
):
    """
    Create a human-readable evidence summary
    for a vessel candidate.
    """

    if closest_approach_km is not None:
        proximity_text = (
            f"Closest approach: "
            f"{closest_approach_km:.2f} km"
        )
    else:
        proximity_text = (
            "Closest approach: unavailable"
        )

    if closest_approach_time is not None:
        time_text = (
            f"Closest approach time: "
            f"{closest_approach_time}"
        )
    else:
        time_text = (
            "Closest approach time: unavailable"
        )

    if temporal_compatible:
        temporal_text = (
            "Temporal compatibility: yes"
        )
    else:
        temporal_text = (
            "Temporal compatibility: no"
        )

    if drift_compatibility_score is None:
        drift_text = (
            "Drift compatibility: unavailable"
        )
    else:
        drift_text = (
            f"Drift compatibility: "
            f"{drift_compatibility_score:.3f}"
        )

    if ais_completeness_score is None:
        completeness_text = (
            "AIS completeness: unavailable"
        )
    else:
        completeness_text = (
            f"AIS completeness: "
            f"{ais_completeness_score:.3f}"
        )

    return (
        f"{proximity_text}; "
        f"{time_text}; "
        f"{temporal_text}; "
        f"{drift_text}; "
        f"{completeness_text}. "
        "AIS completeness is contextual only and "
        "does not affect the composite score."
    )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print("Composite score examples:")

    score_1 = calculate_composite_score(
        closest_approach_km=2.5,
        temporal_compatible=True,
        drift_compatibility_score=1.0
    )

    score_2 = calculate_composite_score(
        closest_approach_km=50,
        temporal_compatible=True,
        drift_compatibility_score=0.5
    )

    score_3 = calculate_composite_score(
        closest_approach_km=100,
        temporal_compatible=False,
        drift_compatibility_score=0.0
    )

    print(
        f"Vessel 1: {score_1:.4f}"
    )

    print(
        f"Vessel 2: {score_2:.4f}"
    )

    print(
        f"Vessel 3: {score_3:.4f}"
    )

    print(
        "\nScore type: "
        "heuristic_composite_score"
    )

    print(
        "AIS completeness: NOT included in score"
    )

    print(
        "Interpretation: ranking signal, NOT probability"
    )

    evidence = create_evidence_summary(
        closest_approach_km=2.556,
        closest_approach_time=(
            "2025-01-01T00:00:00Z"
        ),
        temporal_compatible=True,
        drift_compatibility_score=1.0,
        ais_completeness_score=1.0
    )

    print("\nEvidence summary:")
    print(evidence)