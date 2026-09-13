"""
inference_result.json schema — exactly per spec §9. This is what Bhumika's
backend and Ishita's post-processing module code against (§33). Do not
change field names/types here without telling both of them (spec's own words).
"""
from __future__ import annotations

from typing import Any

REQUIRED_FIELDS: dict[str, tuple] = {
    "scene_id": (str,),
    "model_version": (str,),
    "inference_timestamp_utc": (str,),
    "acquisition_timestamp_utc": (str, type(None)),
    "crs": (str, type(None)),
    "transform": (list, type(None)),
    "geolocation_incomplete": (bool,),
    "threshold_used": (float, int),
    "positive_pixel_fraction": (float, int),
    "mean_score_in_positive_region": (float, int, type(None)),
    "no_oil_detected": (bool,),
    "score_type": (str,),
    "fallback_used": (bool,),
    "prob_map_path": (str,),
    "mask_path": (str,),
}


class SchemaValidationError(ValueError):
    pass


def validate_inference_result(result: dict[str, Any]) -> None:
    missing = [f for f in REQUIRED_FIELDS if f not in result]
    if missing:
        raise SchemaValidationError(
            f"inference_result.json is missing required field(s): {missing}"
        )

    wrong_type = []
    for field, allowed_types in REQUIRED_FIELDS.items():
        value = result[field]
        if not isinstance(value, allowed_types):
            wrong_type.append(
                f"{field} (expected one of {[t.__name__ for t in allowed_types]}, "
                f"got {type(value).__name__})"
            )
    if wrong_type:
        raise SchemaValidationError(
            f"inference_result.json has field(s) with the wrong type: {wrong_type}"
        )

    valid_score_types = {"raw_sigmoid_output", "rule_based_threshold"}
    if result["score_type"] not in valid_score_types:
        raise SchemaValidationError(
            f"score_type must be one of {valid_score_types} unless calibration was "
            f"actually done (spec §9/§38) — got '{result['score_type']}'. "
            f"Do not relabel this as 'probability'."
        )
    # NOTE beyond the spec's literal §9 text: the spec shows score_type as a single
    # fixed value ('raw_sigmoid_output'), but doesn't address what the fallback
    # detector (§35) should report, and "raw_sigmoid_output" would misrepresent a
    # rule-based result. Added 'rule_based_threshold' for that case — flagged in
    # the README as a gap I filled rather than one the document specifies.
    if result["fallback_used"] and result["score_type"] != "rule_based_threshold":
        raise SchemaValidationError(
            "fallback_used=True but score_type is not 'rule_based_threshold' — "
            "the fallback path must not claim its output is a model's raw_sigmoid_output."
        )
    if not result["fallback_used"] and result["score_type"] != "raw_sigmoid_output":
        raise SchemaValidationError(
            "fallback_used=False but score_type is not 'raw_sigmoid_output' — "
            "the trained-model path must use this exact label per spec §9."
        )

    if not 0.0 <= result["threshold_used"] <= 1.0:
        raise SchemaValidationError(
            f"threshold_used must be in [0, 1], got {result['threshold_used']}"
        )
    if not 0.0 <= result["positive_pixel_fraction"] <= 1.0:
        raise SchemaValidationError(
            f"positive_pixel_fraction must be in [0, 1], got "
            f"{result['positive_pixel_fraction']}"
        )
    if result["no_oil_detected"] and result["positive_pixel_fraction"] != 0.0:
        raise SchemaValidationError(
            "no_oil_detected is True but positive_pixel_fraction is not 0.0 — "
            "spec §9 defines no_oil_detected as true iff positive_pixel_fraction == 0."
        )
