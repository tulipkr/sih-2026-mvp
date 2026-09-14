"""
Module 4 main entrypoint: AIS ingestion + vessel candidate ranking.

Run as:  python -m src.ranking_pipeline [--source_estimate PATH] [--ais PATH]
         [--ais_source_format marinecadastre|synthetic] [--ais_data_source_label LABEL]
         [--config PATH] [--output_dir PATH]

(Module invocation, not `python src/ranking_pipeline.py` directly -- this
file uses package-relative imports so it can be dropped into a different
repository without needing `src/` itself on sys.path.)
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .search_window import calculate_search_window, load_config
from .ingest_ais import clean_ais_data
from .normalize_schema import normalize_ais_schema, detect_source_format
from .trajectory_reconstruction import preprocess_ais_data, reconstruct_trajectories
from .spatial_filter import filter_by_bbox
from .temporal_filter import filter_by_time
from .closest_approach import calculate_closest_approach
from .temporal_compatibility import check_temporal_compatibility
from .drift_compatibility import calculate_drift_compatibility
from .ais_completeness import calculate_completeness_for_trajectories
from .scoring import calculate_composite_score, create_evidence_summary
from .geojson_export import export_ais_to_geojson
from .utils import validate_source_estimate, extract_region_lat_lon, extract_drift_direction_degrees

logger = logging.getLogger(__name__)

MODULE_ROOT = Path(__file__).resolve().parent.parent

DISCLAIMER = (
    "Ranking indicates compatibility with available evidence; it is not a "
    "determination of legal responsibility. AIS gaps increase uncertainty "
    "and are not evidence of guilt."
)


def rank_candidates(candidate_scores, top_k=10):
    """Rank vessels by heuristic composite score. Ties at the top-k cutoff
    are all retained (spec §16: never silently break ties)."""
    if not candidate_scores:
        return pd.DataFrame()

    ranking_df = pd.DataFrame(candidate_scores)
    ranking_df = ranking_df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = ranking_df["composite_score"].rank(method="min", ascending=False).astype(int)

    if len(ranking_df) > top_k:
        cutoff_score = ranking_df.iloc[top_k - 1]["composite_score"]
        ranking_df = ranking_df[ranking_df["composite_score"] >= cutoff_score].copy()

    return ranking_df.reset_index(drop=True)


def _candidate_dict(row):
    return {
        "rank": int(row["rank"]),
        "mmsi": str(row["MMSI"]),
        "vessel_name": row.get("vessel_name") if pd.notna(row.get("vessel_name")) else None,
        "vessel_type": row.get("vessel_type") if pd.notna(row.get("vessel_type")) else None,
        "closest_approach_km": (
            float(row["closest_approach_km"]) if pd.notna(row.get("closest_approach_km")) else None
        ),
        "closest_approach_time_utc": row.get("closest_approach_time_utc") or None,
        "temporal_compatible": bool(row["temporal_compatible"]),
        "trajectory_drift_compatibility_score": (
            float(row["trajectory_drift_compatibility_score"])
            if pd.notna(row.get("trajectory_drift_compatibility_score")) else None
        ),
        "ais_completeness_score": (
            float(row["ais_completeness_score"]) if pd.notna(row.get("ais_completeness_score")) else None
        ),
        "composite_score": float(row["composite_score"]),
        "score_type": "heuristic_composite_score",
        "evidence_summary": row.get("evidence_summary") or [],
    }


def build_output(
    source_estimate, ais_data_source, search_bbox=None, search_time_window=None,
    total_vessels_in_window=0, ais_coverage_note="", ranked_candidates_df=None,
):
    """Assembles the exact spec §9 candidate_ranking.json schema. Used for
    both the full-pipeline result AND every short-circuit/empty case, so
    every code path produces the same schema-valid shape."""
    ranked_candidates = []
    if ranked_candidates_df is not None and not ranked_candidates_df.empty:
        ranked_candidates = [_candidate_dict(row) for _, row in ranked_candidates_df.iterrows()]

    return {
        "scene_id": source_estimate.get("scene_id"),
        "region_id": source_estimate.get("region_id"),
        "ais_data_source": ais_data_source,
        "search_bbox": search_bbox,
        "search_time_window": search_time_window,
        "total_vessels_in_window": int(total_vessels_in_window),
        "ais_coverage_note": ais_coverage_note,
        "ranked_candidates": ranked_candidates,
        "disclaimer": DISCLAIMER,
        "backtracking_valid": bool(source_estimate.get("backtracking_valid", False)),
        "no_oil_detected": bool(source_estimate.get("no_oil_detected", False)),
    }


def save_candidate_ranking(output_data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)
    return output_path


def run_pipeline(
    source_estimate_path=None,
    ais_input_path=None,
    ais_source_format=None,
    ais_data_source_label=None,
    config_path=None,
    output_dir=None,
):
    """
    Run the complete AIS attribution pipeline.

    Args:
        source_estimate_path: path to Shazmeen's source_estimate.json
            (default: config/../data/source_estimate/source_estimate.json).
        ais_input_path: path to a raw AIS CSV, ANY supported source format
            (default: the bundled synthetic demo CSV).
        ais_source_format: 'marinecadastre' | 'synthetic' | None (auto-detect
            from columns -- the module never needs code changes to accept a
            different source, only a different input file, per spec §16).
        ais_data_source_label: what to record in the output's
            ais_data_source field. Defaults to ais_source_format (or the
            auto-detected format) unless explicitly overridden -- e.g. pass
            the real incident's name once one is chosen.
        config_path: defaults to this module's own config/config.yaml.
        output_dir: where candidate_ranking.json / normalized_ais_tracks.geojson
            go (default: <module_root>/outputs/).

    Returns:
        (ranked_candidates_df, output_data_dict) -- the dict is exactly
        what was written to candidate_ranking.json.
    """
    logger.info("Starting AIS attribution pipeline.")

    source_estimate_path = Path(source_estimate_path) if source_estimate_path else (
        MODULE_ROOT / "data" / "source_estimate" / "source_estimate.json"
    )
    ais_input_path = Path(ais_input_path) if ais_input_path else (
        MODULE_ROOT / "data" / "synthetic" / "synthetic_ais.csv"
    )
    output_dir = Path(output_dir) if output_dir else (MODULE_ROOT / "outputs")

    config = load_config(config_path)
    top_k = config.get("top_k_candidates", 10)
    score_weights = config.get("score_weights", {})
    expected_interval = config.get("expected_ais_reporting_interval_minutes", 6)
    temporal_margin_hours = config.get("search_time_margin_hours", 6)

    with open(source_estimate_path, "r", encoding="utf-8") as file:
        source_estimate = json.load(file)

    validation = validate_source_estimate(source_estimate)  # raises on structurally malformed input

    if validation["short_circuit"]:
        logger.warning(f"Pipeline short-circuited: {validation['reason']}")
        output_data = build_output(
            source_estimate,
            ais_data_source=ais_data_source_label or "synthetic",
            ais_coverage_note=f"Short-circuited before AIS analysis: {validation['reason']}.",
        )
        save_candidate_ranking(output_data, output_dir / "candidate_ranking.json")
        return pd.DataFrame(), output_data

    search_window = calculate_search_window(source_estimate, config)
    search_bbox_out = {
        "min_lat": search_window["bbox"]["min_latitude"],
        "max_lat": search_window["bbox"]["max_latitude"],
        "min_lon": search_window["bbox"]["min_longitude"],
        "max_lon": search_window["bbox"]["max_longitude"],
    }
    search_time_window_out = dict(search_window["time_window"])

    if not ais_input_path.exists():
        raise FileNotFoundError(
            f"AIS source file not found: {ais_input_path.resolve()} -- cannot proceed "
            f"without AIS data (spec §15: this must be an explicit error, never a "
            f"fabricated/empty result presented as if analysis ran)."
        )
    try:
        raw_ais_df = pd.read_csv(ais_input_path)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"AIS source file at {ais_input_path} could not be parsed: {exc}") from exc

    detected_format = ais_source_format or detect_source_format(raw_ais_df)
    ais_data_source = ais_data_source_label or detected_format
    logger.info(f"AIS source: {ais_input_path} (format={detected_format}, "
                f"ais_data_source label={ais_data_source}), {len(raw_ais_df)} raw records")

    normalized_df = normalize_ais_schema(raw_ais_df, source_format=detected_format)

    cleaned_df, drop_reasons = clean_ais_data(normalized_df)
    logger.info(f"AIS cleaning drop reasons: {drop_reasons}")

    processed_df = preprocess_ais_data(cleaned_df, expected_interval_minutes=expected_interval)

    spatial_filtered = filter_by_bbox(processed_df, search_window["bbox"])
    filtered_df = filter_by_time(spatial_filtered, search_window["time_window"])
    logger.info(f"Combined filtering: {len(processed_df)} -> {len(filtered_df)} records")

    total_vessels_in_window = filtered_df["MMSI"].nunique() if not filtered_df.empty else 0

    coverage_note = (
        f"{len(raw_ais_df)} raw AIS records loaded; {total_vessels_in_window} distinct vessel(s) "
        f"found in the search window. Zero vessels in-window does not rule out AIS-dark vessels "
        f"that were present but not transmitting."
    )

    if filtered_df.empty:
        logger.warning("No AIS records found in the search window.")
        # Bypass geopandas for the empty case: writing a 0-row GeoDataFrame
        # via fiona/pyogrio can fail (no data to infer a geometry schema
        # from) depending on the installed GIS stack version -- a plain,
        # valid empty FeatureCollection is unambiguous and always correct.
        _write_empty_geojson(output_dir / "normalized_ais_tracks.geojson")
        output_data = build_output(
            source_estimate, ais_data_source, search_bbox_out, search_time_window_out,
            total_vessels_in_window=0, ais_coverage_note=coverage_note,
        )
        save_candidate_ranking(output_data, output_dir / "candidate_ranking.json")
        return pd.DataFrame(), output_data

    trajectories = reconstruct_trajectories(filtered_df)
    completeness_results = calculate_completeness_for_trajectories(
        trajectories, expected_interval_minutes=expected_interval
    )

    region_lat, region_lon = extract_region_lat_lon(source_estimate["probable_source_region"])
    drift_direction_degrees = extract_drift_direction_degrees(source_estimate)

    proximity_weight = score_weights.get("proximity", 0.5)
    temporal_weight = score_weights.get("temporal", 0.2)
    drift_weight = score_weights.get("trajectory_drift", 0.2)

    candidate_scores = []
    for mmsi, vessel_df in trajectories.items():
        closest = calculate_closest_approach(vessel_df, region_lat, region_lon)
        closest_distance = closest["closest_approach_km"]
        closest_time = closest["closest_approach_time"]

        temporal_result = check_temporal_compatibility(
            vessel_df, search_window["time_window"], margin_hours=temporal_margin_hours
        )
        temporal_compatible = temporal_result["temporal_compatible"]

        drift_score = calculate_drift_compatibility(vessel_df, drift_direction_degrees)

        completeness_score = completeness_results.get(mmsi)

        composite_score = calculate_composite_score(
            closest_distance, temporal_compatible, drift_score,
            proximity_weight=proximity_weight, temporal_weight=temporal_weight, drift_weight=drift_weight,
        )

        vessel_name = None
        if "vessel_name" in vessel_df.columns:
            names = vessel_df["vessel_name"].dropna()
            vessel_name = names.iloc[0] if not names.empty else None
        vessel_type = None
        if "vessel_type" in vessel_df.columns:
            types = vessel_df["vessel_type"].dropna()
            vessel_type = types.iloc[0] if not types.empty else None

        evidence_summary = create_evidence_summary(
            closest_distance, closest_time, temporal_compatible, drift_score, completeness_score
        )

        candidate_scores.append({
            "MMSI": mmsi,
            "vessel_name": vessel_name,
            "vessel_type": vessel_type,
            "closest_approach_km": closest_distance,
            "closest_approach_time_utc": closest_time,
            "temporal_compatible": temporal_compatible,
            "trajectory_drift_compatibility_score": drift_score,
            "ais_completeness_score": completeness_score,
            "composite_score": composite_score,
            "evidence_summary": [evidence_summary],
        })

    ranked_candidates = rank_candidates(candidate_scores, top_k=top_k)

    export_ais_to_geojson(filtered_df, output_path=output_dir / "normalized_ais_tracks.geojson")

    output_data = build_output(
        source_estimate, ais_data_source, search_bbox_out, search_time_window_out,
        total_vessels_in_window=total_vessels_in_window, ais_coverage_note=coverage_note,
        ranked_candidates_df=ranked_candidates,
    )
    save_candidate_ranking(output_data, output_dir / "candidate_ranking.json")

    logger.info(f"AIS attribution pipeline completed. {len(ranked_candidates)} candidate(s) ranked.")
    return ranked_candidates, output_data


def _write_empty_geojson(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": []}, f, indent=2)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Module 4: AIS vessel attribution ranking.")
    parser.add_argument("--source_estimate", default=None)
    parser.add_argument("--ais", default=None)
    parser.add_argument("--ais_source_format", default=None, choices=["marinecadastre", "synthetic"])
    parser.add_argument("--ais_data_source_label", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    ranked, output_data = run_pipeline(
        source_estimate_path=args.source_estimate, ais_input_path=args.ais,
        ais_source_format=args.ais_source_format, ais_data_source_label=args.ais_data_source_label,
        config_path=args.config, output_dir=args.output_dir,
    )

    print("\nPipeline completed.")
    print(f"Ranked candidates: {len(ranked)}")
    if not ranked.empty:
        print("\nTop candidates:")
        print(ranked[["rank", "MMSI", "composite_score"]].to_string(index=False))
    else:
        print("No candidates available.")
