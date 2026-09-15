import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union

logger = logging.getLogger("RunManager")

# Active lock tracking for concurrent run prevention
_ACTIVE_RUNS = set()


def generate_run_id() -> str:
    """Generates a unique, timestamped run_id."""
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:6]
    return f"run_{now_str}_{short_uuid}"


def get_utc_now_iso() -> str:
    """Returns current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def acquire_run_lock(scene_id: str) -> bool:
    """Prevents duplicate concurrent runs on the same scene."""
    if scene_id in _ACTIVE_RUNS:
        return False
    _ACTIVE_RUNS.add(scene_id)
    return True


def release_run_lock(scene_id: str) -> None:
    """Releases lock for a scene_id once execution completes."""
    _ACTIVE_RUNS.discard(scene_id)


def init_run_directory(base_outputs_dir: str, run_id: str) -> Path:
    """Creates directory structure for a specific run on disk."""
    run_dir = Path(base_outputs_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_json_safely(file_path_or_data: Union[str, Path, Dict[str, Any], list, None]) -> Optional[Any]:
    """
    Safely loads a JSON/GeoJSON file from disk or returns the data directly 
    if it is already a parsed dictionary or list.
    """
    if file_path_or_data is None:
        return None

    # If already parsed dict or list passed in, return directly
    if isinstance(file_path_or_data, (dict, list)):
        return file_path_or_data

    try:
        p = Path(file_path_or_data)
        if not p.exists():
            logger.warning(f"Expected file missing from disk: {file_path_or_data}")
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading file {file_path_or_data}: {e}")
        return None


def aggregate_and_save_run_result(
    run_dir: Path,
    run_id: str,
    scene_id: str,
    run_mode: str,
    started_at_utc: str,
    overall_status: str,
    stages_manifest: Dict[str, Any],
    failed_stage: Optional[str] = None,
    output_file_paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Aggregates stage statuses and output files into the final run_result.json schema.
    Passes through GeoJSON and text unmodified byte-for-byte/key-for-key.
    """
    completed_at_utc = get_utc_now_iso()
    output_file_paths = output_file_paths or {}

    # Extract raw data payloads from file paths on disk (or directly if dict)
    spill_geojson = load_json_safely(output_file_paths.get("spill_geometry_geojson"))
    spill_summary = load_json_safely(output_file_paths.get("spill_summary"))
    source_estimate = load_json_safely(output_file_paths.get("source_estimate"))
    candidate_ranking = load_json_safely(output_file_paths.get("candidate_ranking"))
    normalized_ais_tracks = load_json_safely(output_file_paths.get("normalized_ais_tracks"))

    run_result = {
        "run_id": run_id,
        "scene_id": scene_id,
        "run_mode": run_mode,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "overall_status": overall_status,
        "stages": stages_manifest,
        "failed_stage": failed_stage,
        "results": {
            "spill_geometry_geojson": spill_geojson,
            "spill_summary": spill_summary,
            "source_estimate": source_estimate,
            "candidate_ranking": candidate_ranking,
            "normalized_ais_tracks": normalized_ais_tracks,
        },
    }

    result_path = run_dir / "run_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(run_result, f, indent=2, ensure_ascii=False)

    logger.info(f"Aggregated result successfully written to: {result_path}")
    return run_result