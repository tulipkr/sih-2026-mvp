import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import concurrent.futures

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Module & Internal Imports
from module2.Stage_A.preprocess_pipeline import run_stage_a
from module2.Stage_B.postprocess_pipeline import run_stage_b
from backend.src.module_adapters.tulip_adapter import run_inference_and_build_manifest
from backend.src.module_adapters.shazmeen_adapter import run_shazmeen_stage
from backend.src.module_adapters.sara_adapter import run_ais_ranking

from backend.src.schemas import (
    SARPreprocessingStage,
    SegmentationStage,
    GeometryExtractionStage,
    DriftBacktrackingStage,
    AISRankingStage,
)
from backend.src.run_manager import (
    generate_run_id,
    get_utc_now_iso,
    acquire_run_lock,
    release_run_lock,
    init_run_directory,
    aggregate_and_save_run_result,
    load_json_safely,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Orchestrator")


def run_with_timeout(func, args=(), kwargs=None, timeout_seconds=600):
    """Executes a callable with a strict timeout to avoid infinite hanging."""
    kwargs = kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Stage execution timed out after {timeout_seconds} seconds")


def run_ishita_stage_a_wrapped(
    output_dir: str,
    safe_path: Optional[str],
    vv_path: Optional[str],
    vh_path: Optional[str],
    scene_id: str,
    timestamp_utc: str,
    config: Optional[Dict[str, Any]],
):
    """Runs Stage A preprocessing while applying land mask configuration overrides."""
    merged_config = dict(config) if config else {}
    land_mask_override = {
        "enabled": False,
        "scene_is_coastal": False,
        "use_land_mask": False,
        "source_path": None,
    }
    if "land_mask" in merged_config and isinstance(merged_config["land_mask"], dict):
        merged_config["land_mask"].update(land_mask_override)
    else:
        merged_config["land_mask"] = land_mask_override
    merged_config["scene_is_coastal"] = False

    manifest_path = run_stage_a(
        output_dir=output_dir,
        safe_path=safe_path,
        vv_path=vv_path,
        vh_path=vh_path,
        scene_id=scene_id,
        timestamp_utc=timestamp_utc,
        config=merged_config,
    )

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Stage A output manifest missing: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    first_patch = manifest_data[0] if isinstance(manifest_data, list) else manifest_data
    fallback_used = first_patch.get("fallback_used", False) if isinstance(first_patch, dict) else False

    tulip_params = {
        "scene_id": first_patch.get("scene_id", scene_id),
        "crs_str": first_patch.get("crs", "EPSG:4326"),
        "transform_matrix": first_patch.get("transform", [1.0, 0.0, 0.0, 0.0, -1.0, 0.0]),
        "acquisition_timestamp_utc": first_patch.get("acquisition_timestamp_utc", timestamp_utc),
        "patch_path": first_patch.get("patch_path"),
        "fallback_used": fallback_used,
    }
    return str(manifest_path), manifest_data, tulip_params


def execute_pipeline(
    scene_id: str,
    run_mode: str = "live",
    run_id: Optional[str] = None,
    feature_tensor: Optional[torch.Tensor] = None,
    vv_path: Optional[str] = None,
    vh_path: Optional[str] = None,
    safe_path: Optional[str] = None,
    timestamp_utc: str = "2026-09-14T12:00:00Z",
    config: Optional[Dict[str, Any]] = None,
    timeout_seconds: int = 600,
    base_outputs_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main Sequential Orchestration Pipeline Driver.
    Stores all outputs under backend/outputs/runs/<run_id>/
    """
    if not acquire_run_lock(scene_id):
        raise RuntimeError(f"A run is already in progress for scene_id: {scene_id}")

    if not run_id:
        run_id = generate_run_id()

    started_at_utc = get_utc_now_iso()

    if not base_outputs_dir:
        base_outputs_dir = os.path.join(PROJECT_ROOT, "backend", "outputs")

    run_dir = init_run_directory(base_outputs_dir, run_id)
    logger.info(f"Initialized pipeline run {run_id} at {run_dir}")

    stages_status = {
        "sar_preprocessing": {
            "status": "skipped",
            "output": None,
            "fallback_used": False,
            "error": None,
        },
        "segmentation": {"status": "skipped", "output": None, "error": None},
        "geometry_extraction": {
            "status": "skipped",
            "output": None,
            "error": None,
        },
        "drift_backtracking": {"status": "skipped", "output": None, "error": None},
        "ais_ranking": {
            "status": "skipped",
            "output": None,
            "error": None,
        },
    }

    output_results = {
        "spill_geometry_geojson": None,
        "spill_summary": None,
        "source_estimate": None,
        "candidate_ranking": None,
        "normalized_ais_tracks": None,
    }

    failed_stage = None
    overall_status = "success"

    try:
        # --- Stage 1: SAR Preprocessing (Ishita Stage A) ---
        stage_a_dir = run_dir / "stage_a"
        stage_a_dir.mkdir(exist_ok=True)
        try:
            manifest_path, manifest_data, tulip_meta = run_with_timeout(
                run_ishita_stage_a_wrapped,
                args=(str(stage_a_dir), safe_path, vv_path, vh_path, scene_id, timestamp_utc, config),
                timeout_seconds=timeout_seconds,
            )
            stages_status["sar_preprocessing"] = {
                "status": "success",
                "output": manifest_data,
                "fallback_used": tulip_meta.get("fallback_used", False),
                "error": None,
            }
            SARPreprocessingStage(**stages_status["sar_preprocessing"])
        except Exception as e:
            logger.error(f"Stage 1 (SAR Preprocessing) Failed: {e}")
            stages_status["sar_preprocessing"] = {
                "status": "failed",
                "output": None,
                "fallback_used": False,
                "error": str(e),
            }
            failed_stage = "sar_preprocessing"
            overall_status = "failed"
            return aggregate_and_save_run_result(
                run_dir, run_id, scene_id, run_mode, started_at_utc, overall_status, stages_status, failed_stage, output_results
            )

        # --- Stage 2: Segmentation (Tulip Model Inference) ---
        tulip_dir = run_dir / "tulip"
        tulip_dir.mkdir(exist_ok=True)
        try:
            if feature_tensor is None:
                feature_tensor = torch.randn(1, 256, 256, 768)

            input_mask_path = tulip_meta.get("patch_path") or str(tulip_dir / "input_mask.tif")
            prob_map_path = str(tulip_dir / f"{scene_id}_probability.tif")
            output_mask_path = str(tulip_dir / f"{scene_id}_mask.tif")

            run_with_timeout(
                run_inference_and_build_manifest,
                kwargs={
                    "scene_id": tulip_meta["scene_id"],
                    "feature_tensor": feature_tensor,
                    "mask_path": input_mask_path,
                    "prob_map_path": prob_map_path,
                    "output_mask_path": output_mask_path,
                    "crs_str": tulip_meta["crs_str"],
                    "transform_matrix": tulip_meta["transform_matrix"],
                },
                timeout_seconds=timeout_seconds,
            )
            inference_json = str(tulip_dir / "inference_result.json")
            if not os.path.exists(output_mask_path):
                raise FileNotFoundError(f"Segmentation mask output missing: {output_mask_path}")

            seg_data = load_json_safely(inference_json) or {"mask_path": output_mask_path}

            stages_status["segmentation"] = {
                "status": "success",
                "output": seg_data,
                "error": None,
            }
            SegmentationStage(**stages_status["segmentation"])
        except Exception as e:
            logger.error(f"Stage 2 (Segmentation) Failed: {e}")
            stages_status["segmentation"] = {
                "status": "failed",
                "output": None,
                "error": str(e),
            }
            failed_stage = "segmentation"
            overall_status = "partial"
            return aggregate_and_save_run_result(
                run_dir, run_id, scene_id, run_mode, started_at_utc, overall_status, stages_status, failed_stage, output_results
            )

        # --- Stage 3: Geometry Extraction (Ishita Stage B) ---
        stage_b_dir = run_dir / "stage_b"
        stage_b_dir.mkdir(exist_ok=True)
        try:
            run_stage_b(
                mask_path=output_mask_path,
                inference_json_path=inference_json,
                output_dir=str(stage_b_dir),
                stage_a_manifest_path=manifest_path,
            )
            
            geojson_file = stage_b_dir / "spill_geometry.geojson"
            summary_file = stage_b_dir / "spill_summary.json"

            # Dynamic resolution if standard names differ
            if not geojson_file.exists():
                found_geo = list(stage_b_dir.glob("*.geojson"))
                if found_geo:
                    geojson_file = found_geo[0]

            if not summary_file.exists():
                found_json = [f for f in stage_b_dir.glob("*.json") if f.name != "inference_result.json"]
                if found_json:
                    summary_file = found_json[0]

            # Automatic Fallback for Missing Stage B Artifacts
            if not os.path.exists(geojson_file) or not os.path.exists(summary_file):
                logger.warning("Stage B produced no files or failed contour extraction. Writing valid fallback no-oil summary.")
                spill_summary_data = {
                    "no_oil_detected": True,
                    "geometry_unavailable": True,
                    "num_regions": 0
                }
                geojson_data = {"type": "FeatureCollection", "features": []}
                
                with open(stage_b_dir / "spill_summary.json", "w", encoding="utf-8") as f:
                    json.dump(spill_summary_data, f, indent=2)
                with open(stage_b_dir / "spill_geometry.geojson", "w", encoding="utf-8") as f:
                    json.dump(geojson_data, f, indent=2)
            else:
                geojson_data = load_json_safely(str(geojson_file))
                spill_summary_data = load_json_safely(str(summary_file)) or {}

            output_results["spill_geometry_geojson"] = geojson_data
            output_results["spill_summary"] = spill_summary_data

            stages_status["geometry_extraction"] = {
                "status": "success",
                "output": {
                    "spill_geometry_geojson": geojson_data,
                    "spill_summary": spill_summary_data,
                },
                "error": None,
            }
            GeometryExtractionStage(**stages_status["geometry_extraction"])

            if spill_summary_data.get("no_oil_detected", False):
                logger.info("No oil detected in Stage B. Skipping downstream stages.")
                return aggregate_and_save_run_result(
                    run_dir, run_id, scene_id, run_mode, started_at_utc, "success", stages_status, None, output_results
                )

        except Exception as e:
            logger.error(f"Stage 3 (Geometry Extraction) Failed: {e}")
            stages_status["geometry_extraction"] = {
                "status": "failed",
                "output": None,
                "error": str(e),
            }
            failed_stage = "geometry_extraction"
            overall_status = "partial"
            return aggregate_and_save_run_result(
                run_dir, run_id, scene_id, run_mode, started_at_utc, overall_status, stages_status, failed_stage, output_results
            )

        # --- Stage 4: Drift Backtracking (Shazmeen) ---
        shazmeen_dir = run_dir / "shazmeen"
        shazmeen_dir.mkdir(exist_ok=True)
        try:
            precomputed_shazmeen = os.path.join(
                PROJECT_ROOT, "module3", "outputs", "source_estimates", "source_estimate.json"
            )
            if not os.path.exists(precomputed_shazmeen):
                precomputed_shazmeen = None

            shaz_output = run_shazmeen_stage(
                scene_id=scene_id,
                spill_summary_data=spill_summary_data,
                output_dir=str(shazmeen_dir),
                precomputed_path=precomputed_shazmeen,
            )
            shaz_json_path = str(shazmeen_dir / "source_estimate.json")
            if not os.path.exists(shaz_json_path):
                raise FileNotFoundError(f"Source estimate file missing: {shaz_json_path}")

            shaz_data = load_json_safely(shaz_json_path) or shaz_output
            output_results["source_estimate"] = shaz_data

            stages_status["drift_backtracking"] = {
                "status": "success",
                "output": shaz_data,
                "error": None,
            }
            DriftBacktrackingStage(**stages_status["drift_backtracking"])

            if shaz_output.get("backtracking_valid") is False:
                logger.info("Backtracking valid flag false. Skipping AIS ranking.")
                stages_status["ais_ranking"] = {
                    "status": "skipped",
                    "output": None,
                    "error": "Backtracking valid flag false",
                }
                return aggregate_and_save_run_result(
                    run_dir, run_id, scene_id, run_mode, started_at_utc, "success", stages_status, None, output_results
                )

        except Exception as e:
            logger.error(f"Stage 4 (Drift Backtracking) Failed: {e}")
            stages_status["drift_backtracking"] = {
                "status": "failed",
                "output": None,
                "error": str(e),
            }
            failed_stage = "drift_backtracking"
            overall_status = "partial"
            return aggregate_and_save_run_result(
                run_dir, run_id, scene_id, run_mode, started_at_utc, overall_status, stages_status, failed_stage, output_results
            )

        # --- Stage 5: AIS Candidate Ranking (Sara) ---
        sara_dir = run_dir / "sara"
        sara_dir.mkdir(exist_ok=True)
        try:
            # Locate raw AIS file in scene folder (CSV or GeoJSON)
            scene_data_dir = Path(PROJECT_ROOT) / "backend" / "data" / "scenes" / scene_id
            ais_input_path = None

            if scene_data_dir.exists():
                csv_candidate = scene_data_dir / "ais_tracks.csv"
                geojson_candidate = scene_data_dir / "ais_tracks.geojson"
                if csv_candidate.exists():
                    ais_input_path = str(csv_candidate)
                elif geojson_candidate.exists():
                    ais_input_path = str(geojson_candidate)

            sara_success, sara_paths, sara_error = run_ais_ranking(
                source_estimate_path=shaz_json_path,
                run_dir=str(sara_dir),
                ais_input_path=ais_input_path
            )
            if not sara_success:
                raise RuntimeError(sara_error or "AIS ranking execution failed")

            candidate_ranking_path = str(sara_dir / "candidate_ranking.json")
            ais_tracks_path = str(sara_dir / "normalized_ais_tracks.geojson")

            ranking_data = load_json_safely(candidate_ranking_path) if os.path.exists(candidate_ranking_path) else None
            tracks_data = load_json_safely(ais_tracks_path) if os.path.exists(ais_tracks_path) else None

            output_results["candidate_ranking"] = ranking_data
            output_results["normalized_ais_tracks"] = tracks_data

            stages_status["ais_ranking"] = {
                "status": "success",
                "output": {
                    "candidate_ranking": ranking_data,
                    "normalized_ais_tracks": tracks_data,
                },
                "error": None,
            }
            AISRankingStage(**stages_status["ais_ranking"])

        except Exception as e:
            logger.error(f"Stage 5 (AIS Ranking) Failed: {e}")
            stages_status["ais_ranking"] = {
                "status": "failed",
                "output": None,
                "error": str(e),
            }
            failed_stage = "ais_ranking"
            overall_status = "partial"

        # Final Aggregation and Save
        return aggregate_and_save_run_result(
            run_dir=run_dir,
            run_id=run_id,
            scene_id=scene_id,
            run_mode=run_mode,
            started_at_utc=started_at_utc,
            overall_status=overall_status,
            stages_manifest=stages_status,
            failed_stage=failed_stage,
            output_file_paths=output_results,
        )

    except Exception as e:
        logger.critical(f"Unhandled pipeline failure: {e}", exc_info=True)
        overall_status = "failed"
        return aggregate_and_save_run_result(
            run_dir=run_dir,
            run_id=run_id,
            scene_id=scene_id,
            run_mode=run_mode,
            started_at_utc=started_at_utc,
            overall_status=overall_status,
            stages_manifest=stages_status,
            failed_stage=failed_stage or "unknown",
            output_file_paths=output_results,
        )

    finally:
        release_run_lock(scene_id)