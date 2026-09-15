# backend/src/module_adapters/shazmeen_adapter.py

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal

# 1. ALWAYS modify sys.path BEFORE importing custom local modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 2. Local module imports
from module3.src.source_estimation_pipeline import run_pipeline

# 3. Third-party imports
from pydantic import BaseModel, Field, field_validator, ValidationInfo

logger = logging.getLogger("ShazmeenAdapter")

# --- Input Schemas (Stage B: Ishita -> Shazmeen) ---

class PrimarySpillFeatureProperties(BaseModel):
    region_id: int
    centroid_lat: float
    centroid_lon: float
    area_km2: float
    acquisition_timestamp_utc: str

class SpillSummaryInput(BaseModel):
    no_oil_detected: bool
    geometry_unavailable: bool
    num_regions: int

# --- Output Schemas (Shazmeen -> Sara / Orchestrator) ---

class ProbableSourceRegion(BaseModel):
    type: Literal["Point"]
    coordinates: List[float]  # [lon, lat] per GeoJSON standard
    radius_km: float

class ProbableSourceTimeWindow(BaseModel):
    start: str
    end: str

class DriftTrajectoryPoint(BaseModel):
    timestamp: str
    lat: float
    lon: float

class EnvironmentalSourcesUsed(BaseModel):
    wind: str
    wind_resolution_deg: float
    current: str
    current_resolution_deg: float
    current_temporal_resolution: str

class SourceEstimateOutput(BaseModel):
    scene_id: str
    region_id: int
    spill_centroid: List[float]  # [lat, lon]
    acquisition_timestamp_utc: str
    backward_window_hours: int
    probable_source_region: ProbableSourceRegion
    probable_source_time_window: ProbableSourceTimeWindow
    uncertainty_method: Literal['fixed_radius_heuristic', 'perturbation_ensemble']
    uncertainty_radius_km: float
    drift_trajectory: List[DriftTrajectoryPoint]
    environmental_sources_used: EnvironmentalSourcesUsed
    no_oil_detected: bool
    geometry_unavailable: bool
    backtracking_valid: bool
    physically_implausible: bool
    fallback_used: bool
    invalid_reason: Optional[str] = None

    @field_validator("uncertainty_radius_km")
    @classmethod
    def validate_radius_match(cls, v: float, info: ValidationInfo) -> float:
        if "probable_source_region" in info.data:
            region_radius = info.data["probable_source_region"].radius_km
            if v != region_radius:
                raise ValueError(
                    f"uncertainty_radius_km ({v}) does not match probable_source_region.radius_km ({region_radius})"
                )
        return v


# --- Helper Function for Fallbacks ---

def create_fallback_payload(scene_id: str, no_oil: bool, geom_unavail: bool, reason: str) -> SourceEstimateOutput:
    return SourceEstimateOutput(
        scene_id=scene_id,
        region_id=-1,
        spill_centroid=[0.0, 0.0],
        acquisition_timestamp_utc="",
        backward_window_hours=0,
        probable_source_region=ProbableSourceRegion(
            type="Point", coordinates=[0.0, 0.0], radius_km=0.0
        ),
        probable_source_time_window=ProbableSourceTimeWindow(start="", end=""),
        uncertainty_method="fixed_radius_heuristic",
        uncertainty_radius_km=0.0,
        drift_trajectory=[],
        environmental_sources_used=EnvironmentalSourcesUsed(
            wind="ERA5", wind_resolution_deg=0.25, current="GLORYS12V1",
            current_resolution_deg=0.0833, current_temporal_resolution="daily_mean"
        ),
        no_oil_detected=no_oil,
        geometry_unavailable=geom_unavail,
        backtracking_valid=False,
        physically_implausible=False,
        fallback_used=True,
        invalid_reason=reason
    )


# --- Adapter Execution Function ---

def run_shazmeen_stage(
    scene_id: str,
    spill_summary_data: Dict[str, Any],
    output_dir: str,
    spill_feature_properties: Optional[Dict[str, Any]] = None,
    precomputed_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Adapter function called by Bhumika's Orchestrator.
    Executes or loads Shazmeen's drift backtracking module and validates output.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "source_estimate.json")

    # 1. Precomputed / Cached Mode
    if precomputed_path and os.path.exists(precomputed_path):
        logger.info(f"Using precomputed Shazmeen result from: {precomputed_path}")
        with open(precomputed_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        validated_output = SourceEstimateOutput(**raw_data)
        
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(validated_output.model_dump_json(indent=2))
            
        return validated_output.model_dump()

    # 2. Upstream Skipping Logic (no_oil_detected / geometry_unavailable)
    summary = SpillSummaryInput(**spill_summary_data)
    if summary.no_oil_detected or summary.geometry_unavailable:
        logger.info("Upstream flags indicate no oil or unavailable geometry. Generating valid fallback output.")
        reason = "Skipped: Upstream stage reported no oil detected or geometry unavailable."
        fallback_payload = create_fallback_payload(scene_id, summary.no_oil_detected, summary.geometry_unavailable, reason)
        
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(fallback_payload.model_dump_json(indent=2))
            
        return fallback_payload.model_dump()

    # 3. Dynamic Artifact Path Resolution
    run_dir = Path(output_dir).parent
    stage_b_dir = run_dir / "stage_b"

    summary_path = stage_b_dir / "spill_summary.json"
    geometry_path = stage_b_dir / "spill_geometry.geojson"

    # Search dynamically if standard named files don't exist
    if not summary_path.exists():
        found_summaries = [f for f in stage_b_dir.glob("*.json") if "summary" in f.name.lower()]
        if found_summaries:
            summary_path = found_summaries[0]

    if not geometry_path.exists():
        found_geometries = list(stage_b_dir.glob("*.geojson"))
        if found_geometries:
            geometry_path = found_geometries[0]

    # Handle missing inputs gracefully
    if not summary_path.exists() or not geometry_path.exists():
        logger.error(f"Required Stage B inputs missing in {stage_b_dir}. Summary: {summary_path.exists()}, Geometry: {geometry_path.exists()}")
        reason = f"Missing Stage B inputs in {stage_b_dir}"
        fallback_payload = create_fallback_payload(scene_id, summary.no_oil_detected, summary.geometry_unavailable, reason)
        
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(fallback_payload.model_dump_json(indent=2))
            
        return fallback_payload.model_dump()

    # Locate config path relative to PROJECT_ROOT
    config_path = Path(PROJECT_ROOT) / "module3" / "configs" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Module 3 config missing at {config_path}")
        reason = f"Configuration file missing at {config_path}"
        fallback_payload = create_fallback_payload(scene_id, summary.no_oil_detected, summary.geometry_unavailable, reason)
        
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(fallback_payload.model_dump_json(indent=2))
            
        return fallback_payload.model_dump()

    # 4. Live Pipeline Invocation
    logger.info("Executing Shazmeen's live source_estimation_pipeline...")
    try:
        raw_result = run_pipeline(
            summary_path=str(summary_path),
            geometry_path=str(geometry_path),
            config_path=str(config_path),
            output_path=output_file_path,
        )

        # Validate against adapter schema contract
        validated_output = SourceEstimateOutput(**raw_result)
        return validated_output.model_dump()

    except Exception as e:
        logger.error(f"Error during live execution of Module 3: {e}")
        reason = f"Module 3 execution failed: {str(e)}"
        fallback_payload = create_fallback_payload(scene_id, summary.no_oil_detected, summary.geometry_unavailable, reason)
        
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(fallback_payload.model_dump_json(indent=2))
            
        return fallback_payload.model_dump()