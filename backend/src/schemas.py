from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field


class SARPreprocessingStage(BaseModel):
    status: str = Field(..., description="Stage status: success, failed, skipped")
    output: Optional[Union[List[Dict[str, Any]], Dict[str, Any], str]] = Field(
        None, description="Actual output content or metadata from stage A manifest"
    )
    fallback_used: bool = Field(False, description="Spec §8 required indicator for land mask fallback")
    error: Optional[str] = Field(None, description="Error message if failed")


class SegmentationStage(BaseModel):
    status: str = Field(..., description="Stage status: success, failed, skipped")
    output: Optional[Union[Dict[str, Any], str]] = Field(None, description="Actual segmentation mask metadata or content")
    error: Optional[str] = Field(None, description="Error message if failed")


class GeometryOutputData(BaseModel):
    spill_geometry_geojson: Optional[Dict[str, Any]] = None
    spill_summary: Optional[Dict[str, Any]] = None


class GeometryExtractionStage(BaseModel):
    status: str = Field(..., description="Stage status: success, failed, skipped")
    output: Optional[Union[GeometryOutputData, Dict[str, Any]]] = Field(None, description="Actual extracted GeoJSON geometry and summary data")
    error: Optional[str] = Field(None, description="Error message if failed")


class DriftBacktrackingStage(BaseModel):
    status: str = Field(..., description="Stage status: success, failed, skipped")
    output: Optional[Union[Dict[str, Any], str]] = Field(None, description="Actual source estimation output data")
    error: Optional[str] = Field(None, description="Error message if failed")


class AISOutputData(BaseModel):
    candidate_ranking: Optional[Dict[str, Any]] = None
    normalized_ais_tracks: Optional[Dict[str, Any]] = None


class AISRankingStage(BaseModel):
    status: str = Field(..., description="Stage status: success, failed, skipped")
    output: Optional[Union[AISOutputData, Dict[str, Any]]] = Field(None, description="Actual candidate rankings and AIS trajectories")
    error: Optional[str] = Field(None, description="Error message if failed")


class PipelineStages(BaseModel):
    sar_preprocessing: SARPreprocessingStage
    segmentation: SegmentationStage
    geometry_extraction: GeometryExtractionStage
    drift_backtracking: DriftBacktrackingStage
    ais_ranking: AISRankingStage


class AggregateRunResult(BaseModel):
    run_id: str
    scene_id: str
    run_mode: str
    started_at_utc: str
    completed_at_utc: Optional[str] = None
    overall_status: str
    stages: PipelineStages
    failed_stage: Optional[str] = None
    results: Dict[str, Any]