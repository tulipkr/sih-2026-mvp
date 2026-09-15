import pytest
from pydantic import ValidationError
import os,sys

# 1. ALWAYS modify sys.path BEFORE importing custom local modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",  ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from backend.src.schemas import (
    SARPreprocessingStage,
    SegmentationStage,
    GeometryExtractionStage,
    DriftBacktrackingStage,
    AISRankingStage,
    PipelineStages,
    AggregateRunResult,
)


def test_valid_stage_models():
    sar_stage = SARPreprocessingStage(status="success", output_path="/path/to/manifest.json")
    assert sar_stage.status == "success"

    geom_stage = GeometryExtractionStage(
        status="success",
        output_paths={
            "spill_geometry_geojson": "/path/to/geo.geojson",
            "spill_summary": "/path/to/summary.json",
        },
    )
    assert geom_stage.output_paths.spill_geometry_geojson == "/path/to/geo.geojson"


def test_missing_required_fields_raises_validation_error():
    with pytest.raises(ValidationError):
        # Missing 'status' field
        SARPreprocessingStage(output_path="/path/to/manifest.json")