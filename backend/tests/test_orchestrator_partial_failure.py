import os
import json
import pytest
from unittest.mock import patch
import os,sys

# 1. ALWAYS modify sys.path BEFORE importing custom local modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",  ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from backend.src.orchestrator import execute_pipeline


@patch("backend.src.orchestrator.run_ais_ranking")
@patch("backend.src.orchestrator.run_shazmeen_stage")
@patch("backend.src.orchestrator.run_stage_b")
@patch("backend.src.orchestrator.run_inference_and_build_manifest")
@patch("backend.src.orchestrator.run_stage_a")
def test_partial_failure_on_stage_5(
    mock_stage_a, mock_tulip, mock_stage_b, mock_shazmeen, mock_sara, tmp_path
):
    # Mock successful returns for stages 1-4
    mock_stage_a.return_value = str(tmp_path / "manifest.json")
    with open(tmp_path / "manifest.json", "w") as f:
        json.dump([{"scene_id": "00052"}], f)

    mock_tulip.return_value = {}
    
    # Create fake Stage B output files
    stage_b_dir = tmp_path / "runs"
    mock_stage_b.return_value = None

    mock_shazmeen.return_value = {"backtracking_valid": True}

    # Simulate Sara's stage failing
    mock_sara.return_value = (False, None, "AIS service timeout error")

    # Run execution with mocked components
    result = execute_pipeline(
        scene_id="00052",
        run_mode="live",
        base_outputs_dir=str(tmp_path),
    )

    assert result["overall_status"] == "partial"
    assert result["failed_stage"] == "ais_ranking"
    assert result["stages"]["ais_ranking"]["status"] == "failed"
    assert result["stages"]["sar_preprocessing"]["status"] == "success"