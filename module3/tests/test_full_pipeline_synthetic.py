import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.source_estimation_pipeline import run_pipeline


def test_full_pipeline_synthetic(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "source_estimate.json"
    result = run_pipeline(root/"inputs/spill_summary.json", root/"inputs/spill_geometry.geojson",
                          root/"data_cache/era5/era5_demo.nc", root/"data_cache/glorys/glorys_demo.nc",
                          root/"configs/config.yaml", out)
    saved = json.loads(out.read_text())
    required = {"scene_id","region_id","spill_centroid","acquisition_timestamp_utc","backward_window_hours",
                "probable_source_region","probable_source_time_window","uncertainty_method","uncertainty_radius_km",
                "drift_trajectory","environmental_sources_used","no_oil_detected","geometry_unavailable",
                "backtracking_valid","physically_implausible","fallback_used","invalid_reason"}
    assert required <= saved.keys()
    assert saved == result
    assert saved["backtracking_valid"] is True
    assert saved["physically_implausible"] is False
    assert saved["fallback_used"] is False
    assert saved["backward_window_hours"] == 48
    assert len(saved["drift_trajectory"]) == 49
    assert saved["probable_source_region"]["type"] == "Point"
    assert len(saved["probable_source_region"]["coordinates"]) == 2
