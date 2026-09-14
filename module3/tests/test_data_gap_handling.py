from pathlib import Path
import sys
import xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.source_estimation_pipeline import run_pipeline


def test_environmental_data_gap_is_explicit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    era5_gap = tmp_path / "era5_gap.nc"
    glorys_gap = tmp_path / "glorys_gap.nc"
    with xr.open_dataset(root/"data_cache/era5/era5_demo.nc") as ds:
        ds.isel(time=slice(0, 2)).to_netcdf(era5_gap)
    with xr.open_dataset(root/"data_cache/glorys/glorys_demo.nc") as ds:
        ds.isel(time=slice(0, 1)).to_netcdf(glorys_gap)
    result = run_pipeline(root/"inputs/spill_summary.json", root/"inputs/spill_geometry.geojson",
                          era5_gap, glorys_gap, root/"configs/config.yaml", tmp_path/"out.json")
    assert result["backtracking_valid"] is False
    assert result["invalid_reason"] == "environmental_data_unavailable"
