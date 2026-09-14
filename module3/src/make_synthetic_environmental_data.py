"""
Generates synthetic ERA5-like and GLORYS12V1-like NetCDF grids for
development/testing, independent of live cdsapi/copernicusmarine access
(spec §34's explicit mock-data-strategy deliverable).

NOT real environmental data — every value here is a fabricated, uniform,
hand-computable field for testing purposes only. Never point Module 3's
config at these files for anything other than tests/demos of the pipeline
mechanics; they carry a "history" attribute saying exactly this, and
every field in source_estimate.json's environmental_sources_used still
correctly names the real datasets Module 3 is *designed* to use, not
these mocks — do not confuse "the pipeline ran successfully against mock
data" with "this used real ERA5/GLORYS data".

This is the "real stack" version of the same files committed under
data_cache/ (which were generated with scipy.io.netcdf_file in the
environment these files were authored in, since xarray/netCDF4 weren't
available there — see the repo's audit notes). Regenerate with this
script instead of the committed files whenever xarray/netCDF4 are
available; the two should produce numerically equivalent grids.

Usage:
    python -m src.make_synthetic_environmental_data
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

# Chennai/Ennore demo scene: acquisition 2026-01-14T00:30:00Z,
# backward_window_hours=48, search_bbox_margin_deg=2.0 -> required coverage
# is roughly lat [11.15,15.15], lon [78.32,82.32], time
# [2026-01-12T00:30Z, 2026-01-14T00:30Z]. Grids below have generous buffer
# on every side.


def make_era5_demo(output_path: Path) -> Path:
    lat = np.arange(10.0, 16.25, 0.25)
    lon = np.arange(78.0, 83.25, 0.25)
    start = datetime(2026, 1, 11, 23, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 14, 1, 0, tzinfo=timezone.utc)
    n_hours = int((end - start).total_seconds() // 3600) + 1
    times = [start + timedelta(hours=h) for h in range(n_hours)]

    # Gentle, uniform, hand-computable wind: u10=3.0 m/s east, v10=1.0 m/s north.
    shape = (len(times), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), np.full(shape, 3.0, dtype="f4"),
                     {"units": "m s**-1", "long_name": "10 metre U wind component"}),
            "v10": (("time", "latitude", "longitude"), np.full(shape, 1.0, dtype="f4"),
                     {"units": "m s**-1", "long_name": "10 metre V wind component"}),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
        attrs={"history": "Synthetic ERA5-like mock data for Module 3 tests - NOT real ERA5 data."},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    return output_path


def make_glorys_demo(output_path: Path) -> Path:
    lat = np.arange(10.0, 16.1, 1 / 12)
    lon = np.arange(78.0, 83.1, 1 / 12)
    day0 = datetime(2026, 1, 11, 12, 0, tzinfo=timezone.utc)  # daily-mean convention: 12:00 UTC representative
    times = [day0 + timedelta(days=d) for d in range(5)]

    # Gentle, uniform, hand-computable current: uo=0.2 m/s east, vo=0.1 m/s north.
    shape = (len(times), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "uo": (("time", "latitude", "longitude"), np.full(shape, 0.2, dtype="f4"),
                    {"units": "m s**-1", "long_name": "eastward_sea_water_velocity"}),
            "vo": (("time", "latitude", "longitude"), np.full(shape, 0.1, dtype="f4"),
                    {"units": "m s**-1", "long_name": "northward_sea_water_velocity"}),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
        attrs={"history": "Synthetic GLORYS12V1-like mock data for Module 3 tests - NOT real GLORYS data."},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    return output_path


def main():
    root = Path(__file__).resolve().parents[1]
    era5_path = make_era5_demo(root / "data_cache" / "era5" / "era5_demo.nc")
    glorys_path = make_glorys_demo(root / "data_cache" / "glorys" / "glorys_demo.nc")
    print(f"Wrote {era5_path}")
    print(f"Wrote {glorys_path}")


if __name__ == "__main__":
    main()
