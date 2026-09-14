"""
Regression test for the 'valid_time' dimension bug: recent CDS ERA5
downloads deliver 'valid_time' as the real per-timestep DIMENSION (not a
'time' dimension), with a different size per daily file depending on how
many hours were requested that day. Concatenating several such files used
to raise:

    AlignmentError: cannot reindex or align along dimension 'valid_time'
    because of conflicting dimension sizes: {24, 2, 23}

because xr.concat(..., dim="time") was told to concatenate along a "time"
dimension that didn't exist in the inputs at all, causing xarray to try
(and fail) to align the real, differently-sized 'valid_time' dimension
instead. This test reproduces the exact reported shapes (23, 24, 2 hourly
steps for three daily files) without needing real CDS access.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from src.fetch_era5 import _prepare_dataset


def _make_day_dataset(day: datetime, n_hours: int, start_hour: int = 0):
    """Builds one day's worth of a synthetic ERA5-like dataset using
    'valid_time' as the REAL dimension name -- reproducing the new CDS
    backend's actual format, not the legacy 'time'-named format."""
    times = [day + timedelta(hours=start_hour + h) for h in range(n_hours)]
    lat = np.array([10.0, 11.0])
    lon = np.array([80.0, 81.0])
    shape = (n_hours, len(lat), len(lon))
    return xr.Dataset(
        {
            "u10": (("valid_time", "latitude", "longitude"), np.full(shape, 3.0, dtype="f4")),
            "v10": (("valid_time", "latitude", "longitude"), np.full(shape, 1.0, dtype="f4")),
        },
        coords={"valid_time": times, "latitude": lat, "longitude": lon},
    )


def test_prepare_dataset_combines_valid_time_dimensioned_files_with_different_sizes():
    """Exact reported scenario: three daily files with 23, 24, and 2 hourly
    steps respectively, each using 'valid_time' as the real dimension."""
    day1 = datetime(2021, 10, 1, tzinfo=timezone.utc)
    day2 = datetime(2021, 10, 2, tzinfo=timezone.utc)
    day3 = datetime(2021, 10, 3, tzinfo=timezone.utc)

    ds1 = _make_day_dataset(day1, n_hours=23, start_hour=1)   # 2021-10-01 01:00-23:00
    ds2 = _make_day_dataset(day2, n_hours=24, start_hour=0)   # 2021-10-02 00:00-23:00
    ds3 = _make_day_dataset(day3, n_hours=2, start_hour=0)    # 2021-10-03 00:00-01:00

    start_utc = day1 + timedelta(hours=1)
    end_utc = day3 + timedelta(hours=1)
    bbox = (80.0, 10.0, 81.0, 11.0)  # exactly matches the synthetic grid extent below

    combined = _prepare_dataset([ds1, ds2, ds3], start_utc, end_utc, bbox)

    assert "time" in combined.dims
    assert "valid_time" not in combined.dims
    assert combined.sizes["time"] == 23 + 24 + 2  # no data lost, nothing silently dropped
    assert "u10" in combined.data_vars and "v10" in combined.data_vars
    assert bool(np.isfinite(combined["u10"].values).all())

    # canonical 'time' coordinate must actually be usable the way the rest
    # of the pipeline expects (sortable, sliceable)
    time_values = combined["time"].values
    assert (time_values == np.sort(time_values)).all()


def test_prepare_dataset_still_handles_legacy_time_dimensioned_files():
    """Older CDS responses that already use 'time' as the real dimension
    (with an optional redundant 'valid_time' auxiliary coordinate) must
    keep working exactly as before -- this isn't a one-format-only fix."""
    day1 = datetime(2021, 10, 1, tzinfo=timezone.utc)
    times = [day1 + timedelta(hours=h) for h in range(3)]
    lat = np.array([10.0, 11.0])
    lon = np.array([80.0, 81.0])
    shape = (3, len(lat), len(lon))
    ds = xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), np.full(shape, 3.0, dtype="f4")),
            "v10": (("time", "latitude", "longitude"), np.full(shape, 1.0, dtype="f4")),
        },
        coords={"time": times, "latitude": lat, "longitude": lon,
                "valid_time": ("time", times)},  # redundant auxiliary coord, not a real second dim
    )

    combined = _prepare_dataset([ds], day1, day1 + timedelta(hours=2), (80.0, 10.0, 81.0, 11.0))
    assert "time" in combined.dims
    assert combined.sizes["time"] == 3


def test_prepare_dataset_raises_clearly_if_neither_time_dim_present():
    ds = xr.Dataset(
        {"u10": (("step", "latitude", "longitude"), np.zeros((2, 2, 2), dtype="f4"))},
        coords={"step": [0, 1], "latitude": [10.0, 11.0], "longitude": [80.0, 81.0]},
    )
    with pytest.raises(ValueError, match="time dimension"):
        _prepare_dataset([ds], datetime(2021, 10, 1, tzinfo=timezone.utc),
                         datetime(2021, 10, 1, tzinfo=timezone.utc), (80.0, 10.0, 81.0, 11.0))
