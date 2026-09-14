"""
Fetch and cache the bounded daily-mean GLORYS12V1 uo/vo window required by
Module 3. Mirrors fetch_era5.py's validation/caching discipline (spec §21:
"don't rely on live API calls during development or the demo itself";
§11/§15: reject/flag insufficient coverage rather than silently proceeding).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)

GLORYS_DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
GLORYS_VARIABLES = ("uo", "vo")


def _validate_utc(dt: datetime, name: str) -> datetime:
    if not isinstance(dt, datetime):
        raise TypeError(f"{name} must be a datetime")
    if dt.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware and explicitly UTC")
    return dt.astimezone(timezone.utc)


def _find_coord_name(ds: xr.Dataset, candidates) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    raise ValueError(f"Could not find any of the expected coordinates: {candidates}. "
                      f"Available coordinates: {list(ds.coords)}")


def _validate_variables(ds: xr.Dataset) -> None:
    missing = set(GLORYS_VARIABLES).difference(ds.data_vars)
    if missing:
        raise ValueError(f"GLORYS dataset is missing required variables: {sorted(missing)}. "
                          f"Available variables: {list(ds.data_vars)}")


def _validate_finite_values(ds: xr.Dataset) -> None:
    """Reject NaN/Inf in the requested current fields — matches fetch_era5's
    equivalent check. Note GLORYS legitimately has NaN over land (it's an
    ocean-only product), so this validates the raw fetched extent is at
    least internally consistent; per-point NaN/Inf handling at the actual
    trajectory location is interpolation.py's job (nearest-valid fallback),
    not this fetch-time check's."""
    for variable in GLORYS_VARIABLES:
        values = ds[variable].values
        if values.size == 0:
            continue
        # GLORYS legitimately has NaN over land (it's an ocean-only
        # product) — only Inf is a genuine problem here. Per-point NaN
        # handling at the actual trajectory location is interpolation.py's
        # job (bounded nearest-valid fallback), not this fetch-time check's.
        finite_or_nan = np.isnan(values) | np.isfinite(values)
        if not finite_or_nan.all():
            raise ValueError(f"GLORYS variable {variable} contains Inf values")


def _validate_coverage(ds: xr.Dataset, bbox, start_utc: datetime, end_utc: datetime) -> None:
    west, south, east, north = bbox
    lat_name = _find_coord_name(ds, ("latitude", "lat"))
    lon_name = _find_coord_name(ds, ("longitude", "lon"))
    time_name = _find_coord_name(ds, ("time",))

    lat_values = np.asarray(ds[lat_name].values)
    lon_values = np.asarray(ds[lon_name].values)
    if lat_values.size == 0 or lon_values.size == 0:
        raise ValueError("GLORYS response contains an empty spatial grid")

    actual_south, actual_north = float(np.nanmin(lat_values)), float(np.nanmax(lat_values))
    actual_west, actual_east = float(np.nanmin(lon_values)), float(np.nanmax(lon_values))
    coverage_tolerance_deg = 0.25
    if (
        actual_south > south + coverage_tolerance_deg
        or actual_north < north - coverage_tolerance_deg
        or actual_west > west + coverage_tolerance_deg
        or actual_east < east - coverage_tolerance_deg
    ):
        raise ValueError(
            "GLORYS spatial coverage is insufficient. "
            f"Requested bbox={bbox}; "
            f"actual coverage=("
            f"{actual_west}, {actual_south}, "
            f"{actual_east}, {actual_north})"
        )

    time_values = np.asarray(ds[time_name].values)
    if time_values.size == 0:
        raise ValueError("GLORYS response contains no time values")
    actual_start, actual_end = np.datetime64(time_values.min()), np.datetime64(time_values.max())
    requested_start = np.datetime64(start_utc.replace(tzinfo=None))
    requested_end = np.datetime64(end_utc.replace(tzinfo=None))
    # GLORYS is daily-mean (spec §12/§20): a day's value is only usable for
    # hourly steps that fall within it, so allow up to ~1 day of slack on
    # each side rather than requiring an exact-timestamp match.
    slack = np.timedelta64(1, "D")
    if actual_start > requested_start + slack or actual_end < requested_end - slack:
        raise ValueError(
            f"GLORYS temporal coverage is insufficient. Requested={start_utc.isoformat()} "
            f"to {end_utc.isoformat()}; actual={actual_start} to {actual_end}"
        )

    LOGGER.info("GLORYS coverage validated: requested bbox=%s, actual bbox=(%.4f, %.4f, %.4f, %.4f), "
                "requested time=%s to %s, actual time=%s to %s",
                bbox, actual_west, actual_south, actual_east, actual_north,
                start_utc.isoformat(), end_utc.isoformat(), actual_start, actual_end)


def fetch_glorys(*, output_path, bbox, start_utc, end_utc, username=None, password=None, overwrite=False):
    """Fetch the bounded daily GLORYS12V1 uo/vo window for Module 3.

    Args:
        output_path: local NetCDF cache path.
        bbox: (west, south, east, north) in EPSG:4326.
        start_utc, end_utc: timezone-aware UTC datetimes.
        username/password: optional copernicusmarine credentials (falls
            back to the library's own configured/cached login if omitted).
        overwrite: if False and output_path already exists, validate and
            reuse the cached file rather than re-fetching (spec §21).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_utc = _validate_utc(start_utc, "start_utc")
    end_utc = _validate_utc(end_utc, "end_utc")
    if end_utc < start_utc:
        raise ValueError("end_utc must be >= start_utc")

    if output_path.exists() and not overwrite:
        LOGGER.info("Using cached GLORYS file: %s", output_path)
        with xr.open_dataset(output_path) as cached:
            _validate_variables(cached)
            _validate_coverage(cached, bbox, start_utc, end_utc)
            _validate_finite_values(cached)
        return output_path

    import copernicusmarine

    LOGGER.info("GLORYS request: bbox=%s, start=%s, end=%s", bbox, start_utc.isoformat(), end_utc.isoformat())

    kwargs = dict(
        dataset_id=GLORYS_DATASET_ID,
        variables=list(GLORYS_VARIABLES),
        minimum_longitude=bbox[0],
        maximum_longitude=bbox[2],
        minimum_latitude=bbox[1],
        maximum_latitude=bbox[3],
        start_datetime=start_utc.isoformat(),
        end_datetime=end_utc.isoformat(),
        output_filename=str(output_path),
    )
    if username is not None:
        kwargs["username"] = username
    if password is not None:
        kwargs["password"] = password

    copernicusmarine.subset(**kwargs)

    if not output_path.exists():
        raise RuntimeError(f"copernicusmarine reported success but file was not created: {output_path}")

    with xr.open_dataset(output_path) as fetched:
        _validate_variables(fetched)
        _validate_coverage(fetched, bbox, start_utc, end_utc)
        _validate_finite_values(fetched)

    LOGGER.info("GLORYS fetch completed successfully: %s", output_path)
    return output_path
