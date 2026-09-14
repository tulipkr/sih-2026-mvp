from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import tempfile

import numpy as np
import xarray as xr


LOGGER = logging.getLogger(__name__)


ERA5_DATASET = "reanalysis-era5-single-levels"
ERA5_VARIABLES = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
)


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    """
    Validate bbox in (west, south, east, north) order.
    """
    if len(bbox) != 4:
        raise ValueError(
            "bbox must contain exactly four values: "
            "(west, south, east, north)"
        )

    west, south, east, north = bbox

    if not (-180 <= west <= 180):
        raise ValueError(f"Invalid bbox west longitude: {west}")

    if not (-180 <= east <= 180):
        raise ValueError(f"Invalid bbox east longitude: {east}")

    if not (-90 <= south <= 90):
        raise ValueError(f"Invalid bbox south latitude: {south}")

    if not (-90 <= north <= 90):
        raise ValueError(f"Invalid bbox north latitude: {north}")

    if west >= east:
        raise ValueError(
            "bbox must satisfy west < east. "
            "Dateline-crossing boxes are not supported by this MVP fetcher."
        )

    if south >= north:
        raise ValueError("bbox must satisfy south < north.")


def _validate_utc(dt: datetime, name: str) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if not isinstance(dt, datetime):
        raise TypeError(f"{name} must be a datetime")

    if dt.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware and explicitly UTC")

    dt_utc = dt.astimezone(timezone.utc)

    if dt_utc != dt:
        LOGGER.warning("%s converted to UTC: %s", name, dt_utc.isoformat())

    return dt_utc


def _calendar_days(
    start_utc: datetime,
    end_utc: datetime,
) -> list[datetime]:
    """Return every calendar day touched by the requested window."""
    cursor = start_utc.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    last_day = end_utc.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    days = []

    while cursor <= last_day:
        days.append(cursor)
        cursor += timedelta(days=1)

    return days


def _requested_hours_for_day(
    day: datetime,
    start_utc: datetime,
    end_utc: datetime,
) -> list[str]:
    """
    Return only the hourly timestamps needed from this calendar day.
    """
    day_start = day
    day_end = day + timedelta(hours=23)

    first_hour = max(start_utc, day_start).replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    last_hour = min(end_utc, day_end).replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    if first_hour > last_hour:
        return []

    hours = []
    cursor = first_hour

    while cursor <= last_hour:
        hours.append(cursor.strftime("%H:00"))
        cursor += timedelta(hours=1)

    return hours


def _get_client(cdsapi_config=None):
    """
    Create the CDS API client.

    cdsapi_config may contain normal cdsapi.Client keyword arguments,
    for example:
        {"url": "...", "key": "..."}
    """
    import cdsapi

    if cdsapi_config is None:
        cdsapi_config = {}

    if not isinstance(cdsapi_config, dict):
        raise TypeError("cdsapi_config must be a dictionary or None")

    return cdsapi.Client(**cdsapi_config)


def _find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    """Find a coordinate name from common ERA5 naming conventions."""
    for name in candidates:
        if name in ds.coords:
            return name

    raise ValueError(
        f"Could not find any of the expected coordinates: {candidates}. "
        f"Available coordinates: {list(ds.coords)}"
    )


def _normalise_longitude(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """
    Sort longitude and leave it in the normal ERA5 0..360/-180..180
    representation returned by CDS.

    This function does not alter longitude values because changing the
    coordinate system here could make coverage validation ambiguous.
    """
    return ds.sortby(lon_name)


def _validate_coverage(
    ds: xr.Dataset,
    bbox: tuple[float, float, float, float],
    start_utc: datetime,
    end_utc: datetime,
) -> None:
    """
    Verify that the returned dataset actually covers the requested
    spatial and temporal window.
    """
    west, south, east, north = bbox

    lat_name = _find_coord_name(ds, ("latitude", "lat"))
    lon_name = _find_coord_name(ds, ("longitude", "lon"))
    time_name = _find_coord_name(ds, ("time", "valid_time"))

    lat_values = np.asarray(ds[lat_name].values)
    lon_values = np.asarray(ds[lon_name].values)

    if lat_values.size == 0 or lon_values.size == 0:
        raise ValueError("ERA5 response contains an empty spatial grid")

    actual_south = float(np.nanmin(lat_values))
    actual_north = float(np.nanmax(lat_values))
    actual_west = float(np.nanmin(lon_values))
    actual_east = float(np.nanmax(lon_values))

    if actual_west >= 0 and actual_east > 180:
        requested_west = west % 360
        requested_east = east % 360
    else:
        requested_west = west
        requested_east = east

    # ERA5 returns data on its native grid, so the actual bounds may
    # differ slightly from the requested bbox. Allow one grid-cell
    # tolerance while still rejecting genuinely insufficient coverage.
    coverage_tolerance_deg = 0.25

    if (
        actual_south > south + coverage_tolerance_deg
        or actual_north < north - coverage_tolerance_deg
        or actual_west > requested_west + coverage_tolerance_deg
        or actual_east < requested_east - coverage_tolerance_deg
    ):
        raise ValueError(
            "ERA5 spatial coverage is insufficient. "
            f"Requested bbox={bbox}; "
            f"actual coverage=("
            f"{actual_west}, {actual_south}, "
            f"{actual_east}, {actual_north})"
        )

    time_values = np.asarray(ds[time_name].values)

    if time_values.size == 0:
        raise ValueError("ERA5 response contains no time values")

    actual_start = np.datetime64(time_values.min())
    actual_end = np.datetime64(time_values.max())

    requested_start = np.datetime64(start_utc.replace(tzinfo=None))
    requested_end = np.datetime64(end_utc.replace(tzinfo=None))

    # ERA5 is hourly, while the satellite acquisition timestamp may
# contain minutes/seconds. Allow the returned hourly grid to bracket
# the requested timestamps.
    if actual_start > requested_end or actual_end < requested_start:
        raise ValueError(
            "ERA5 temporal coverage is insufficient. "
            f"Requested={start_utc.isoformat()} to {end_utc.isoformat()}; "
            f"actual={actual_start} to {actual_end}"
        )

    LOGGER.info(
        "ERA5 coverage validated: requested bbox=%s, "
        "actual bbox=(%.4f, %.4f, %.4f, %.4f), "
        "requested time=%s to %s, actual time=%s to %s",
        bbox,
        actual_west,
        actual_south,
        actual_east,
        actual_north,
        start_utc.isoformat(),
        end_utc.isoformat(),
        actual_start,
        actual_end,
    )


def _validate_variables(ds: xr.Dataset) -> None:
    """Ensure the required ERA5 variables are present."""
    required = {"u10", "v10"}

    missing = required.difference(ds.data_vars)

    if missing:
        raise ValueError(
            f"ERA5 dataset is missing required variables: {sorted(missing)}. "
            f"Available variables: {list(ds.data_vars)}"
        )


def _validate_finite_values(ds: xr.Dataset) -> None:
    """
    Reject NaN/Inf in the requested environmental fields.

    Downstream interpolation must not silently consume invalid
    environmental values.
    """
    for variable in ("u10", "v10"):
        values = ds[variable].values

        if not np.isfinite(values).all():
            raise ValueError(
                f"ERA5 variable {variable} contains NaN/Inf values"
            )


def _canonicalize_time_dim(ds: xr.Dataset) -> xr.Dataset:
    """Recent CDS ERA5 downloads deliver 'valid_time' as the actual
    per-timestep DIMENSION (not merely an auxiliary coordinate alongside a
    'time' dimension) — this is a backend/format change on ECMWF's side,
    not something this code requested. Concatenating several such daily
    files with `xr.concat(..., dim="time")` fails, because 'time' isn't a
    real dimension in the inputs at all: xarray invents a new length-1
    'time' axis per file and then tries to align every OTHER dimension
    across files — including 'valid_time', whose size genuinely differs
    per file (23/24/2 hours for a partial/full/partial day), which is
    exactly what raises AlignmentError.

    This canonicalizes whichever real dimension is actually present
    ('time' or 'valid_time') to 'time', per file, BEFORE concatenation —
    so xr.concat is told the truth about which dimension is the real
    per-step axis, rather than being pointed at one that doesn't exist.
    """
    if "time" in ds.dims:
        # Legacy CDS format: a real 'time' dimension is already present.
        # A separate 'valid_time' here (if any) is a redundant auxiliary
        # coordinate, not a second real dimension — drop it defensively
        # so it can't collide with anything downstream, but do NOT touch
        # the real 'time' dimension itself.
        if "valid_time" in ds.coords and "valid_time" not in ds.dims:
            ds = ds.drop_vars("valid_time")
        return ds
    if "valid_time" in ds.dims:
        # New CDS backend format: 'valid_time' IS the real per-step
        # dimension; there is no separate 'time' dimension in this file
        # at all. Rename it so every downstream function (which expects
        # 'time' uniformly, e.g. _validate_coverage's own
        # ("time", "valid_time") fallback search) sees a consistent name
        # regardless of which backend produced the file.
        return ds.rename({"valid_time": "time"})
    raise ValueError(
        f"Could not find a time dimension (checked 'time', 'valid_time') in "
        f"a downloaded ERA5 file. Available dimensions: {list(ds.dims)}"
    )


def _prepare_dataset(
    datasets: list[xr.Dataset],
    start_utc: datetime,
    end_utc: datetime,
    bbox: tuple[float, float, float, float],
) -> xr.Dataset:
    """Combine daily files, trim to exact request, and validate."""
    if not datasets:
        raise ValueError("No ERA5 datasets were downloaded")

    normalised = [_canonicalize_time_dim(ds) for ds in datasets]

    combined = xr.concat(
        normalised,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    combined = combined.sortby("time")

    # Remove duplicate timestamps defensively.
    time_values = combined["time"].values
    _, unique_indices = np.unique(time_values, return_index=True)

    if len(unique_indices) != len(time_values):
        combined = combined.isel(
            time=np.sort(unique_indices)
        )

    time_values = combined["time"].values

    combined = combined.assign_coords(
        time=time_values.astype("datetime64[ns]")
    )

    start_naive = start_utc.replace(tzinfo=None)
    end_naive = end_utc.replace(tzinfo=None)

    combined = combined.sel(
        time=slice(start_naive, end_naive)
    )

    if combined.sizes.get("time", 0) == 0:
        raise ValueError(
            "ERA5 dataset became empty after requested time filtering"
        )

    _validate_variables(combined)
    _validate_coverage(
        combined,
        bbox,
        start_utc,
        end_utc,
    )
    _validate_finite_values(combined)

    combined.attrs.update(
        {
            "source": "ERA5",
            "dataset": ERA5_DATASET,
            "variables": "u10, v10",
            "temporal_resolution": "hourly",
            "spatial_resolution_deg": 0.25,
            "time_reference": "UTC",
            "bbox_west_south_east_north": ",".join(
                str(v) for v in bbox
            ),
            "requested_start_utc": start_utc.isoformat(),
            "requested_end_utc": end_utc.isoformat(),
        }
    )

    return combined


def fetch_era5(
    *,
    output_path,
    bbox,
    start_utc,
    end_utc,
    cdsapi_config=None,
    overwrite=False,
):
    """
    Fetch and cache the bounded hourly ERA5 u10/v10 window required
    by Module 3.

    Parameters
    ----------
    output_path:
        Local NetCDF cache path.

    bbox:
        (west, south, east, north) in EPSG:4326.

    start_utc, end_utc:
        Timezone-aware UTC datetimes.

    cdsapi_config:
        Optional dictionary passed to cdsapi.Client().

    overwrite:
        If False and output_path already exists, validate and reuse
        the cached file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _validate_bbox(bbox)

    start_utc = _validate_utc(start_utc, "start_utc")
    end_utc = _validate_utc(end_utc, "end_utc")

    if end_utc < start_utc:
        raise ValueError("end_utc must be >= start_utc")

    LOGGER.info(
        "ERA5 request: bbox=%s, start=%s, end=%s",
        bbox,
        start_utc.isoformat(),
        end_utc.isoformat(),
    )

    # Cache hit
    
    if output_path.exists() and not overwrite:
        LOGGER.info(
            "Using cached ERA5 file: %s",
            output_path,
        )

        with xr.open_dataset(output_path) as cached:
            _validate_variables(cached)
            _validate_coverage(
                cached,
                bbox,
                start_utc,
                end_utc,
            )
            _validate_finite_values(cached)

        return output_path

    # Live CDS retrieval
 
    client = _get_client(cdsapi_config)

    days = _calendar_days(
        start_utc,
        end_utc,
    )

    with tempfile.TemporaryDirectory(
        prefix="module3_era5_"
    ) as tmp_dir:

        tmp_dir = Path(tmp_dir)
        downloaded_files = []

        for day in days:
            hours = _requested_hours_for_day(
                day,
                start_utc,
                end_utc,
            )

            if not hours:
                continue

            day_path = tmp_dir / f"era5_{day:%Y%m%d}.nc"

            LOGGER.info(
                "Downloading ERA5 day=%s hours=%s "
                "bbox=%s",
                day.strftime("%Y-%m-%d"),
                hours[0] + "..." + hours[-1],
                bbox,
            )

            # CDS area order is:
            # north, west, south, east
            north = bbox[3]
            west = bbox[0]
            south = bbox[1]
            east = bbox[2]

            request = {
                "product_type": "reanalysis",
                "variable": list(ERA5_VARIABLES),
                "year": day.strftime("%Y"),
                "month": day.strftime("%m"),
                "day": day.strftime("%d"),
                "time": hours,
                "area": [
                    north,
                    west,
                    south,
                    east,
                ],
                "format": "netcdf",
            }

            client.retrieve(
                ERA5_DATASET,
                request,
                str(day_path),
            )

            if not day_path.exists():
                raise RuntimeError(
                    f"CDS retrieval reported success but "
                    f"file was not created: {day_path}"
                )

            downloaded_files.append(day_path)

        if not downloaded_files:
            raise RuntimeError(
                "ERA5 retrieval produced no files"
            )

        datasets = [
            xr.open_dataset(
                path,
                engine=None,
            )
            for path in downloaded_files
        ]

        try:
            combined = _prepare_dataset(
                datasets,
                start_utc,
                end_utc,
                bbox,
            )

            # Write atomically so a failed write cannot leave behind
            # a misleading partial cache file.
            temporary_output = output_path.with_suffix(
                output_path.suffix + ".tmp"
            )

            try:
                combined.to_netcdf(
                    temporary_output,
                    engine="netcdf4",
                )

                temporary_output.replace(output_path)

            finally:
                if temporary_output.exists():
                    temporary_output.unlink()

        finally:
            for ds in datasets:
                ds.close()

    LOGGER.info(
        "ERA5 fetch completed successfully: %s",
        output_path,
    )

    return output_path
