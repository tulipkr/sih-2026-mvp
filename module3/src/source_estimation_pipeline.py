import argparse
import json
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from pyproj import Geod

from .backward_advection import backward_trajectory, combine_velocity
from .fetch_era5 import fetch_era5
from .fetch_glorys import fetch_glorys
from .interpolation import bilinear_or_nearest_valid, select_time_slice
from .uncertainty import uncertainty_radius
from .utils import choose_primary_region, load_geojson, load_json, parse_utc, validate_region, validate_summary

LOG = logging.getLogger("module3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
GEOD = Geod(ellps="WGS84")


def _coord_name(ds, names):
    for n in names:
        if n in ds.coords:
            return n
    raise KeyError(f"None of coordinates {names} found")


def _bbox(ds):
    lat = _coord_name(ds, ["latitude", "lat"])
    lon = _coord_name(ds, ["longitude", "lon"])
    return float(ds[lat].min()), float(ds[lat].max()), float(ds[lon].min()), float(ds[lon].max())


def _coverage(ds, lat, lon, start, end, margin, label):
    min_lat, max_lat, min_lon, max_lon = _bbox(ds)
    requested = (lat - margin, lat + margin, lon - margin, lon + margin)

    # ERA5/GLORYS use native grids, so exact bbox containment is too strict.
    # Allow up to 0.25 degrees of edge difference.
    coverage_tolerance_deg = 0.25

    if (
        requested[0] < min_lat - coverage_tolerance_deg
        or requested[1] > max_lat + coverage_tolerance_deg
        or requested[2] < min_lon - coverage_tolerance_deg
        or requested[3] > max_lon + coverage_tolerance_deg
    ):
        raise ValueError(
            f"{label} spatial coverage insufficient: "
            f"requested={requested}, available={(min_lat, max_lat, min_lon, max_lon)}"
        )

    if "time" not in ds.coords:
        raise KeyError(f"{label} dataset has no time coordinate")

    times = np.asarray(ds.time.values)
    if len(times) == 0:
        raise ValueError(f"{label} dataset has no time values")

    t0 = np.datetime64(start.replace(tzinfo=None))
    t1 = np.datetime64(end.replace(tzinfo=None))

    # Environmental products are hourly/daily, while acquisition is
    # second-level. Require temporal overlap rather than exact containment.
    if t0 > times.max() or t1 < times.min():
        raise ValueError(
            f"{label} temporal coverage insufficient: "
            f"requested={t0}..{t1}, available={times.min()}..{times.max()}"
        )

    LOG.info(
        "%s coverage: actual bbox=%s time=%s..%s; "
        "requested bbox=%s time=%s..%s",
        label,
        _bbox(ds),
        times.min(),
        times.max(),
        requested,
        t0,
        t1,
    )

def _first_nc(folder):
    files = sorted(Path(folder).glob("*.nc"))
    if not files:
        return None
    return files[0]


def _ts(dt):
    return dt.isoformat().replace("+00:00", "Z")


def _write_result(result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOG.info("source_estimate.json written to %s", output_path)


def _invalid(summary, region_id=0, acquisition=None, reason="environmental_data_unavailable", centroid=None, fallback_used=False):
    acquisition_text = _ts(acquisition) if acquisition is not None else ""
    centroid_value = list(centroid) if centroid is not None else []
    return {
        "scene_id": str(summary["scene_id"]),
        "region_id": int(region_id),
        "spill_centroid": centroid_value,
        "acquisition_timestamp_utc": acquisition_text,
        "backward_window_hours": 0,
        "probable_source_region": {"type": "Point", "coordinates": [], "radius_km": 0.0},
        "probable_source_time_window": {"start": acquisition_text, "end": acquisition_text},
        "uncertainty_method": "fixed_radius_heuristic",
        "uncertainty_radius_km": 0.0,
        "drift_trajectory": [],
        "environmental_sources_used": {
            "wind": "ERA5", "wind_resolution_deg": 0.25,
            "current": "GLORYS12V1", "current_resolution_deg": 0.0833,
            "current_temporal_resolution": "daily_mean"
        },
        "no_oil_detected": bool(summary["no_oil_detected"]),
        "geometry_unavailable": bool(summary["geometry_unavailable"]),
        "backtracking_valid": False,
        "physically_implausible": False,
        "fallback_used": bool(fallback_used),
        "invalid_reason": reason,
    }


def _distance_km(points):
    total = 0.0
    for a, b in zip(points[:-1], points[1:]):
        _, _, metres = GEOD.inv(a["lon"], a["lat"], b["lon"], b["lat"])
        total += metres / 1000.0
    return total


def _implausible(points, hours):
    return _distance_km(points) > 500.0 * (hours / 24.0)


def _prepare_glorys_surface(ds):
    # Real GLORYS files may include depth/deptht/depthu/depthv. MVP uses the
    # shallowest available level as the surface current.
    depth_names = [n for n in ("depth", "deptht", "depthu", "depthv") if n in ds.dims or n in ds.coords]
    for variable in ("uo", "vo"):
        if variable not in ds:
            raise KeyError(f"GLORYS dataset missing required variable: {variable}")
        da = ds[variable]
        for depth_name in depth_names:
            if depth_name in da.dims:
                da = da.isel({depth_name: 0})
        ds[variable] = da
    return ds


def _has_nearby_nan(da, lat, lon, radius_cells=1):
    if da.ndim != 2:
        return False
    lat_name = _coord_name(da, ["latitude", "lat"])
    lon_name = _coord_name(da, ["longitude", "lon"])
    lats = np.asarray(da[lat_name].values, dtype=float)
    lons = np.asarray(da[lon_name].values, dtype=float)
    i0 = int(np.argmin(np.abs(lats - float(lat))))
    j0 = int(np.argmin(np.abs(lons - float(lon))))
    for i in range(max(0, i0-radius_cells), min(len(lats), i0+radius_cells+1)):
        for j in range(max(0, j0-radius_cells), min(len(lons), j0+radius_cells+1)):
            if not np.isfinite(float(da.isel({lat_name: i, lon_name: j}).values)):
                return True
    return False


def run_pipeline(summary_path, geometry_path, era5_path=None, glorys_path=None, config_path=None, output_path=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    summary = load_json(summary_path)
    validate_summary(summary)

    # Status short-circuits happen before environmental access or advection.
    # Geometry is inspected only as needed to populate the schema fields.
    geo = None
    features = []
    geometry_error = None
    try:
        geo = load_geojson(geometry_path)
        if geo.get("type") != "FeatureCollection":
            raise ValueError("spill_geometry.geojson must be a FeatureCollection")
        features = geo.get("features", [])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        geometry_error = exc

    if summary["no_oil_detected"]:
        LOG.info("no_oil_detected=true; short-circuiting before drift computation")
        region_id = 0; acquisition = None; centroid = None
        if features:
            props = features[0].get("properties", {})
            region_id = props.get("region_id", 0)
            try:
                acquisition = parse_utc(props["acquisition_timestamp_utc"])
                centroid = [float(props["centroid_lat"]), float(props["centroid_lon"])]
            except (KeyError, ValueError, TypeError):
                pass
        result = _invalid(summary, region_id, acquisition, "no_spill_detected", centroid=centroid)
        _write_result(result, output_path)
        return result

    if summary["geometry_unavailable"]:
        LOG.info("geometry_unavailable=true; short-circuiting before drift computation")
        result = _invalid(summary, 0, None, "missing_georeferencing")
        _write_result(result, output_path)
        return result

    if geometry_error is not None:
        raise geometry_error

    primary = choose_primary_region(features)
    props = validate_region(primary)
    scene_id = str(summary["scene_id"])
    region_id = int(props["region_id"])
    lat0, lon0 = float(props["centroid_lat"]), float(props["centroid_lon"])
    acquisition = parse_utc(props["acquisition_timestamp_utc"])
    hours = min(int(config["backward_window_hours"]), int(config["max_backward_window_hours"]))
    dt_hours = float(config["time_step_hours"])
    start = acquisition - timedelta(hours=hours)
    margin = float(config["search_bbox_margin_deg"])
    bbox = (lon0 - margin, lat0 - margin, lon0 + margin, lat0 + margin)

    era5_path = Path(era5_path) if era5_path else None
    glorys_path = Path(glorys_path) if glorys_path else None
    cache_era5 = Path(summary_path).resolve().parents[1] / "data_cache" / "era5"
    cache_glorys = Path(summary_path).resolve().parents[1] / "data_cache" / "glorys"
    cache_era5.mkdir(parents=True, exist_ok=True)
    cache_glorys.mkdir(parents=True, exist_ok=True)

    # Load a cache file only if it actually covers this scene's requested bbox/time.
    # Otherwise fetch a new bounded subset; never silently use a nearby/incomplete cache.
    def _cached_covering(folder, label):
        for candidate in sorted(Path(folder).glob("*.nc")):
            try:
                with xr.open_dataset(candidate) as ds:
                    _coverage(ds, lat0, lon0, start, acquisition, margin, label)
                LOG.info("Using cached %s: %s", label, candidate)
                return candidate
            except (OSError, KeyError, ValueError) as exc:
                LOG.info("Ignoring cached %s %s: %s", label, candidate, exc)
        return None

    if era5_path is None:
        era5_path = _cached_covering(cache_era5, "ERA5")
    if glorys_path is None:
        glorys_path = _cached_covering(cache_glorys, "GLORYS12V1")

    if era5_path is None:
        era5_path = cache_era5 / f"era5_{scene_id}_{acquisition.strftime('%Y%m%dT%H%M%SZ')}.nc"
        LOG.info("ERA5 cache miss; fetching bounded subset to %s", era5_path)
        fetch_era5(output_path=era5_path, bbox=bbox, start_utc=start, end_utc=acquisition, cdsapi_config=None)
    if glorys_path is None:
        glorys_path = cache_glorys / f"glorys_{scene_id}_{acquisition.strftime('%Y%m%dT%H%M%SZ')}.nc"
        LOG.info("GLORYS12V1 cache miss; fetching bounded subset to %s", glorys_path)
        fetch_glorys(output_path=glorys_path, bbox=bbox, start_utc=start, end_utc=acquisition)

    era5 = glorys = None
    try:
        era5 = xr.open_dataset(era5_path)
        glorys = _prepare_glorys_surface(xr.open_dataset(glorys_path))
        _coverage(era5, lat0, lon0, start, acquisition, margin, "ERA5")
        _coverage(glorys, lat0, lon0, start, acquisition, margin, "GLORYS12V1")

        near_boundary_logged = False
        nearest_logged = set()

        def velocity_at(when, lat, lon):
            nonlocal near_boundary_logged
            u10_da = select_time_slice(era5["u10"], when, 1.0)
            v10_da = select_time_slice(era5["v10"], when, 1.0)
            uo_da = select_time_slice(glorys["uo"], when, 24.0)
            vo_da = select_time_slice(glorys["vo"], when, 24.0)

            if not near_boundary_logged and any(_has_nearby_nan(da, lat, lon) for da in (u10_da, v10_da, uo_da, vo_da)):
                LOG.warning("masked environmental cells surround the trajectory point; treating this as a possible land-sea boundary/coastal-data case")
                near_boundary_logged = True

            values = []

            # ERA5 wind
            for name, da in (("ERA5 u10", u10_da), ("ERA5 v10", v10_da)):
                value, used_nearest = bilinear_or_nearest_valid(da, lat, lon)
                if used_nearest:
                    key = (name, when.isoformat(), round(lat, 5), round(lon, 5))
                    if key not in nearest_logged:
                        LOG.warning(
                            "nearest-valid fallback used for %s at %s lat=%.5f lon=%.5f",
                            name, when, lat, lon
                        )
                        nearest_logged.add(key)
                values.append(float(value))

            u10, v10 = values

            # GLORYS is ocean-only. If no valid ocean current is available
            # near the trajectory point, fall back to wind-only drift.
            try:
                uo, used_nearest_uo = bilinear_or_nearest_valid(
                    uo_da, lat, lon, max_radius_cells=8
                )
                vo, used_nearest_vo = bilinear_or_nearest_valid(
                    vo_da, lat, lon, max_radius_cells=8
                )

                if used_nearest_uo or used_nearest_vo:
                    LOG.warning(
                        "nearest-valid ocean current used at %s lat=%.5f lon=%.5f",
                        when, lat, lon
                    )

                uo = float(uo)
                vo = float(vo)

            except ValueError:
                LOG.warning(
                    "No valid GLORYS ocean current near lat=%.5f lon=%.5f; "
                    "using wind-only drift",
                    lat, lon
                )
                uo = 0.0
                vo = 0.0

            u, v = combine_velocity(
                uo, vo, u10, v10, config["wind_drift_factor"]
            )
            if not np.isfinite(u) or not np.isfinite(v):
                raise ValueError("NaN/Inf environmental velocity at interpolation point")
            LOG.info("velocity %s: u=%.4f m/s v=%.4f m/s", when, u, v)
            return u, v

        # The advection module still owns stepping; this callback is passed the
        # environmental time selected by that module.
        trajectory = backward_trajectory(lat0, lon0, acquisition, hours, dt_hours, velocity_at)
        implausible = _implausible(trajectory, hours)
        if implausible:
            LOG.warning("implausible backward trajectory flagged: total_distance_km=%.2f window_hours=%s", _distance_km(trajectory), hours)
        radius = uncertainty_radius(config, hours)
        terminal = trajectory[-1]

        result = {
            "scene_id": scene_id,
            "region_id": region_id,
            "spill_centroid": [lat0, lon0],
            "acquisition_timestamp_utc": _ts(acquisition),
            "backward_window_hours": hours,
            "probable_source_region": {"type": "Point", "coordinates": [terminal["lon"], terminal["lat"]], "radius_km": float(radius)},
            "probable_source_time_window": {"start": terminal["timestamp"], "end": trajectory[0]["timestamp"]},
            "uncertainty_method": config["uncertainty_method"],
            "uncertainty_radius_km": float(radius),
            "drift_trajectory": trajectory,
            "environmental_sources_used": {
                "wind": "ERA5", "wind_resolution_deg": 0.25,
                "current": "GLORYS12V1", "current_resolution_deg": 0.0833,
                "current_temporal_resolution": "daily_mean"
            },
            "no_oil_detected": False,
            "geometry_unavailable": False,
            "backtracking_valid": not implausible,
            "physically_implausible": bool(implausible),
            "fallback_used": False,
            "invalid_reason": "physically_implausible_trajectory" if implausible else None,
        }
        LOG.info("final source=%s uncertainty_radius_km=%.2f", result["probable_source_region"], radius)
    except (OSError, KeyError, ValueError, IndexError, ImportError) as exc:
        LOG.error("Module 3 processing failed: %s", exc)
        result = _invalid(summary, region_id, acquisition, "environmental_data_unavailable")
    finally:
        if era5 is not None:
            era5.close()
        if glorys is not None:
            glorys.close()

    _write_result(result, output_path)
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default=str(root / "inputs" / "spill_summary.json"))
    p.add_argument("--geometry", default=str(root / "inputs" / "spill_geometry.geojson"))
    p.add_argument("--era5", default=None)
    p.add_argument("--glorys", default=None)
    p.add_argument("--config", default=str(root / "configs" / "config.yaml"))
    p.add_argument("--output", default=str(root / "outputs" / "source_estimates" / "source_estimate.json"))
    args = p.parse_args()
    run_pipeline(args.summary, args.geometry, args.era5, args.glorys, args.config, args.output)


if __name__ == "__main__":
    main()
