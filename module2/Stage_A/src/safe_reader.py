"""
Locates and parses a real Sentinel-1 GRD SAFE product from a configurable
local path (directory or .zip) — see Module 2 spec §6/§13/§20.

Nothing here is specific to any one scene/incident. `locate_safe_product()`
takes whatever path the caller configures and works from the real SAFE
internal structure (manifest.safe, annotation/*.xml,
annotation/calibration/*.xml, measurement/*.tiff) rather than any
scene-specific filename.

Never fabricates a timestamp or calibration constant: every function here
either returns a real, sourced value or raises an explicit error naming
what's missing (spec §15/§38/§39).
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


class SafeProductError(ValueError):
    """Raised when a SAFE product path is invalid, incomplete, or unparsable."""


@dataclass
class SafeProduct:
    root: Path                 # the .SAFE directory (a temp extraction dir if input was a .zip)
    product_id: str             # e.g. "S1B_IW_GRDH_1SDV_20211003T014927_...3EE4"
    vv_measurement_path: Path
    vh_measurement_path: Path
    vv_annotation_path: Path
    vh_annotation_path: Path
    vv_calibration_path: Path
    vh_calibration_path: Path
    acquisition_start_utc: str  # ISO8601, parsed from annotation XML — never fabricated
    acquisition_end_utc: str | None = None


_POL_PATTERN = re.compile(r"-(vv|vh)-", re.IGNORECASE)


def _find_by_polarization(paths: list[Path], polarization: str) -> Path:
    """Sentinel-1 filenames always embed polarization as e.g. '-vv-' or
    '-vh-' — this is the real, documented naming convention (not a guess)."""
    matches = [p for p in paths if _POL_PATTERN.search(p.name) and
               f"-{polarization.lower()}-" in p.name.lower()]
    if not matches:
        raise SafeProductError(
            f"No file found for polarization '{polarization.upper()}' among: "
            f"{[p.name for p in paths]}. Sentinel-1 filenames must contain "
            f"'-{polarization.lower()}-' — do not guess which file this is."
        )
    if len(matches) > 1:
        raise SafeProductError(
            f"Multiple files matched polarization '{polarization.upper()}': "
            f"{[p.name for p in matches]} — ambiguous, refusing to silently pick one."
        )
    return matches[0]


def _extract_zip_if_needed(safe_path: Path, extract_to: Path) -> Path:
    """If safe_path is a .zip, extract it and return the resulting .SAFE dir.
    If it's already a .SAFE directory, return it unchanged."""
    if safe_path.is_dir():
        return safe_path
    if safe_path.suffix.lower() == ".zip" and safe_path.is_file():
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(safe_path) as zf:
            zf.extractall(extract_to)
        safe_dirs = [p for p in extract_to.iterdir() if p.is_dir() and p.name.endswith(".SAFE")]
        if not safe_dirs:
            raise SafeProductError(
                f"Extracted {safe_path} but found no *.SAFE directory inside it."
            )
        return safe_dirs[0]
    raise SafeProductError(
        f"SAFE product path does not exist or is neither a directory nor a .zip: "
        f"{safe_path.resolve()}"
    )


def _parse_acquisition_time(annotation_xml_path: Path) -> tuple[str, str | None]:
    """Parses <startTime>/<stopTime> from the product annotation XML's
    <adsHeader> — the real, documented location for this in a Sentinel-1
    annotation file. Raises rather than falling back to a filename guess,
    so a real parse failure is never silently masked."""
    try:
        tree = ET.parse(annotation_xml_path)
    except ET.ParseError as e:
        raise SafeProductError(
            f"Annotation XML at {annotation_xml_path} is not valid XML: {e}"
        ) from e

    root = tree.getroot()
    start_el = root.find(".//adsHeader/startTime")
    stop_el = root.find(".//adsHeader/stopTime")
    if start_el is None or not start_el.text:
        raise SafeProductError(
            f"Could not find <adsHeader><startTime> in {annotation_xml_path}. "
            f"Refusing to fabricate an acquisition timestamp — fix the product "
            f"or supply one via config if this file is genuinely non-standard."
        )
    # Sentinel-1 annotation timestamps look like '2021-10-03T01:49:27.000000'
    # (no explicit 'Z') but are always UTC per the product spec (Module 2 §20).
    raw = start_el.text.strip()
    iso = raw.split(".")[0] + "Z" if "." in raw else (raw if raw.endswith("Z") else raw + "Z")
    stop_iso = None
    if stop_el is not None and stop_el.text:
        raw_stop = stop_el.text.strip()
        stop_iso = raw_stop.split(".")[0] + "Z" if "." in raw_stop else (
            raw_stop if raw_stop.endswith("Z") else raw_stop + "Z"
        )
    return iso, stop_iso


def _derive_product_id(safe_root: Path) -> str:
    name = safe_root.name
    if name.upper().endswith(".SAFE"):
        name = name[: -len(".SAFE")]
    return name


def locate_safe_product(safe_path: str | Path, work_dir: str | Path) -> SafeProduct:
    """
    Args:
        safe_path: path to a *.SAFE directory OR a .zip containing one.
            Configurable per-run — never hardcoded to any scene.
        work_dir: scratch directory for zip extraction, if needed.
    Returns:
        SafeProduct with real, located file paths and a real, parsed timestamp.
    Raises:
        SafeProductError: for anything missing, ambiguous, or unparsable —
        per spec §15, never silently substitutes or guesses.
    """
    safe_path = Path(safe_path)
    work_dir = Path(work_dir)

    if not safe_path.exists():
        raise SafeProductError(f"SAFE product path does not exist: {safe_path.resolve()}")

    safe_root = _extract_zip_if_needed(safe_path, work_dir)

    manifest_safe = safe_root / "manifest.safe"
    if not manifest_safe.exists():
        raise SafeProductError(
            f"{safe_root} does not contain manifest.safe — not a valid Sentinel-1 "
            f"SAFE product structure."
        )

    measurement_dir = safe_root / "measurement"
    annotation_dir = safe_root / "annotation"
    calibration_dir = annotation_dir / "calibration"
    for required_dir, label in [
        (measurement_dir, "measurement/"), (annotation_dir, "annotation/"),
        (calibration_dir, "annotation/calibration/"),
    ]:
        if not required_dir.is_dir():
            raise SafeProductError(f"Expected {label} directory not found under {safe_root}")

    measurement_tiffs = sorted(measurement_dir.glob("*.tif*"))
    if not measurement_tiffs:
        raise SafeProductError(f"No measurement GeoTIFFs found under {measurement_dir}")
    vv_measurement = _find_by_polarization(measurement_tiffs, "vv")
    vh_measurement = _find_by_polarization(measurement_tiffs, "vh")

    annotation_xmls = sorted(p for p in annotation_dir.glob("*.xml"))
    if not annotation_xmls:
        raise SafeProductError(f"No annotation XML files found directly under {annotation_dir}")
    vv_annotation = _find_by_polarization(annotation_xmls, "vv")
    vh_annotation = _find_by_polarization(annotation_xmls, "vh")

    calibration_xmls = sorted(calibration_dir.glob("calibration-*.xml"))
    if not calibration_xmls:
        raise SafeProductError(
            f"No calibration-*.xml files found under {calibration_dir}. Cannot "
            f"calibrate without the real LUT — refusing to use a placeholder "
            f"constant (spec §12/§15/§38)."
        )
    vv_calibration = _find_by_polarization(calibration_xmls, "vv")
    vh_calibration = _find_by_polarization(calibration_xmls, "vh")

    acquisition_start, acquisition_end = _parse_acquisition_time(vv_annotation)

    return SafeProduct(
        root=safe_root,
        product_id=_derive_product_id(safe_root),
        vv_measurement_path=vv_measurement,
        vh_measurement_path=vh_measurement,
        vv_annotation_path=vv_annotation,
        vh_annotation_path=vh_annotation,
        vv_calibration_path=vv_calibration,
        vh_calibration_path=vh_calibration,
        acquisition_start_utc=acquisition_start,
        acquisition_end_utc=acquisition_end,
    )
