"""Builds a minimal, structurally-real fake Sentinel-1 SAFE product for
tests — same directory layout and XML element names as a real product
(annotation/calibration/*.xml with calibrationVectorList, annotation/*.xml
with adsHeader/startTime), so tests exercise the real parsing code, not a
simplified stand-in format."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine


def _write_calibration_xml(path: Path, height: int, width: int, sigma_naught_value: float):
    """A real product publishes the LUT on a coarse (line, pixel) grid —
    here it's constant everywhere, at a known value, so calibrated output
    has a hand-computable expected value (sigma0 = DN^2 / value^2)."""
    lines = [0, height // 2, height - 1]
    pixels = list(range(0, width, max(1, width // 5))) + [width - 1]
    pixels = sorted(set(pixels))
    vectors = ""
    for line in lines:
        pixel_str = " ".join(str(p) for p in pixels)
        sigma_str = " ".join(str(sigma_naught_value) for _ in pixels)
        vectors += f"""
    <calibrationVector>
      <azimuthTime>2021-10-03T01:49:27.000000</azimuthTime>
      <line>{line}</line>
      <pixel count="{len(pixels)}">{pixel_str}</pixel>
      <sigmaNought count="{len(pixels)}">{sigma_str}</sigmaNought>
    </calibrationVector>"""
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<calibration>
  <calibrationVectorList count="{len(lines)}">{vectors}
  </calibrationVectorList>
</calibration>"""
    path.write_text(xml)


def _write_annotation_xml(path: Path, start_time: str, stop_time: str):
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <adsHeader>
    <startTime>{start_time}</startTime>
    <stopTime>{stop_time}</stopTime>
  </adsHeader>
</product>"""
    path.write_text(xml)


def build_fake_safe_product(
    root: Path,
    product_id: str = "S1B_IW_GRDH_1SDV_20211003T014927_20211003T014952_028965_0374DA_3EE4",
    height: int = 512,
    width: int = 512,
    sigma_naught_value: float = 237.87,
    dn_value: float = 500.0,
    include_vh: bool = True,
    include_calibration: bool = True,
    valid_georeferencing: bool = True,
    seed: int = 0,
) -> Path:
    """Returns the path to the .SAFE directory it built."""
    safe_root = root / f"{product_id}.SAFE"
    (safe_root / "measurement").mkdir(parents=True, exist_ok=True)
    (safe_root / "annotation" / "calibration").mkdir(parents=True, exist_ok=True)
    (safe_root / "manifest.safe").write_text("<xfdu:XFDU></xfdu:XFDU>")

    rng = np.random.RandomState(seed)
    transform = (
        Affine(10.0, 0, 500000, 0, -10.0, 3700000) if valid_georeferencing
        else Affine.identity()
    )
    crs = "EPSG:32611" if valid_georeferencing else None  # UTM 11N, real zone for Southern California

    dn = np.full((height, width), dn_value, dtype=np.float32) + rng.normal(0, 5, (height, width)).astype(np.float32)
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": rasterio.float32, "crs": crs, "transform": transform,
    }
    vv_name = f"s1b-iw-grd-vv-{product_id.split('_')[4].lower()}-002.tiff"
    with rasterio.open(safe_root / "measurement" / vv_name, "w", **profile) as dst:
        dst.write(dn, 1)

    if include_vh:
        vh_name = f"s1b-iw-grd-vh-{product_id.split('_')[4].lower()}-001.tiff"
        with rasterio.open(safe_root / "measurement" / vh_name, "w", **profile) as dst:
            dst.write(dn * 0.3, 1)

    ann_vv = safe_root / "annotation" / vv_name.replace(".tiff", ".xml")
    _write_annotation_xml(ann_vv, "2021-10-03T01:49:27.000000", "2021-10-03T01:49:52.000000")
    if include_vh:
        ann_vh = safe_root / "annotation" / vh_name.replace(".tiff", ".xml")
        _write_annotation_xml(ann_vh, "2021-10-03T01:49:27.000000", "2021-10-03T01:49:52.000000")

    if include_calibration:
        _write_calibration_xml(
            safe_root / "annotation" / "calibration" / f"calibration-{vv_name}".replace(".tiff", ".xml"),
            height, width, sigma_naught_value,
        )
        if include_vh:
            _write_calibration_xml(
                safe_root / "annotation" / "calibration" / f"calibration-{vh_name}".replace(".tiff", ".xml"),
                height, width, sigma_naught_value,
            )

    return safe_root
