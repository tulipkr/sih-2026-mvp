"""
Radiometric calibration: raw digital numbers (DN) -> Sigma0 (linear), using
the real per-scene calibration LUT from the product's own
annotation/calibration/calibration-*.xml (Module 2 spec §12/§13/§18).

sigma0 = DN^2 / sigmaNought^2

sigmaNought is only published on a coarse (line, pixel) sub-grid within the
XML — this module bilinearly interpolates it up to full resolution before
dividing, which is the standard, documented approach (matches what ESA SNAP
and other SAR toolchains do), not an approximation invented here.

Never uses a hardcoded "typical" calibration constant — if the LUT can't be
parsed, this raises rather than guessing (spec §15/§38/§39).
"""
from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


class CalibrationError(ValueError):
    pass


def parse_calibration_lut(xml_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (lines, pixels, sigma_grid) where sigma_grid has shape
    (len(lines), len(pixels)) — the real sigmaNought LUT read from the XML,
    at native (unsampled) resolution."""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise CalibrationError(f"Calibration LUT file not found: {xml_path.resolve()}")

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        raise CalibrationError(f"Calibration XML at {xml_path} is not valid XML: {e}") from e

    vectors = tree.getroot().findall(".//calibrationVectorList/calibrationVector")
    if not vectors:
        raise CalibrationError(
            f"No <calibrationVector> entries found in {xml_path} — cannot calibrate "
            f"without a real LUT. Refusing to fall back to a placeholder constant."
        )

    lines, pixel_rows, sigma_rows = [], [], []
    for v in vectors:
        line_el, pixel_el, sigma_el = v.find("line"), v.find("pixel"), v.find("sigmaNought")
        if line_el is None or pixel_el is None or sigma_el is None or not all(
            el.text for el in (line_el, pixel_el, sigma_el)
        ):
            raise CalibrationError(
                f"A <calibrationVector> in {xml_path} is missing line/pixel/sigmaNought — "
                f"malformed calibration data, refusing to skip or interpolate around it."
            )
        lines.append(int(line_el.text))
        pixel_rows.append(np.array([int(x) for x in pixel_el.text.split()], dtype=np.float64))
        sigma_rows.append(np.array([float(x) for x in sigma_el.text.split()], dtype=np.float64))

    lines_arr = np.array(lines, dtype=np.float64)
    order = np.argsort(lines_arr)
    lines_arr = lines_arr[order]

    first_pixels = pixel_rows[order[0]]
    # Sentinel-1 calibration vectors normally all share one pixel grid. If a
    # product genuinely doesn't (some products vary it slightly), resample
    # every row onto the first row's pixel grid rather than silently
    # assuming — this keeps sigma_grid rectangular without discarding data.
    sigma_grid = np.empty((len(order), len(first_pixels)), dtype=np.float64)
    for out_i, src_i in enumerate(order):
        if np.array_equal(pixel_rows[src_i], first_pixels):
            sigma_grid[out_i] = sigma_rows[src_i]
        else:
            sigma_grid[out_i] = np.interp(first_pixels, pixel_rows[src_i], sigma_rows[src_i])

    return lines_arr, first_pixels, sigma_grid


def _linear_interp_weights(query: np.ndarray, knots: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each value in `query`, the two bracketing indices into `knots`
    (clamped at the ends) and the linear weight of the upper one."""
    idx_hi = np.clip(np.searchsorted(knots, query, side="left"), 1, len(knots) - 1)
    idx_lo = idx_hi - 1
    x_lo, x_hi = knots[idx_lo], knots[idx_hi]
    denom = np.where(x_hi > x_lo, x_hi - x_lo, 1.0)
    w_hi = np.clip((query - x_lo) / denom, 0.0, 1.0)
    return idx_lo, idx_hi, w_hi


def build_sigma_naught_lut(
    lines: np.ndarray, pixels: np.ndarray, sigma_grid: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Bilinearly upsamples the coarse (lines x pixels) sigmaNought grid to
    full (height x width) resolution, fully vectorized (no per-pixel Python
    loop — real GRD scenes are ~25,000x17,000 px, per spec §21/§22)."""
    full_width = np.arange(width, dtype=np.float64)
    along_pixels = np.empty((len(lines), width), dtype=np.float64)
    for i in range(len(lines)):
        along_pixels[i] = np.interp(full_width, pixels, sigma_grid[i])

    full_height = np.arange(height, dtype=np.float64)
    idx_lo, idx_hi, w_hi = _linear_interp_weights(full_height, lines)
    lo_vals = along_pixels[idx_lo]   # (height, width)
    hi_vals = along_pixels[idx_hi]   # (height, width)
    lut = lo_vals * (1.0 - w_hi)[:, None] + hi_vals * w_hi[:, None]
    return lut.astype(np.float32)


def calibrate_to_sigma0(dn: np.ndarray, sigma_naught_lut: np.ndarray) -> np.ndarray:
    """sigma0 = DN^2 / sigmaNought^2 (Module 2 spec §13's documented formula)."""
    if dn.shape != sigma_naught_lut.shape:
        raise CalibrationError(
            f"DN array shape {dn.shape} does not match calibration LUT shape "
            f"{sigma_naught_lut.shape} — cannot calibrate mismatched arrays."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma0 = (dn.astype(np.float64) ** 2) / (sigma_naught_lut.astype(np.float64) ** 2)
    return sigma0.astype(np.float32)


def calibrate_band(dn: np.ndarray, calibration_xml_path: str | Path) -> np.ndarray:
    """Convenience wrapper: parse the LUT for one band's calibration XML and
    calibrate a DN array of shape (height, width) to Sigma0."""
    height, width = dn.shape
    lines, pixels, sigma_grid = parse_calibration_lut(calibration_xml_path)
    lut = build_sigma_naught_lut(lines, pixels, sigma_grid, height, width)
    return calibrate_to_sigma0(dn, lut)
