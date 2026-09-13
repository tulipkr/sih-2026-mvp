"""
Rule-based fallback detector (spec §35).

Used when the trained model checkpoint is missing/unusable. Thresholds the
VV band (dB scale) below a configurable cutoff to flag oil-like pixels.

Land handling note (flagged in this module's README as a resolved ambiguity):
spec §35 says to combine the threshold "with a basic land mask," but spec §2
explicitly assigns land masking to Ishita's preprocessing module, not Tulip.
Since Tulip only ever receives Ishita's already-preprocessed input, this
detector respects land markers already present in the input (NaN or exactly
0.0, per config.fallback.land_marker) rather than computing its own
coastline mask. This assumes Ishita's preprocessing marks land that way —
that assumption needs her confirmation, not just this module's.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def run_fallback_detection(
    image: np.ndarray,
    fallback_config: dict[str, Any],
    vv_band_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: (C, H, W) float32 array, VV assumed at vv_band_index.
        fallback_config: config['fallback'] dict (db_threshold, min_blob_area_px, land_marker).
    Returns:
        (prob_like_score[H,W] float32 in [0,1], binary_mask[H,W] uint8)
    """
    if image.ndim != 3 or image.shape[0] <= vv_band_index:
        raise ValueError(
            f"Expected image shape (C,H,W) with at least {vv_band_index + 1} bands, "
            f"got {image.shape}"
        )

    vv = image[vv_band_index]
    db_threshold = fallback_config.get("db_threshold", -18.0)
    land_marker = fallback_config.get("land_marker", "nan_or_zero")

    if land_marker == "nan_or_zero":
        land = ~np.isfinite(vv) | (vv == 0.0)
    elif land_marker == "none":
        land = np.zeros_like(vv, dtype=bool)
    else:
        raise ValueError(f"Unknown fallback.land_marker '{land_marker}'")

    valid_pixels = np.isfinite(vv) & ~land
    below_threshold = valid_pixels & (vv < db_threshold)

    binary_mask = below_threshold.astype(np.uint8)

    # Heuristic "confidence" — NOT a model probability. Scaled distance below
    # threshold, clipped to [0,1], purely to give downstream consumers a
    # nonzero prob_map.tif rather than a flat step function. Clearly labeled
    # via score_type='rule_based_threshold' in the output JSON — never
    # confuse this with the trained model's sigmoid output.
    score = np.zeros_like(vv, dtype=np.float32)
    depth_below = np.clip(db_threshold - vv, 0, None)
    score[valid_pixels] = np.clip(depth_below[valid_pixels] / 10.0, 0.0, 1.0)
    score[~valid_pixels] = 0.0

    min_blob_area = fallback_config.get("min_blob_area_px", 20)
    if min_blob_area > 0:
        binary_mask = _remove_small_blobs(binary_mask, min_blob_area)
        score = score * binary_mask  # zero out score wherever the blob filter removed pixels

    logger.warning(
        f"Fallback detector used (db_threshold={db_threshold}, "
        f"land_marker={land_marker}). This is a rule-based heuristic, not the "
        f"trained model — treat results accordingly."
    )

    return score.astype(np.float32), binary_mask.astype(np.uint8)


def _remove_small_blobs(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Connected-component filter without requiring OpenCV/scikit-image —
    pure-numpy flood fill, adequate for the small MVP patch sizes involved."""
    mask = mask.copy()
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape

    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1 and not visited[i, j]:
                stack = [(i, j)]
                component = []
                visited[i, j] = True
                while stack:
                    y, x = stack.pop()
                    component.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and mask[ny, nx] == 1
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(component) < min_area_px:
                    for y, x in component:
                        mask[y, x] = 0
    return mask
