"""
Inference entrypoint (spec §10, §15, §35).

Usage:
    python -m src.infer --config configs/config.yaml --scene_id synthetic_scene_000
    python -m src.infer --config configs/config.yaml --scene_id synthetic_scene_000 \
        --checkpoint checkpoints/best_model.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.dataset import NormalizationConfig, apply_normalization, load_manifest_entries
from src.fallback_detector import run_fallback_detection
from src.model import build_model
from src.schema import validate_inference_result
from src.utils import load_config, resolve_device, setup_logging, write_json

logger = logging.getLogger(__name__)


def _find_entry(entries: list[dict], scene_id: str) -> dict:
    """Find exactly one entry for single-patch inference.

    Ishita's real Stage A manifests give every tile of one scene the *same*
    `scene_id` (only `patch_path`/`patch_id` differ per tile). A unique
    `patch_id` match is tried first. Falling back to `scene_id` must detect
    ambiguity (more than one entry sharing that scene_id) and raise BEFORE
    any image is loaded or resized — ambiguous *identification* of which
    tile to run is a manifest-lookup problem, not a dimension problem, and
    must never be masked by a later, unrelated error from whichever entry
    happened to be picked first.
    """
    patch_id_matches = [e for e in entries if e.get("patch_id") == scene_id]
    if patch_id_matches:
        return patch_id_matches[0]

    scene_matches = [e for e in entries if e["scene_id"] == scene_id]
    if len(scene_matches) > 1:
        raise ValueError(
            f"'{scene_id}' matches {len(scene_matches)} manifest entries "
            f"sharing that scene_id (multiple tiles, no patch_id disambiguates "
            f"them). run_inference() only processes one patch — pass a "
            f"specific patch_id instead, or use "
            f"infer_scene.run_scene_inference() to process the whole scene "
            f"and get one stitched result."
        )
    if len(scene_matches) == 1:
        return scene_matches[0]

    raise ValueError(
        f"'{scene_id}' not found as a patch_id or scene_id in manifest. "
        f"Available scene_ids: {sorted({e['scene_id'] for e in entries})}"
    )


def _find_entries_for_scene(entries: list[dict], scene_id: str) -> list[dict]:
    """Return every manifest entry belonging to one scene (all its tiles),
    in patch_path order, for scene-level orchestration (see infer_scene.py)."""
    matches = [e for e in entries if e["scene_id"] == scene_id]
    if not matches:
        raise ValueError(
            f"scene_id '{scene_id}' not found in manifest. "
            f"Available scene_ids: {sorted({e['scene_id'] for e in entries})}"
        )
    return sorted(matches, key=lambda e: e.get("patch_id") or e["patch_path"])


def _write_geotiff(path: Path, array: np.ndarray, transform_list, crs: str | None, dtype: str):
    import rasterio
    from rasterio.transform import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape[-2], array.shape[-1]
    count = array.shape[0] if array.ndim == 3 else 1
    transform = Affine(*transform_list) if transform_list else None

    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=count,
        dtype=dtype, crs=crs, transform=transform,
    ) as dst:
        if array.ndim == 3:
            dst.write(array)
        else:
            dst.write(array, 1)


def _resize_or_reject(image: np.ndarray, expected_size: int, mode: str, scene_id: str) -> np.ndarray:
    h, w = image.shape[1], image.shape[2]
    if h == expected_size and w == expected_size:
        return image
    if mode == "reject":
        raise ValueError(
            f"scene_id={scene_id}: input is {h}x{w}, expected {expected_size}x{expected_size}. "
            f"inference.on_dimension_mismatch is 'reject' (the default) — set it to "
            f"'resize' in config.yaml if you want auto-resize instead (spec §15 allows either)."
        )
    if mode == "resize":
        import torch
        import torch.nn.functional as F

        logger.warning(
            f"scene_id={scene_id}: resizing input from {h}x{w} to "
            f"{expected_size}x{expected_size} (inference.on_dimension_mismatch='resize'). "
            f"This changes the physical patch's spatial resolution — verify this is "
            f"actually what you want, per spec §15's warning requirement."
        )
        tensor = torch.from_numpy(image).unsqueeze(0)
        resized = F.interpolate(tensor, size=(expected_size, expected_size), mode="bilinear")
        return resized.squeeze(0).numpy()
    raise ValueError(f"Unknown inference.on_dimension_mismatch '{mode}'")


def load_and_validate_patch(entry: dict, config: dict, label: str) -> tuple:
    """Read one manifest entry's patch, apply the same validation/dimension
    handling run_inference always has. `label` is just for error messages
    (a scene_id for single-patch calls, or f"{scene_id}/{patch_id}" for
    scene-level calls) — shared so both callers report errors identically."""
    import rasterio

    patch_path = Path(entry["patch_path"])
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path.resolve()}")

    with rasterio.open(patch_path) as src:
        image = src.read().astype(np.float32)
        crs = entry.get("crs")
        transform = entry.get("transform")

    if not np.isfinite(image).all():
        raise ValueError(
            f"{label}: input patch contains NaN/Inf values — rejecting "
            f"rather than running inference on corrupt input (spec §15/§16)."
        )

    expected_bands = config["data"]["bands"]
    if image.shape[0] != len(expected_bands):
        raise ValueError(
            f"{label}: patch has {image.shape[0]} bands, expected "
            f"{len(expected_bands)} ({expected_bands}). Missing required channel — "
            f"hard error, no substitution (spec §17)."
        )

    image = _resize_or_reject(
        image, config["data"]["patch_size"], config["inference"]["on_dimension_mismatch"], label
    )

    geolocation_incomplete = crs is None or transform is None
    if geolocation_incomplete:
        logger.warning(
            f"{label}: CRS or transform missing from input. Proceeding with "
            f"inference (spec §17), but downstream geolocation will not be possible."
        )

    return image, crs, transform, geolocation_incomplete


def resolve_inference_backend(config: dict, checkpoint_path: str | None = None):
    """Decide once (per run, not per-tile) whether inference uses the trained
    model or the rule-based fallback, and load whichever is needed.

    Returns (fallback_used, model_or_None, device_or_None, checkpoint_path).
    Shared by run_inference (single patch) and infer_scene.run_scene_inference
    (many tiles of one scene) so a scene's tiles are never processed under a
    mix of model/fallback — the decision and the model load happen exactly
    once per run.
    """
    checkpoint_path = Path(checkpoint_path or config["train"]["checkpoint_path"])

    if not checkpoint_path.exists():
        on_missing = config["inference"].get("on_missing_checkpoint", "error")
        if on_missing == "error":
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path.resolve()}. Run training first, "
                f"or set inference.on_missing_checkpoint: fallback in config.yaml to use "
                f"the rule-based fallback detector instead (spec §15/§35)."
            )
        elif on_missing == "fallback":
            logger.warning(
                f"Checkpoint not found at {checkpoint_path.resolve()}. "
                f"inference.on_missing_checkpoint='fallback' — using rule-based fallback detector."
            )
            return True, None, None, checkpoint_path
        else:
            raise ValueError(f"Unknown inference.on_missing_checkpoint '{on_missing}'")

    import torch

    device = resolve_device(config["train"].get("device", "auto"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint.get("config", config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return False, model, device, checkpoint_path


def run_tile_inference(image: np.ndarray, config: dict, fallback_used: bool, model, device):
    """Run the model or the fallback detector on one already-validated,
    already-normalized-or-not (see below) (C,H,W) tile. Returns
    (probs, binary_mask, score_type, model_version)."""
    threshold = config["inference"]["threshold"]

    if fallback_used:
        # Fallback operates on the un-normalized dB values directly (its threshold
        # is itself a dB cutoff), so it does NOT apply config normalization.
        probs, binary_mask = run_fallback_detection(image, config["fallback"])
        score_type = "rule_based_threshold"
        model_version = "fallback_rule_based_v0"
    else:
        import torch

        norm_cfg = NormalizationConfig.from_dict(config["data"].get("normalization"))
        normalized = apply_normalization(image, norm_cfg)

        with torch.no_grad():
            tensor = torch.from_numpy(normalized.copy()).float().unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

        binary_mask = (probs > threshold).astype(np.uint8)
        score_type = "raw_sigmoid_output"
        model_version = config["inference"]["model_version"]

    return probs, binary_mask, score_type, model_version


def run_inference(config_path: str, scene_id: str, checkpoint_path: str | None = None) -> dict:
    """Single-patch inference (unchanged public behavior/signature). scene_id
    is looked up via _find_entry — which now also accepts a patch_id when the
    manifest has one. For a whole scene with multiple tiles, use
    infer_scene.run_scene_inference() instead, which stitches all of a
    scene's tiles into one result."""
    config = load_config(config_path)
    setup_logging(config["logging"]["log_dir"], config["logging"].get("level", "INFO"))

    entries = load_manifest_entries(config["data"]["inference_manifest_path"])
    entry = _find_entry(entries, scene_id)

    image, crs, transform, geolocation_incomplete = load_and_validate_patch(entry, config, scene_id)

    output_dir = Path(config["inference"]["output_dir"])
    fallback_used, model, device, checkpoint_path = resolve_inference_backend(config, checkpoint_path)
    probs, binary_mask, score_type, model_version = run_tile_inference(
        image, config, fallback_used, model, device
    )

    positive_pixel_fraction = float(binary_mask.mean())
    no_oil_detected = positive_pixel_fraction == 0.0
    mean_score_in_positive_region = (
        float(probs[binary_mask == 1].mean()) if positive_pixel_fraction > 0 else None
    )

    prob_map_path = output_dir / "prob_maps" / f"{scene_id}.tif"
    mask_path = output_dir / "masks" / f"{scene_id}.tif"
    _write_geotiff(prob_map_path, probs.astype(np.float32), transform, crs, dtype="float32")
    _write_geotiff(mask_path, binary_mask, transform, crs, dtype="uint8")

    result = {
        "scene_id": scene_id,
        "model_version": model_version,
        "inference_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "acquisition_timestamp_utc": entry.get("acquisition_timestamp_utc"),
        "crs": crs,
        "transform": transform,
        "geolocation_incomplete": geolocation_incomplete,
        "threshold_used": float(config["inference"]["threshold"]),
        "positive_pixel_fraction": positive_pixel_fraction,
        "mean_score_in_positive_region": mean_score_in_positive_region,
        "no_oil_detected": no_oil_detected,
        "score_type": score_type,
        "fallback_used": fallback_used,
        "prob_map_path": str(prob_map_path),
        "mask_path": str(mask_path),
    }

    validate_inference_result(result)

    result_path = output_dir / f"{scene_id}_inference_result.json"
    write_json(result_path, result)
    logger.info(f"Inference complete for {scene_id} (fallback_used={fallback_used}). "
                f"Result written to {result_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on one manifest entry.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--scene_id", required=True)
    args = parser.parse_args()
    result = run_inference(args.config, args.scene_id, args.checkpoint)
    print(result)
