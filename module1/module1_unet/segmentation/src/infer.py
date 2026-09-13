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
    for e in entries:
        if e["scene_id"] == scene_id:
            return e
    raise ValueError(
        f"scene_id '{scene_id}' not found in manifest. "
        f"Available scene_ids: {[e['scene_id'] for e in entries]}"
    )


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


def run_inference(config_path: str, scene_id: str, checkpoint_path: str | None = None) -> dict:
    config = load_config(config_path)
    setup_logging(config["logging"]["log_dir"], config["logging"].get("level", "INFO"))

    entries = load_manifest_entries(config["data"]["manifest_path"])
    entry = _find_entry(entries, scene_id)

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
            f"scene_id={scene_id}: input patch contains NaN/Inf values — rejecting "
            f"rather than running inference on corrupt input (spec §15/§16)."
        )

    expected_bands = config["data"]["bands"]
    if image.shape[0] != len(expected_bands):
        raise ValueError(
            f"scene_id={scene_id}: patch has {image.shape[0]} bands, expected "
            f"{len(expected_bands)} ({expected_bands}). Missing required channel — "
            f"hard error, no substitution (spec §17)."
        )

    image = _resize_or_reject(
        image, config["data"]["patch_size"], config["inference"]["on_dimension_mismatch"], scene_id
    )

    geolocation_incomplete = crs is None or transform is None
    if geolocation_incomplete:
        logger.warning(
            f"scene_id={scene_id}: CRS or transform missing from input. Proceeding with "
            f"inference (spec §17), but downstream geolocation will not be possible."
        )

    checkpoint_path = Path(checkpoint_path or config["train"]["checkpoint_path"])
    output_dir = Path(config["inference"]["output_dir"])

    fallback_used = False
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
            fallback_used = True
        else:
            raise ValueError(f"Unknown inference.on_missing_checkpoint '{on_missing}'")

    threshold = config["inference"]["threshold"]

    if fallback_used:
        norm_cfg = NormalizationConfig.from_dict(config["data"].get("normalization"))
        # Fallback operates on the un-normalized dB values directly (its threshold
        # is itself a dB cutoff), so it does NOT apply config normalization.
        probs, binary_mask = run_fallback_detection(image, config["fallback"])
        score_type = "rule_based_threshold"
        model_version = "fallback_rule_based_v0"
    else:
        import torch

        device = resolve_device(config["train"].get("device", "auto"))
        norm_cfg = NormalizationConfig.from_dict(config["data"].get("normalization"))
        normalized = apply_normalization(image, norm_cfg)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(checkpoint.get("config", config)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            tensor = torch.from_numpy(normalized.copy()).float().unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

        binary_mask = (probs > threshold).astype(np.uint8)
        score_type = "raw_sigmoid_output"
        model_version = config["inference"]["model_version"]

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
        "threshold_used": float(threshold),
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
