"""
Full-scene inference for the oil-spill U-Net.

Pipeline:
    2048x2048 VV/VH GeoTIFF
        -> 256x256 tiles
        -> U-Net prediction
        -> stitched full-scene probability map
        -> binary mask
        -> georeferenced GeoTIFFs

The model must use the same architecture/configuration as training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from src.model import build_model
from src.utils import load_config
from src.dataset import NormalizationConfig, apply_normalization
from src.schema import validate_inference_result
from src.fallback_detector import run_fallback_detection
from datetime import datetime, timezone

def load_checkpoint(model, checkpoint_path, device):
    """Load the trained model checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Training code may save either a raw state_dict
    # or a dictionary containing model_state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def predict_scene(
    model,
    input_path,
    output_dir,
    device, 
    config,
    patch_size=256,
    threshold=0.5,
    fallback_used=False,
):
    """
    Run tiled inference over one full Sentinel-1 VV/VH scene.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_id = input_path.stem
    norm_cfg = NormalizationConfig.from_dict(config["data"].get("normalization"))

    with rasterio.open(input_path) as src:

        if src.count < 2:
            raise ValueError(
                f"{input_path} must contain at least 2 bands (VV and VH)."
            )

        width = src.width
        height = src.height

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

        probability = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        # Tracks how many predictions contributed to each pixel.
        # Normally every pixel receives exactly one prediction
        # when stride == patch_size.
        counts = np.zeros(
            (height, width),
            dtype=np.float32,
        )
        fallback_mask_full = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        with torch.no_grad():

            for row in range(0, height, patch_size):
                for col in range(0, width, patch_size):

                    h = min(patch_size, height - row)
                    w = min(patch_size, width - col)

                    # Read VV + VH.
                    patch = src.read(
                        [1, 2],
                        window=rasterio.windows.Window(
                            col,
                            row,
                            w,
                            h,
                        ),
                    ).astype(np.float32)

                    # Reject corrupt input rather than silently replacing NaN/Inf values.
                    if not np.isfinite(patch).all():
                        raise ValueError(
                            f"{input_path}: input scene contains NaN/Inf values "
                            f"in patch at row={row}, col={col}."
                        )

                    if fallback_used:
                        # Fallback must operate on the original dB values.
                        probs, fallback_mask = run_fallback_detection(
                            patch,
                            config["fallback"],
                        )
                        fallback_mask_full[
                            row:row + h,
                            col:col + w,
                        ] = fallback_mask
                    else:
                        # U-Net receives the configured normalized representation.
                        patch = apply_normalization(patch, norm_cfg)

                        # Pad edge patches to 256x256.
                        padded = np.zeros(
                            (2, patch_size, patch_size),
                            dtype=np.float32,
                        )

                        padded[:, :h, :w] = patch

                        tensor = torch.from_numpy(
                            padded
                        ).unsqueeze(0).to(device)

                        logits = model(tensor)

                        probs = torch.sigmoid(logits)

                        probs = probs.squeeze().detach().cpu().numpy()

                    # Only keep the real image area.
                    probs = probs[:h, :w]

                    probability[
                        row:row + h,
                        col:col + w,
                    ] += probs

                    counts[
                        row:row + h,
                        col:col + w,
                    ] += 1.0

    # Avoid division by zero.
    valid = counts > 0

    probability[valid] /= counts[valid]

    # Pixels never predicted are left at zero.
    if fallback_used:
        binary_mask = fallback_mask_full
    else:
        binary_mask = (
            probability >= threshold
        ).astype(np.uint8)
    # ---------------------------------------------------------
    # Save probability map
    # ---------------------------------------------------------

    probability_path = (
        output_dir / f"{scene_id}_probability.tif"
    )

    probability_profile = profile.copy()

    probability_profile.update(
        count=1,
        dtype="float32",
        nodata=None,
        compress="deflate",
    )

    with rasterio.open(
        probability_path,
        "w",
        **probability_profile,
    ) as dst:

        dst.write(
            probability,
            1,
        )

    # ---------------------------------------------------------
    # Save binary mask
    # ---------------------------------------------------------

    mask_path = (
        output_dir / f"{scene_id}_mask.tif"
    )

    mask_profile = profile.copy()

    mask_profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )

    with rasterio.open(
        mask_path,
        "w",
        **mask_profile,
    ) as dst:

        dst.write(
            binary_mask,
            1,
        )

    # ---------------------------------------------------------
    # Basic detection statistics
    # ---------------------------------------------------------

    oil_pixels = int(binary_mask.sum())

    
    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------

    metadata = {
        "scene_id": scene_id,
        "model_version": (
            "fallback_rule_based_v0"
            if fallback_used
            else config["inference"]["model_version"]
        ),
        "inference_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_timestamp_utc": None,
        "crs": str(crs) if crs else None,
        "transform": [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ],
        "geolocation_incomplete": crs is None,
        "threshold_used": threshold,
        "positive_pixel_fraction": float(binary_mask.mean()),
        "mean_score_in_positive_region": (
            float(probability[binary_mask == 1].mean())
            if oil_pixels > 0
            else 0.0
        ),
        "no_oil_detected": oil_pixels == 0,
        "score_type": (
            "rule_based_threshold"
            if fallback_used
            else "raw_sigmoid_output"
        ),
        "fallback_used": fallback_used,
        "prob_map_path": str(probability_path),
        "mask_path": str(mask_path),
    }

    validate_inference_result(metadata)
    metadata_path = (
        output_dir / "inference_result.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"Scene: {scene_id}"
    )

    print(
        f"Probability map: {probability_path}"
    )

    print(
        f"Binary mask: {mask_path}"
    )

    print(
        f"Oil pixels: {oil_pixels}"
    )


def main():

    parser = argparse.ArgumentParser(
        description="Run U-Net inference over a full VV/VH Sentinel-1 scene."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the full-scene VV/VH GeoTIFF.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best_model.pt.",
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to model config.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/full_scene",
        help="Directory for inference outputs.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Inference patch size.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(
        f"Using device: {device}"
    )

    checkpoint_path = Path(args.checkpoint)

    fallback_used = False

    if not checkpoint_path.exists():
        on_missing = config["inference"].get(
            "on_missing_checkpoint",
            "error",
        )

        if on_missing == "error":
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path.resolve()}. "
                f"Run training first, or set "
                f"inference.on_missing_checkpoint: fallback "
                f"in config.yaml to use the rule-based fallback detector."
            )

        elif on_missing == "fallback":
            print(
                "Checkpoint not found — using rule-based fallback detector."
            )
            model = None
            fallback_used = True

        else:
            raise ValueError(
                f"Unknown inference.on_missing_checkpoint '{on_missing}'"
            )

    else:
        print(
            "Loading checkpoint..."
        )

        model = build_model(config)

        model = load_checkpoint(
            model,
            checkpoint_path,
            device,
        )

    print(
        "Running full-scene inference..."
    )

    predict_scene(
        model=model,
        input_path=args.input,
        output_dir=args.output_dir,
        device=device,
        config=config,
        patch_size=args.patch_size,
        threshold=config["inference"]["threshold"],
        fallback_used=fallback_used,
    )


if __name__ == "__main__":
    main()