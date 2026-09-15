import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import numpy as np
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from module1.schema import validate_inference_result


class MaskedTulipAdapter(nn.Module):
    """
    Lightweight Adapter Module that applies spatial mask gating for feature refinement.
    Handles dynamic spatial mask interpolation and tensor dimension broadcasting.
    """
    def __init__(self, in_features: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(in_features, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, in_features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features of shape (B, H, W, C).
            mask (torch.Tensor): Binary mask of shape (H_mask, W_mask) or (B, H_mask, W_mask).
        """
        residual = x
        
        # 1. Forward pass through adapter projections
        delta = self.down_proj(x)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up_proj(delta)

        # 2. Format mask tensor to (B, 1, H, W) for interpolation & broadcasting
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)               # (B, H, W) -> (B, 1, H, W)

        target_h, target_w = x.shape[1], x.shape[2]

        # 3. Resize mask if spatial resolution doesn't match feature tensor
        if mask.shape[2:] != (target_h, target_w):
            mask = F.interpolate(mask, size=(target_h, target_w), mode="nearest")

        # 4. Reshape mask to (B, H, W, 1) to match (B, H, W, C) feature layout
        mask = mask.permute(0, 2, 3, 1)

        # 5. Masked residual output
        return residual + (delta * mask)


def run_inference_and_build_manifest(
    scene_id: str,
    feature_tensor: torch.Tensor,
    mask_path: str,
    prob_map_path: str,
    output_mask_path: str,
    crs_str: str,
    transform_matrix: list,
    threshold: float = 0.5,
    fallback_used: bool = False
) -> dict:
    # 1. Read binary mask from TIF file and extract profile metadata
    with rasterio.open(mask_path) as src:
        mask_array = src.read(1)
        src_meta = src.meta.copy()

    mask_tensor = torch.from_numpy(mask_array).float()

    # 2. Pass features and mask into TulipAdapter
    adapter = MaskedTulipAdapter(in_features=feature_tensor.shape[-1], bottleneck_dim=64)
    adapter.eval()

    with torch.no_grad():
        adapted_features = adapter(feature_tensor, mask_tensor)
        # Generate probability score map
        prob_map = torch.sigmoid(adapted_features.mean(dim=-1)).squeeze(0).cpu().numpy()

    # Derive statistics for schema validation
    binary_mask = (prob_map >= threshold).astype(np.uint8)
    
    # 3. Save segmentation mask GeoTIFF output to disk
    out_meta = src_meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": binary_mask.shape[0],
        "width": binary_mask.shape[1],
        "count": 1,
        "crs": crs_str,
        "transform": rasterio.Affine(*transform_matrix[:6]) if len(transform_matrix) >= 6 else src_meta["transform"],
        "dtype": rasterio.uint8
    })

    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    with rasterio.open(output_mask_path, "w", **out_meta) as dst:
        dst.write(binary_mask, 1)

    # 4. Save probability map GeoTIFF output to disk
    prob_meta = out_meta.copy()
    prob_meta.update({"dtype": rasterio.float32})
    
    os.makedirs(os.path.dirname(prob_map_path), exist_ok=True)
    with rasterio.open(prob_map_path, "w", **prob_meta) as dst:
        dst.write(prob_map.astype(np.float32), 1)

    # 5. Metrics calculation
    total_pixels = binary_mask.size
    positive_pixels = int(np.sum(binary_mask))
    pos_fraction = float(positive_pixels / total_pixels) if total_pixels > 0 else 0.0
    no_oil = (positive_pixels == 0)

    mean_pos_score = (
        float(np.mean(prob_map[binary_mask == 1])) if positive_pixels > 0 else None
    )

    score_type = "rule_based_threshold" if fallback_used else "raw_sigmoid_output"

    # Construct manifest payload matching schema
    inference_result = {
        "scene_id": scene_id,
        "model_version": "v1.2.0-tulip-adapter",
        "inference_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_timestamp_utc": "2026-03-14T10:00:00Z",
        "crs": crs_str,
        "transform": transform_matrix,
        "geolocation_incomplete": False,
        "threshold_used": float(threshold),
        "positive_pixel_fraction": pos_fraction,
        "mean_score_in_positive_region": mean_pos_score,
        "no_oil_detected": no_oil,
        "score_type": score_type,
        "fallback_used": fallback_used,
        "prob_map_path": prob_map_path,
        "mask_path": output_mask_path
    }

    # Validate schema integrity before saving
    validate_inference_result(inference_result)

    # 6. Save manifest JSON relative to the current run output directory
    json_path = os.path.join(os.path.dirname(output_mask_path), "inference_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(inference_result, f, indent=4)

    return inference_result


if __name__ == "__main__":
    SCENES_DIR = os.path.join(PROJECT_ROOT, "backend", "data", "scenes")
    scene_id = "huntington_beach_2021"
    
    # Locate scene directory
    scene_folder = os.path.join(SCENES_DIR, scene_id)
    scene_mask_path = None

    # Find the target TIF file in scene folder
    if os.path.isdir(scene_folder):
        tif_files = [f for f in os.listdir(scene_folder) if f.endswith(".tif")]
        if tif_files:
            scene_mask_path = os.path.join(scene_folder, tif_files[0])

    if not scene_mask_path or not os.path.exists(scene_mask_path):
        raise FileNotFoundError(
            f"Could not locate any .tif mask file inside folder '{scene_folder}'."
        )

    out_dir = os.path.join(PROJECT_ROOT, "backend", "outputs", "test_runs", scene_id)
    os.makedirs(out_dir, exist_ok=True)

    output_mask_path = os.path.join(out_dir, f"{scene_id}_mask.tif")
    prob_map_path = os.path.join(out_dir, f"{scene_id}_probability.tif")

    dummy_features = torch.randn(1, 256, 256, 768)

    res = run_inference_and_build_manifest(
        scene_id=scene_id,
        feature_tensor=dummy_features,
        mask_path=scene_mask_path,
        prob_map_path=prob_map_path,
        output_mask_path=output_mask_path,
        crs_str="EPSG:4326",
        transform_matrix=[-118.1000, 0.0001, 0.0, 33.6800, 0.0, -0.0001],
        threshold=0.5,
        fallback_used=False
    )

    print("Success! Found mask and executed adapter:")
    print(json.dumps(res, indent=2))