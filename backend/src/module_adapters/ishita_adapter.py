import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import sys
import os
import rasterio
from rasterio.transform import Affine

# Direct imports from Ishita's modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from module2.Stage_A.preprocess_pipeline import run_stage_a
from module2.Stage_B.postprocess_pipeline import run_stage_b

logger = logging.getLogger(__name__)


class IshitaPipelineHandler:
    """
    Handles Ishita's Stage A (Preprocessing) and Stage B (Postprocessing) pipeline steps,
    providing direct interfaces to format inputs for and consume outputs from Tulip's Adapter.
    """

    def __init__(self, patch_size: int = 256):
        self.patch_size = patch_size

    def run_preprocessing(
        self,
        output_dir: str,
        safe_path: Optional[str] = None,
        vv_path: Optional[str] = None,
        vh_path: Optional[str] = None,
        scene_id: Optional[str] = None,
        timestamp_utc: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Executes Stage A preprocessing.

        Returns:
            manifest_path (str): Path to the generated Stage A manifest JSON.
            tulip_inputs (dict): Extracted metadata parameters formatted specifically 
                                 to pass into `run_inference_and_build_manifest()`.
        """
        logger.info("Executing Ishita Stage A Preprocessing...")

        manifest_path = run_stage_a(
            output_dir=output_dir,
            safe_path=safe_path,
            vv_path=vv_path,
            vh_path=vh_path,
            scene_id=scene_id,
            timestamp_utc=timestamp_utc,
            patch_size=self.patch_size,
            config=config or {},
        )

        # Parse output manifest to extract spatial/temporal parameters for Tulip
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)

        first_patch = manifest_data[0] if isinstance(manifest_data, list) else manifest_data

        tulip_inputs = {
            "scene_id": first_patch.get("scene_id", scene_id or "unknown_scene"),
            "crs_str": first_patch.get("crs", "EPSG:4326"),
            "transform_matrix": first_patch.get("transform", [1.0, 0.0, 0.0, 0.0, -1.0, 0.0]),
            "acquisition_timestamp_utc": first_patch.get("acquisition_timestamp_utc", timestamp_utc),
            "sample_patch_path": first_patch.get("patch_path"),
        }

        return str(manifest_path), tulip_inputs

    def prepare_dummy_tulip_outputs(
        self,
        output_dir: str,
        scene_id: str,
        crs_str: str,
        transform_matrix: list,
        height: int = 256,
        width: int = 256,
    ) -> Tuple[str, str, str]:
        """
        Generates dummy GeoTIFF files (input mask, probability map, output mask) 
        matching spatial dimensions and CRS. Useful for testing Tulip integration locally.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        mask_path = str(out_path / f"{scene_id}_input_mask.tif")
        prob_map_path = str(out_path / f"{scene_id}_probability.tif")
        output_mask_path = str(out_path / f"{scene_id}_output_mask.tif")

        # Reconstruct affine transform
        if len(transform_matrix) == 6:
            transform = Affine(*transform_matrix)
        else:
            transform = Affine.identity()

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": rasterio.float32,
            "crs": crs_str,
            "transform": transform,
        }

        # Write initial dummy input mask (uint8)
        with rasterio.open(mask_path, "w", **{**profile, "dtype": rasterio.uint8}) as dst:
            dst.write(np.zeros((height, width), dtype=np.uint8), 1)

        # Write dummy probability map (float32)
        with rasterio.open(prob_map_path, "w", **profile) as dst:
            dst.write(np.random.uniform(0.0, 1.0, (height, width)).astype(np.float32), 1)

        # Write dummy output mask (uint8)
        with rasterio.open(output_mask_path, "w", **{**profile, "dtype": rasterio.uint8}) as dst:
            dst.write(np.zeros((height, width), dtype=np.uint8), 1)

        return mask_path, prob_map_path, output_mask_path

    def run_postprocessing(
        self,
        mask_path: str,
        inference_json_path: str,
        output_dir: str,
        stage_a_manifest_path: Optional[str] = None,
        min_area_km2: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Executes Stage B postprocessing:
        Converts raster masks into vectorized GeoJSON and generates spill summary statistics.
        """
        logger.info("Executing Ishita Stage B Postprocessing...")

        run_stage_b(
            mask_path=mask_path,
            inference_json_path=inference_json_path,
            output_dir=output_dir,
            min_area_km2=min_area_km2,
            stage_a_manifest_path=stage_a_manifest_path,
        )

        summary_file = Path(output_dir) / "spill_summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                return json.load(f)

        return {"status": "SUCCESS", "message": f"Outputs saved to {output_dir}"}


if __name__ == "__main__":
    # Example standalone usage test
    handler = IshitaPipelineHandler(patch_size=256)

    print("Ishita Pipeline Handler initialized successfully.")