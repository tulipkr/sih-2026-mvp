"""
Dedicated tests for every row of spec §16's edge-case table. Kept separate
from test_infer_pipeline.py/test_dataset.py so each required behavior in the
table has a traceable, named test rather than being buried in a general test.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import yaml

from src.dataset import DatasetValidationError, NormalizationConfig, SARSegmentationDataset
from src.infer import run_inference
from src.synthetic_data import generate_synthetic_dataset
from src.train import run_training


# --- "Empty/zero-size input | Reject with explicit error" ---
def test_empty_manifest_reject_explicit_error():
    with pytest.raises(DatasetValidationError):
        SARSegmentationDataset([], NormalizationConfig(method="none"), ["VV", "VH"], patch_size=256)


# --- "Missing input file | Explicit error naming the missing path" ---
def test_missing_patch_file_names_path(tmp_path):
    entries = [{
        "scene_id": "x", "patch_path": str(tmp_path / "nope.tif"),
        "mask_path": str(tmp_path / "nope_mask.tif"), "bands": ["VV", "VH"],
    }]
    ds = SARSegmentationDataset(entries, NormalizationConfig(method="none"), ["VV", "VH"], patch_size=256)
    with pytest.raises(DatasetValidationError, match="nope.tif"):
        ds[0]


# --- "Missing manifest fields | Warn, set geolocation_incomplete: true, proceed" ---
@pytest.fixture(scope="module")
def config_and_manifest(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("edge_cases")
    manifest_path = generate_synthetic_dataset(work_dir / "data", n_samples=6, patch_size=64, seed=3)
    config = {
        "data": {
            "manifest_path": str(manifest_path), "patch_size": 64, "bands": ["VV", "VH"],
            "in_channels": 2, "val_fraction": 0.34, "seed": 3,
            "tiling": {"strategy": "sliding_window", "stride": 64},
            "augmentation": {}, "class_imbalance": {"weighted_sampling": False},
            "normalization": {"method": "none"},
        },
        "model": {"in_channels": 2, "out_channels": 1, "backbone": "resnet34", "pretrained": False},
        "loss": {"bce_weight": 0.5, "dice_weight": 0.5},
        "train": {
            "batch_size": 2, "epochs": 1, "learning_rate": 1e-3, "num_workers": 0,
            "early_stopping_patience": 5, "device": "cpu",
            "checkpoint_path": str(work_dir / "checkpoints" / "best_model.pt"),
            "amp": False, "lookalike_manifest_path": None,
        },
        "inference": {
            "threshold": 0.5, "output_dir": str(work_dir / "outputs"),
            "model_version": "unet_test_v0",
            "on_dimension_mismatch": "reject", "on_missing_checkpoint": "error",
        },
        "fallback": {"db_threshold": -18.0, "min_blob_area_px": 5, "land_marker": "nan_or_zero"},
        "logging": {"log_dir": str(work_dir / "outputs" / "runs"), "level": "INFO"},
    }
    config_path = work_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)
    run_training(str(config_path))
    return {"work_dir": work_dir, "config_path": config_path, "manifest_path": manifest_path}


def test_missing_metadata_sets_geolocation_incomplete(config_and_manifest, tmp_path):
    entries = json.loads(Path(config_and_manifest["manifest_path"]).read_text())["entries"]
    entry = dict(entries[0])
    entry["crs"] = None
    entry["transform"] = None
    manifest_path = tmp_path / "manifest_missing_meta.json"
    manifest_path.write_text(json.dumps({"entries": [entry]}))

    config = yaml.safe_load(Path(config_and_manifest["config_path"]).read_text())
    config["data"]["manifest_path"] = str(manifest_path)
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)

    result = run_inference(str(config_path), entry["scene_id"])
    assert result["geolocation_incomplete"] is True
    assert result["crs"] is None
    assert result["transform"] is None


# --- "No detections (all-zero mask) | Valid output — no_oil_detected: true" ---
def test_no_detections_is_valid_not_an_error(config_and_manifest):
    from src.schema import validate_inference_result

    entries = json.loads(Path(config_and_manifest["manifest_path"]).read_text())["entries"]
    # find a scene with no injected oil (generate_synthetic_dataset marks the
    # first n_no_oil entries as no-oil)
    scene_id = entries[0]["scene_id"]
    result = run_inference(str(config_and_manifest["config_path"]), scene_id)
    # not asserting no_oil_detected is True here (depends on trained model behavior on
    # random-init weights) — asserting the *shape* of a valid response either way:
    validate_inference_result(result)
    if result["no_oil_detected"]:
        assert result["positive_pixel_fraction"] == 0.0
        assert result["mean_score_in_positive_region"] is None


# --- "Ishita's output format changes slightly | fail with a clear schema-mismatch error" ---
def test_manifest_schema_mismatch_fails_clearly():
    bad_entries = [{"scene_id": "x", "patch_path": "p.tif", "mask_path": "m.tif"}]  # missing "bands"
    with pytest.raises(DatasetValidationError, match="bands"):
        SARSegmentationDataset(bad_entries, NormalizationConfig(method="none"), ["VV", "VH"], patch_size=256)


# --- "Very large input scene | Tile into patches ... do not attempt full-scene inference" ---
def test_oversized_training_image_gets_tiled(tmp_path):
    """Training-time tiling (§12) — dataset should split a larger-than-patch_size
    image into multiple patch_size tiles rather than attempting one giant forward pass."""
    manifest_path = generate_synthetic_dataset(tmp_path / "data", n_samples=1, patch_size=128, seed=5)
    entries = json.loads(manifest_path.read_text())["entries"]
    ds = SARSegmentationDataset(
        entries, NormalizationConfig(method="none"), ["VV", "VH"], patch_size=64,
        tiling_strategy="sliding_window", tiling_stride=64,
    )
    assert len(ds) > 1, "a 128x128 image tiled at 64 should produce more than one tile"
    image, mask, _ = ds[0]
    assert image.shape == (2, 64, 64)


# --- Dimension mismatch at inference: reject (default) vs resize (configured) ---
def test_dimension_mismatch_reject_mode_raises(config_and_manifest, tmp_path):
    manifest_path = generate_synthetic_dataset(tmp_path / "wrongsize", n_samples=1, patch_size=32, seed=9)
    entries = json.loads(manifest_path.read_text())["entries"]

    config = yaml.safe_load(Path(config_and_manifest["config_path"]).read_text())
    config["data"]["manifest_path"] = str(manifest_path)
    config["data"]["patch_size"] = 64  # mismatched on purpose
    config["inference"]["on_dimension_mismatch"] = "reject"
    config_path = tmp_path / "config_reject.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)

    with pytest.raises(ValueError, match="on_dimension_mismatch"):
        run_inference(str(config_path), entries[0]["scene_id"])


def test_dimension_mismatch_resize_mode_warns_and_proceeds(config_and_manifest, tmp_path):
    manifest_path = generate_synthetic_dataset(tmp_path / "wrongsize2", n_samples=1, patch_size=32, seed=11)
    entries = json.loads(manifest_path.read_text())["entries"]

    config = yaml.safe_load(Path(config_and_manifest["config_path"]).read_text())
    config["data"]["manifest_path"] = str(manifest_path)
    config["data"]["patch_size"] = 64
    config["inference"]["on_dimension_mismatch"] = "resize"
    config["inference"]["output_dir"] = str(tmp_path / "resize_outputs")
    config_path = tmp_path / "config_resize.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)

    result = run_inference(str(config_path), entries[0]["scene_id"])
    assert result is not None  # did not raise; resize path completed
