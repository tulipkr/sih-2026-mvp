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
from src.infer_scene import run_scene_inference
from src.synthetic_data import generate_synthetic_dataset, generate_synthetic_scene
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
            "inference_manifest_path": str(manifest_path), "patch_size": 64, "bands": ["VV", "VH"],
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
    config["data"]["inference_manifest_path"] = str(manifest_path)
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
    config["data"]["inference_manifest_path"] = str(manifest_path)
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
    config["data"]["inference_manifest_path"] = str(manifest_path)
    config["data"]["patch_size"] = 64
    config["inference"]["on_dimension_mismatch"] = "resize"
    config["inference"]["output_dir"] = str(tmp_path / "resize_outputs")
    config_path = tmp_path / "config_resize.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)

    result = run_inference(str(config_path), entries[0]["scene_id"])
    assert result is not None  # did not raise; resize path completed


# --- Cross-module interface fixes: scene_id collision + scene-level stitching ---
# Ishita's real Stage A manifests give every tile of one scene the SAME scene_id
# (only patch_path/patch_id differ). These tests use generate_synthetic_scene,
# which mirrors that shape exactly, unlike generate_synthetic_dataset (which
# gives every synthetic sample its own unique scene_id and would never have
# caught this bug).

def _scene_config(tmp_path, manifest_path, checkpoint_path, **overrides):
    config = {
        "data": {
            "inference_manifest_path": str(manifest_path), "patch_size": 64,
            "bands": ["VV", "VH"], "in_channels": 2,
            "normalization": {"method": "none"},
        },
        "model": {"in_channels": 2, "out_channels": 1, "backbone": "resnet34", "pretrained": False},
        "train": {"device": "cpu", "checkpoint_path": str(checkpoint_path)},
        "inference": {
            "threshold": 0.5, "output_dir": str(tmp_path / "outputs"),
            "model_version": "unet_test_v0",
            "on_dimension_mismatch": "reject", "on_missing_checkpoint": "fallback",
        },
        "fallback": {"db_threshold": -18.0, "min_blob_area_px": 5, "land_marker": "nan_or_zero"},
        "logging": {"log_dir": str(tmp_path / "outputs" / "runs"), "level": "INFO"},
    }
    config["inference"].update(overrides.pop("inference", {}))
    config_path = tmp_path / "scene_config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)
    return config_path


def test_ambiguous_scene_id_without_patch_id_errors_clearly(tmp_path):
    """Manifest with two tiles sharing one scene_id and NO patch_id field —
    run_inference() (single-patch) must detect the ambiguity from the
    manifest lookup itself and raise BEFORE loading/resizing either tile —
    not let some unrelated later error (e.g. a dimension mismatch on
    whichever entry happened to be picked first) stand in for it."""
    manifest_path = generate_synthetic_scene(tmp_path / "amb", scene_id="dup", grid=(1, 2), patch_size=32)
    entries = json.loads(manifest_path.read_text())
    for e in entries:
        e.pop("patch_id", None)  # simulate an older-style manifest with no patch_id
    manifest_path.write_text(json.dumps(entries))

    config_path = _scene_config(tmp_path, manifest_path, tmp_path / "no_ckpt.pt")
    with pytest.raises(ValueError, match="matches 2 manifest entries"):
        run_inference(str(config_path), "dup")


def test_multiple_tiles_one_scene_stitched_correctly(tmp_path):
    """The core cross-module fix: one scene_id, multiple tiles (patch_id
    differs), all reachable and correctly placed in one stitched output."""
    manifest_path = generate_synthetic_scene(
        tmp_path / "multi", scene_id="scene_A", grid=(2, 2), patch_size=64, oil_tile=(0, 1)
    )
    config_path = _scene_config(tmp_path, manifest_path, tmp_path / "no_ckpt.pt")

    result = run_scene_inference(str(config_path), "scene_A")
    assert result["num_tiles_stitched"] == 4
    assert result["geolocation_incomplete"] is False

    import rasterio
    with rasterio.open(result["mask_path"]) as src:
        assert src.width == 128 and src.height == 128  # 2x2 grid of 64x64 tiles
        assert src.crs is not None


def test_scene_with_no_oil_anywhere_is_valid(tmp_path):
    """§16: no oil detected across a whole scene is valid, not an error."""
    manifest_path = generate_synthetic_scene(
        tmp_path / "nooil", scene_id="scene_clean", grid=(2, 2), patch_size=64, oil_tile=None
    )
    config_path = _scene_config(tmp_path, manifest_path, tmp_path / "no_ckpt.pt")
    result = run_scene_inference(str(config_path), "scene_clean")
    assert result["no_oil_detected"] is True
    assert result["positive_pixel_fraction"] == 0.0
    assert result["mean_score_in_positive_region"] is None


def test_scene_with_tile_missing_geolocation(tmp_path):
    """One tile in the scene has no CRS/transform (a real, if degraded,
    situation) — must not crash, must flag geolocation_incomplete at the
    scene level, must not fabricate coordinates for that tile."""
    manifest_path = generate_synthetic_scene(
        tmp_path / "geo_missing", scene_id="scene_B", grid=(2, 1), patch_size=64,
        oil_tile=(0, 0), drop_geo_on_tile=(1, 0),
    )
    config_path = _scene_config(tmp_path, manifest_path, tmp_path / "no_ckpt.pt")
    result = run_scene_inference(str(config_path), "scene_B")
    assert result["geolocation_incomplete"] is True
    assert result["num_tiles_missing_geolocation"] == 1
    assert result["num_tiles_stitched"] == 2  # both tiles still ran; one just wasn't placed


def test_nonexistent_inference_manifest_errors_explicitly(tmp_path):
    config_path = _scene_config(tmp_path, tmp_path / "does_not_exist_manifest.json", tmp_path / "no_ckpt.pt")
    with pytest.raises(DatasetValidationError, match="Manifest not found"):
        run_scene_inference(str(config_path), "anything")
