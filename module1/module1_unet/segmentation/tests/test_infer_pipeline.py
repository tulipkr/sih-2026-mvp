"""
spec §29 integration tests: test_train_tiny_subset_smoke, test_infer_end_to_end,
test_infer_on_corrupt_input, test_fallback_path.

Requires torch + rasterio + segmentation-models-pytorch installed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import yaml

from src.infer import run_inference
from src.synthetic_data import generate_lookalike_set, generate_synthetic_dataset
from src.train import run_training


def _make_config(work_dir: Path, manifest_path: Path, lookalike_path: Path | None = None) -> Path:
    config = {
        "data": {
            "manifest_path": str(manifest_path),
            "patch_size": 64,
            "bands": ["VV", "VH"],
            "in_channels": 2,
            "val_fraction": 0.25,
            "seed": 7,
            "tiling": {"strategy": "sliding_window", "stride": 64},
            "augmentation": {"horizontal_flip": True, "vertical_flip": True, "rotate_90": True},
            "class_imbalance": {"weighted_sampling": False, "min_oil_fraction_target": 0.3},
            "normalization": {"method": "none"},
        },
        "model": {
            "in_channels": 2, "out_channels": 1, "backbone": "resnet34", "pretrained": False,
        },
        "loss": {"bce_weight": 0.5, "dice_weight": 0.5},
        "train": {
            "batch_size": 2, "epochs": 2, "learning_rate": 1e-3, "num_workers": 0,
            "early_stopping_patience": 10, "device": "cpu",
            "checkpoint_path": str(work_dir / "checkpoints" / "best_model.pt"),
            "amp": False,
            "lookalike_manifest_path": str(lookalike_path) if lookalike_path else None,
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
    return config_path


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("e2e")
    manifest_path = generate_synthetic_dataset(work_dir / "data", n_samples=8, patch_size=64, seed=7)
    lookalike_path = generate_lookalike_set(work_dir / "lookalike", n_samples=3, patch_size=64, seed=8)
    config_path = _make_config(work_dir, manifest_path, lookalike_path)

    train_result = run_training(str(config_path))
    entries = json.loads(manifest_path.read_text())["entries"]
    scene_id = entries[0]["scene_id"]
    infer_result = run_inference(str(config_path), scene_id)

    return {
        "work_dir": work_dir, "config_path": config_path, "manifest_path": manifest_path,
        "train_result": train_result, "infer_result": infer_result, "scene_id": scene_id,
    }


def test_train_tiny_subset_smoke(trained_run):
    """spec §29: one epoch on a tiny subset completes without error and saves a checkpoint."""
    ckpt = Path(trained_run["train_result"]["best_ckpt_path"])
    assert ckpt.exists()


def test_lookalike_false_positive_check_ran(trained_run):
    report = trained_run["train_result"]["report"]["lookalike_false_positive_check"]
    assert report is not None
    assert "false_positive_rate" in report


def test_infer_end_to_end(trained_run):
    """spec §29: produces valid mask.tif + prob_map.tif + JSON with georeferencing preserved."""
    import rasterio

    result = trained_run["infer_result"]
    assert Path(result["prob_map_path"]).exists()
    assert Path(result["mask_path"]).exists()
    output_dir = Path(trained_run["work_dir"]) / "outputs"
    assert (output_dir / f"{trained_run['scene_id']}_inference_result.json").exists()

    with rasterio.open(result["prob_map_path"]) as src:
        assert src.crs is not None
        assert src.transform is not None


def test_infer_on_corrupt_input_raises_explicit_error(trained_run, tmp_path):
    """spec §29: corrupt/missing file raises the expected explicit error, not a silent crash."""
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps({"entries": [{
        "scene_id": "corrupt_scene",
        "patch_path": str(tmp_path / "does_not_exist.tif"),
        "mask_path": str(tmp_path / "does_not_exist_mask.tif"),
        "acquisition_timestamp_utc": "2026-01-01T00:00:00Z",
        "crs": "EPSG:4326", "transform": [1, 0, 0, 0, 1, 0], "bands": ["VV", "VH"],
    }]}))

    config = yaml.safe_load(Path(trained_run["config_path"]).read_text())
    config["data"]["manifest_path"] = str(bad_manifest)
    bad_config_path = tmp_path / "bad_config.yaml"
    with bad_config_path.open("w") as f:
        yaml.safe_dump(config, f)

    with pytest.raises(FileNotFoundError):
        run_inference(str(bad_config_path), "corrupt_scene")


def test_fallback_path_produces_schema_valid_output(trained_run, tmp_path):
    """spec §29 test_fallback_path: with the checkpoint deliberately removed, the
    fallback detector still produces schema-valid output."""
    from src.schema import validate_inference_result

    config = yaml.safe_load(Path(trained_run["config_path"]).read_text())
    config["train"]["checkpoint_path"] = str(tmp_path / "no_such_checkpoint.pt")
    config["inference"]["on_missing_checkpoint"] = "fallback"
    config["inference"]["output_dir"] = str(tmp_path / "fallback_outputs")
    fallback_config_path = tmp_path / "fallback_config.yaml"
    with fallback_config_path.open("w") as f:
        yaml.safe_dump(config, f)

    result = run_inference(str(fallback_config_path), trained_run["scene_id"])
    assert result["fallback_used"] is True
    assert result["score_type"] == "rule_based_threshold"
    validate_inference_result(result)  # raises on any schema problem


def test_missing_checkpoint_without_fallback_flag_raises(trained_run, tmp_path):
    config = yaml.safe_load(Path(trained_run["config_path"]).read_text())
    config["train"]["checkpoint_path"] = str(tmp_path / "no_such_checkpoint.pt")
    config["inference"]["on_missing_checkpoint"] = "error"  # the default
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)

    with pytest.raises(FileNotFoundError):
        run_inference(str(config_path), trained_run["scene_id"])
