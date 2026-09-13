"""spec §28: test_dataset_loader (shapes/dtypes, rejects malformed sample),
test_normalization_matches_contract."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.dataset import (
    DatasetValidationError,
    NormalizationConfig,
    SARSegmentationDataset,
    apply_normalization,
    load_manifest_entries,
    train_val_split,
)
from src.synthetic_data import generate_synthetic_dataset


@pytest.fixture(scope="module")
def synthetic_manifest(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("synthetic")
    return generate_synthetic_dataset(out_dir, n_samples=6, patch_size=256, seed=1)


def test_manifest_loads(synthetic_manifest):
    entries = load_manifest_entries(synthetic_manifest)
    assert len(entries) == 6
    for e in entries:
        assert e["bands"] == ["VV", "VH"]


def test_train_val_split(synthetic_manifest):
    entries = load_manifest_entries(synthetic_manifest)
    train, val = train_val_split(entries, val_fraction=0.3, seed=42)
    assert len(train) + len(val) == len(entries)
    assert len(val) >= 1


def test_image_and_mask_shapes(synthetic_manifest):
    """spec §28 test_model_forward companion: image shape = 2x256x256."""
    entries = load_manifest_entries(synthetic_manifest)
    norm_cfg = NormalizationConfig(method="none")
    ds = SARSegmentationDataset(entries, norm_cfg, ["VV", "VH"], patch_size=256)
    image, mask, scene_id = ds[0]
    assert image.shape == (2, 256, 256), f"expected (2,256,256), got {tuple(image.shape)}"
    assert mask.shape == (1, 256, 256), f"expected (1,256,256), got {tuple(mask.shape)}"
    assert image.dtype.is_floating_point
    assert isinstance(scene_id, str)


def test_rejects_wrong_band_order(synthetic_manifest):
    entries = load_manifest_entries(synthetic_manifest)
    norm_cfg = NormalizationConfig(method="none")
    with pytest.raises(DatasetValidationError):
        SARSegmentationDataset(entries, norm_cfg, ["VH", "VV"], patch_size=256)


def test_rejects_missing_manifest_field():
    bad_entries = [{"scene_id": "x", "patch_path": "p.tif"}]  # missing mask_path, bands
    norm_cfg = NormalizationConfig(method="none")
    with pytest.raises(DatasetValidationError):
        SARSegmentationDataset(bad_entries, norm_cfg, ["VV", "VH"], patch_size=256)


def test_empty_manifest_rejected():
    norm_cfg = NormalizationConfig(method="none")
    with pytest.raises(DatasetValidationError):
        SARSegmentationDataset([], norm_cfg, ["VV", "VH"], patch_size=256)


def test_check_positive_fraction(synthetic_manifest):
    """spec §11 dataset-level check: confirm a non-trivial fraction of tiles have oil."""
    entries = load_manifest_entries(synthetic_manifest)
    norm_cfg = NormalizationConfig(method="none")
    ds = SARSegmentationDataset(entries, norm_cfg, ["VV", "VH"], patch_size=256)
    fraction = ds.check_positive_fraction()
    assert fraction > 0.0


def test_normalization_none_is_passthrough():
    import numpy as np

    image = np.array([[[1.0, -5.0], [3.0, 100.0]]], dtype=np.float32)
    out = apply_normalization(image, NormalizationConfig(method="none"))
    assert np.array_equal(out, image)


def test_normalization_matches_contract_db_clip_minmax():
    """spec §28: normalized output range matches the agreed spec (§12/§25 example)."""
    import numpy as np

    image = np.full((1, 4, 4), -15.0, dtype=np.float32)  # midpoint of [-30, 0]
    cfg = NormalizationConfig(method="db_clip_minmax", db_clip_min=-30.0, db_clip_max=0.0)
    out = apply_normalization(image, cfg)
    assert out.min() >= 0.0 and out.max() <= 1.0, "output must stay within [0,1] per §18"
    assert abs(out[0, 0, 0] - 0.5) < 1e-5


def test_normalization_db_clip_minmax_missing_params_raises():
    import numpy as np

    image = np.zeros((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_normalization(image, NormalizationConfig(method="db_clip_minmax"))


def test_normalization_unknown_method_raises():
    import numpy as np

    image = np.zeros((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_normalization(image, NormalizationConfig(method="not_a_real_method"))
