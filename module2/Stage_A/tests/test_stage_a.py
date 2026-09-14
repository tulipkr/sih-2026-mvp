import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import rasterio
from affine import Affine

from preprocess_pipeline import StageAError, run_stage_a
from src.safe_reader import SafeProductError, locate_safe_product
from src.calibration import CalibrationError, calibrate_band
from src.georeferencing import GeoreferencingError
from tests.fixtures import build_fake_safe_product

DEFAULT_CFG = {
    "speckle_filter": {"apply": False},  # off in most tests: isolates the behavior under test
    "normalization": {"apply_db_clipping": False, "apply_normalization": False},
    "land_mask": {"source_path": None},
    "scene_is_coastal": False,  # most synthetic scenes here aren't real coastlines
}


# --- 1. Valid VV + VH input --------------------------------------------------
def test_valid_safe_product_end_to_end(tmp_path):
    safe_root = build_fake_safe_product(tmp_path / "raw", height=512, width=512)
    out_dir = tmp_path / "out"
    manifest_path = run_stage_a(str(out_dir), safe_path=str(safe_root), patch_size=256, config=DEFAULT_CFG)

    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 4  # 512x512 at patch_size=256 -> 2x2 tiles
    for entry in manifest:
        assert entry["bands"] == ["VV", "VH"]
        assert entry["scene_id"] == "S1B_IW_GRDH_1SDV_20211003T014927_20211003T014952_028965_0374DA_3EE4"
        assert entry["acquisition_timestamp_utc"] == "2021-10-03T01:49:27Z"  # real, parsed, not fabricated
        with rasterio.open(entry["patch_path"]) as src:
            assert src.count == 2  # wrong-channel-count check: must always be exactly VV+VH
            assert src.width == 256 and src.height == 256


# --- 2/3. Missing VV / Missing VH -------------------------------------------
def test_missing_vh_band_raises_explicit_error(tmp_path):
    safe_root = build_fake_safe_product(tmp_path / "raw", include_vh=False)
    with pytest.raises(SafeProductError, match="vh"):
        run_stage_a(str(tmp_path / "out"), safe_path=str(safe_root), config=DEFAULT_CFG)


def test_missing_vv_measurement_raises(tmp_path):
    safe_root = build_fake_safe_product(tmp_path / "raw")
    # Delete the VV measurement file after building a normally-valid product.
    vv_files = list((safe_root / "measurement").glob("*vv*"))
    for f in vv_files:
        f.unlink()
    with pytest.raises(SafeProductError, match="vv"):
        run_stage_a(str(tmp_path / "out"), safe_path=str(safe_root), config=DEFAULT_CFG)


# --- 4/7. Mismatched VV/VH dimensions / input dimension mismatch -----------
def test_mismatched_vv_vh_dimensions_raises(tmp_path):
    safe_root = build_fake_safe_product(tmp_path / "raw", height=512, width=512)
    # Overwrite the VH file with a different size.
    vh_files = list((safe_root / "measurement").glob("*vh*"))
    profile = {
        "driver": "GTiff", "height": 256, "width": 256, "count": 1,
        "dtype": rasterio.float32, "crs": "EPSG:32611",
        "transform": Affine(10.0, 0, 500000, 0, -10.0, 3700000),
    }
    with rasterio.open(vh_files[0], "w", **profile) as dst:
        dst.write(np.full((256, 256), 500.0, dtype=np.float32), 1)

    with pytest.raises(StageAError, match="mismatched"):
        run_stage_a(str(tmp_path / "out"), safe_path=str(safe_root), config=DEFAULT_CFG)


def test_scene_smaller_than_patch_size_produces_zero_tiles_error(tmp_path):
    safe_root = build_fake_safe_product(tmp_path / "raw", height=100, width=100)
    with pytest.raises(StageAError, match="zero usable tiles"):
        run_stage_a(str(tmp_path / "out"), safe_path=str(safe_root), patch_size=256, config=DEFAULT_CFG)


# --- 5. Missing/corrupt georeferencing --------------------------------------
def test_missing_georeferencing_raises(tmp_path):
    """No CRS AND no GCPs at all — genuinely ungeoreferenced input, a hard
    error per spec §16 ('CRS missing after calibration -> cannot proceed to
    tiling'). Raised by georeferencing.resolve_geotransform specifically,
    since that's the layer that actually knows there's no fallback left."""
    safe_root = build_fake_safe_product(tmp_path / "raw", valid_georeferencing=False)
    with pytest.raises(GeoreferencingError, match="georeferencing"):
        run_stage_a(str(tmp_path / "out"), safe_path=str(safe_root), config=DEFAULT_CFG)


# --- 8. Multiple patches belonging to one scene -----------------------------
def test_multiple_tiles_share_scene_id_with_unique_patch_id(tmp_path):
    safe_root = build_fake_safe_product(tmp_path / "raw", height=768, width=512)
    manifest_path = run_stage_a(str(tmp_path / "out"), safe_path=str(safe_root), patch_size=256, config=DEFAULT_CFG)
    manifest = json.loads(manifest_path.read_text())

    assert len(manifest) == 6  # 768x512 -> 3 rows x 2 cols of 256 tiles
    scene_ids = {e["scene_id"] for e in manifest}
    patch_ids = [e["patch_id"] for e in manifest]
    assert len(scene_ids) == 1                    # all tiles share one scene_id
    assert len(set(patch_ids)) == len(patch_ids)   # every patch_id is unique

    transforms = [tuple(e["transform"]) for e in manifest]
    assert len(set(transforms)) == len(transforms)  # every tile's transform is distinct (spec §19/§38)


# --- 13. Missing metadata (calibration LUT / annotation timestamp) ---------
def test_missing_calibration_xml_raises_explicit_error(tmp_path):
    safe_root = build_fake_safe_product(tmp_path / "raw", include_calibration=False)
    with pytest.raises(SafeProductError, match="calibration"):
        run_stage_a(str(tmp_path / "out"), safe_path=str(safe_root), config=DEFAULT_CFG)


def test_missing_start_time_in_annotation_raises(tmp_path):
    safe_root = build_fake_safe_product(tmp_path / "raw")
    for ann in (safe_root / "annotation").glob("*vv*.xml"):
        ann.write_text("<product><adsHeader></adsHeader></product>")  # no startTime
    with pytest.raises(SafeProductError, match="startTime"):
        run_stage_a(str(tmp_path / "out"), safe_path=str(safe_root), config=DEFAULT_CFG)


# --- 14. Invalid/nonexistent SAFE path --------------------------------------
def test_nonexistent_safe_path_raises():
    with pytest.raises(SafeProductError, match="does not exist"):
        locate_safe_product("/definitely/not/a/real/path.SAFE", "/tmp/scratch_never_used")


def test_run_stage_a_with_nonexistent_safe_path(tmp_path):
    with pytest.raises(SafeProductError):
        run_stage_a(str(tmp_path / "out"), safe_path="/definitely/not/a/real/path.SAFE", config=DEFAULT_CFG)


# --- Calibration correctness (known DN + LUT -> hand-computable Sigma0) ----
def test_calibration_known_value(tmp_path):
    """spec §28 test_calibration: known synthetic DN + constant -> expected
    Sigma0 value, sigma0 = DN^2 / sigmaNought^2."""
    safe_root = build_fake_safe_product(
        tmp_path / "raw", height=64, width=64, dn_value=1000.0, sigma_naught_value=100.0, seed=None,
    )
    # dn_value has added noise in the fixture; use a zero-noise direct check instead:
    dn = np.full((64, 64), 1000.0, dtype=np.float64)
    calib_xml = next((safe_root / "annotation" / "calibration").glob("*vv*"))
    sigma0 = calibrate_band(dn, calib_xml)
    expected = (1000.0 ** 2) / (100.0 ** 2)  # = 100.0
    assert np.allclose(sigma0, expected, rtol=1e-4)


# --- Land masking blocking-error behavior (spec §15) ------------------------
def test_missing_land_mask_source_blocks_coastal_scene(tmp_path):
    safe_root = build_fake_safe_product(tmp_path / "raw", height=256, width=256)
    cfg = dict(DEFAULT_CFG, land_mask={"source_path": None}, scene_is_coastal=True)
    with pytest.raises(Exception, match="[Ll]and.mask|[Cc]oastal"):
        run_stage_a(str(tmp_path / "out"), safe_path=str(safe_root), config=cfg)


def test_missing_land_mask_source_allowed_for_non_coastal_scene(tmp_path):
    safe_root = build_fake_safe_product(tmp_path / "raw", height=256, width=256)
    cfg = dict(DEFAULT_CFG, land_mask={"source_path": None}, scene_is_coastal=False)
    manifest_path = run_stage_a(str(tmp_path / "out"), safe_path=str(safe_root), config=cfg)
    assert manifest_path.exists()
