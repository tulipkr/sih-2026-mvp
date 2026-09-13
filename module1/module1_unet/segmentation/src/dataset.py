"""
Dataset loader (spec §5, §6, §7, §11, §12).

Loads manifest entries, reads 2-band VV/VH GeoTIFF patches + binary masks,
applies fully-configurable normalization (default: none — see config.yaml's
warning about §7's PENDING ZENODO VERIFICATION note), and tiles oversized
training images down to patch_size (§12).

This file never hardcodes a clip range, mean/std, or filter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DatasetValidationError(ValueError):
    """Raised when an input file or manifest entry fails a §11 sanity check."""


@dataclass
class NormalizationConfig:
    method: str = "none"          # none | db_clip_minmax
    db_clip_min: float | None = None
    db_clip_max: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "NormalizationConfig":
        d = d or {}
        return cls(
            method=d.get("method", "none"),
            db_clip_min=d.get("db_clip_min"),
            db_clip_max=d.get("db_clip_max"),
        )


def apply_normalization(image: np.ndarray, cfg: NormalizationConfig) -> np.ndarray:
    """
    method == "none": pass-through (only a float32 cast happens elsewhere).
    method == "db_clip_minmax": clip to [db_clip_min, db_clip_max], rescale to [0, 1].
        This is the spec's §12/§25 example convention — NOT applied by default,
        since §7 marks it PENDING verification against the real dataset.
    """
    if cfg.method == "none":
        return image

    if cfg.method == "db_clip_minmax":
        if cfg.db_clip_min is None or cfg.db_clip_max is None:
            raise ValueError(
                "normalization.method is 'db_clip_minmax' but db_clip_min/db_clip_max "
                "are not set in config.yaml. Set both explicitly — do not guess a range."
            )
        if cfg.db_clip_max <= cfg.db_clip_min:
            raise ValueError(
                f"normalization.db_clip_max ({cfg.db_clip_max}) must be greater than "
                f"db_clip_min ({cfg.db_clip_min})."
            )
        clipped = np.clip(image, cfg.db_clip_min, cfg.db_clip_max)
        return (clipped - cfg.db_clip_min) / (cfg.db_clip_max - cfg.db_clip_min)

    raise ValueError(
        f"Unknown normalization.method '{cfg.method}'. Expected 'none' or 'db_clip_minmax'."
    )


def _load_geotiff(
    path: str | Path,
    window=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import rasterio

    path = Path(path)
    if not path.exists():
        raise DatasetValidationError(f"Patch file not found: {path.resolve()}")

    try:
        with rasterio.open(path) as src:
            if window is None:
                arr = src.read().astype(np.float32)
            else:
                arr = src.read(window=window).astype(np.float32)

            meta = {
                "crs": str(src.crs) if src.crs else None,
                "transform": list(src.transform)[:6] if src.transform else None,
                "width": src.width,
                "height": src.height,
                "count": src.count,
            }

    except rasterio.errors.RasterioIOError as e:
        raise DatasetValidationError(
            f"Could not read GeoTIFF at {path}: {e}"
        ) from e

    return arr, meta

def _load_mask(path: str | Path, window=None) -> np.ndarray:
    import rasterio

    path = Path(path)
    if not path.exists():
        raise DatasetValidationError(f"Mask file not found: {path.resolve()}")

    with rasterio.open(path) as src:
        arr = src.read(1, window=window).astype(np.float32)

    unique_vals = np.unique(arr)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise DatasetValidationError(
            f"Mask {path} contains non-binary values {unique_vals.tolist()}. "
            f"Expected only {{0, 1}} (spec §11)."
        )
    return arr.astype(np.uint8)


def _tile_indices(h: int, w: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    """Sliding-window top-left corners covering an (h, w) image at patch_size."""
    ys = list(range(0, max(h - patch_size, 0) + 1, stride)) or [0]
    xs = list(range(0, max(w - patch_size, 0) + 1, stride)) or [0]
    if ys[-1] + patch_size < h:
        ys.append(h - patch_size)
    if xs[-1] + patch_size < w:
        xs.append(w - patch_size)
    return [(y, x) for y in ys for x in xs]


class SARSegmentationDataset:
    """
    torch-compatible Dataset over manifest entries.

    Training entries additionally need "mask_path" — this is this module's
    own addition for pairing image+label at training time (not part of
    Ishita's inference-time manifest contract, per spec §7).

    If a manifest entry's image is larger than patch_size, it is tiled
    (spec §12) into patch_size x patch_size crops at dataset-build time.
    Inference-time patches from Ishita are assumed already exactly
    patch_size x patch_size (handled by infer.py, not here).
    """

    def __init__(
        self,
        manifest_entries: list[dict[str, Any]],
        normalization: NormalizationConfig,
        expected_bands: list[str],
        patch_size: int,
        require_mask: bool = True,
        tiling_strategy: str = "sliding_window",
        tiling_stride: int | None = None,
        weighted_sampling: bool = False,
        min_oil_fraction_target: float = 0.3,
        seed: int = 42,
    ):
        self.normalization = normalization
        self.expected_bands = expected_bands
        self.patch_size = patch_size
        self.require_mask = require_mask
        self.tiling_strategy = tiling_strategy
        self.tiling_stride = tiling_stride or patch_size
        self.weighted_sampling = weighted_sampling
        self.min_oil_fraction_target = min_oil_fraction_target
        self._rng = np.random.RandomState(seed)

        self._validate_manifest(manifest_entries)
        self.entries = manifest_entries
        # (entry_idx, y, x) tuples — resolved lazily per __getitem__, avoids
        # reading every training image up front just to compute tile grids.
        self._index: list[tuple[int, int, int]] | None = None

    def _validate_manifest(self, entries: list[dict[str, Any]]) -> None:
        if len(entries) == 0:
            raise DatasetValidationError(
                "Manifest produced zero usable entries. Check manifest_path in config.yaml "
                "and that the referenced files actually exist."
            )
        for i, entry in enumerate(entries):
            required = ["scene_id", "patch_path", "bands"]
            if self.require_mask:
                required.append("mask_path")
            missing = [k for k in required if k not in entry]
            if missing:
                raise DatasetValidationError(
                    f"Manifest entry {i} (scene_id={entry.get('scene_id', '?')}) is missing "
                    f"required field(s): {missing}"
                )
            if entry["bands"] != self.expected_bands:
                raise DatasetValidationError(
                    f"Manifest entry {i} (scene_id={entry['scene_id']}) has bands "
                    f"{entry['bands']}, but config expects {self.expected_bands}. "
                    f"Band order matters (spec §7) — do not assume it silently matches."
                )

    def _build_tile_index(self) -> list[tuple[int, int, int]]:
            
        index: list[tuple[int, int, int]] = []

        import rasterio

        for i, entry in enumerate(self.entries):
            # Only read image metadata, not the entire image.
            try:
                with rasterio.open(entry["patch_path"]) as src:
                    h = src.height
                    w = src.width
            except rasterio.errors.RasterioIOError as exc:
                raise DatasetValidationError(
                    f"Could not open patch file: {entry['patch_path']}"
                ) from exc
            
            if h == self.patch_size and w == self.patch_size:
                index.append((i, 0, 0))
                continue

            if h < self.patch_size or w < self.patch_size:
                raise DatasetValidationError(
                    f"scene_id={entry['scene_id']}: image is {h}x{w}, smaller than "
                    f"patch_size={self.patch_size}. Cannot tile an undersized image."
                )

            if self.tiling_strategy == "sliding_window":
                for y, x in _tile_indices(
                    h, w, self.patch_size, self.tiling_stride
                ):
                    index.append((i, y, x))

            elif self.tiling_strategy == "random_crop":
                index.append((i, -1, -1))

            else:
                raise ValueError(
                    f"Unknown tiling.strategy '{self.tiling_strategy}'"
                )

        return index

    def __len__(self) -> int:
        if self._index is None:
            self._index = self._build_tile_index()
        return len(self._index)

    def __getitem__(self, idx: int):
        import torch

        if self._index is None:
            self._index = self._build_tile_index()
        entry_idx, y, x = self._index[idx]
        entry = self.entries[entry_idx]

        if y == -1:  # random_crop
            import rasterio

            with rasterio.open(entry["patch_path"]) as src:
                h = src.height
                w = src.width

            y = int(self._rng.randint(0, max(h - self.patch_size, 0) + 1))
            x = int(self._rng.randint(0, max(w - self.patch_size, 0) + 1))

        from rasterio.windows import Window

        window = Window(
            x,
            y,
            self.patch_size,
            self.patch_size
        )

        image, meta = _load_geotiff(
            entry["patch_path"],
            window=window
        )

        if image.shape[0] != len(self.expected_bands):
            raise DatasetValidationError(
                f"scene_id={entry['scene_id']}: patch has {image.shape[0]} bands, "
                f"expected {len(self.expected_bands)} ({self.expected_bands})."
            )
        if not np.isfinite(image).all():
            n_bad = np.size(image) - np.count_nonzero(np.isfinite(image))
            raise DatasetValidationError(
                f"scene_id={entry['scene_id']} tile ({y},{x}) contains {n_bad} NaN/Inf "
                f"pixel values. Rejecting rather than silently zero-filling (spec §11)."
            )

        image = apply_normalization(image, self.normalization)

        if self.require_mask:
            mask = _load_mask(entry["mask_path"], window=window)
            if mask.shape != (self.patch_size, self.patch_size):
                raise DatasetValidationError(
                    f"scene_id={entry['scene_id']}: mask tile shape {mask.shape} does not "
                    f"match image tile shape ({self.patch_size}, {self.patch_size})."
                )
            return (
                torch.from_numpy(image.copy()).float(),
                torch.from_numpy(mask.copy()).float().unsqueeze(0),
                entry["scene_id"],
            )

        return torch.from_numpy(image.copy()).float(), entry["scene_id"], meta

    def check_positive_fraction(self) -> float:
        """
        Spec §11 dataset-level check: confirm a non-trivial fraction of training
        tiles actually contain oil pixels. Raises if the sampling strategy looks broken.
        """
        if not self.require_mask:
            raise ValueError("check_positive_fraction requires require_mask=True")
        n = len(self)
        n_positive = 0
        for i in range(n):
            _img, mask, _sid = self[i]
            if mask.sum() > 0:
                n_positive += 1
        fraction = n_positive / n if n > 0 else 0.0
        if fraction == 0.0:
            raise DatasetValidationError(
                "Zero training tiles contain any oil pixels. The sampling strategy is "
                "broken and training will collapse to all-zero predictions (spec §11)."
            )
        return fraction


def load_manifest_entries(manifest_path: str | Path) -> list[dict[str, Any]]:
    import json

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise DatasetValidationError(f"Manifest not found: {manifest_path.resolve()}")
    with manifest_path.open("r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "entries" in data:
        entries = data["entries"]
    elif isinstance(data, list):
        entries = data
    else:
        raise DatasetValidationError(
            f"Manifest at {manifest_path} is neither a list of entries nor a dict with "
            f"an 'entries' key — unrecognized manifest shape."
        )

    for i, entry in enumerate(entries):
        for field in ["acquisition_timestamp_utc", "crs", "transform"]:
            if field not in entry or entry[field] is None:
                logger.warning(
                    f"Manifest entry {i} (scene_id={entry.get('scene_id', '?')}) is missing "
                    f"'{field}'. Training can proceed, but inference on this entry will set "
                    f"geolocation_incomplete=true (spec §7)."
                )
    return entries


def train_val_split(
    entries: list[dict[str, Any]], val_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    rng = np.random.RandomState(seed)
    indices = np.arange(len(entries))
    rng.shuffle(indices)
    n_val = max(1, int(len(entries) * val_fraction)) if len(entries) > 1 else 0
    val_idx = set(indices[:n_val].tolist())
    train_entries = [e for i, e in enumerate(entries) if i not in val_idx]
    val_entries = [e for i, e in enumerate(entries) if i in val_idx]
    return train_entries, val_entries
