"""
Training entrypoint (spec §10, §25, §37).

Usage:
    python -m src.train --config configs/config.yaml
"""
from __future__ import annotations
import numpy as np
import json
import argparse
import logging
from logging import config
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import (
    load_manifest_entries,
    train_val_split,
    NormalizationConfig,
)
from src.dataset import SARSegmentationDataset
from pretiled_dataset import PreTiledDataset
from src.losses import build_loss
from src.metrics import aggregate_metrics, compute_metrics, false_positive_rate_on_lookalike
from src.model import build_model
from src.utils import load_config, resolve_device, set_seed, setup_logging, write_json

logger = logging.getLogger(__name__)


def _build_dataset(entries, config, require_mask=True, augmentation=None):
    sample_path = entries[0].get("image_path") or entries[0].get("patch_path")
    if str(sample_path).endswith(".pt"):
        norm_cfg = NormalizationConfig.from_dict(config["data"].get("normalization"))
        return PreTiledDataset(
            entries,
            norm_cfg,
            require_mask=require_mask,
            augmentation=augmentation,
        )    
    norm_cfg = NormalizationConfig.from_dict(config["data"].get("normalization"))
    return SARSegmentationDataset(
        entries,
        norm_cfg,
        config["data"]["bands"],
        patch_size=config["data"]["patch_size"],
        require_mask=require_mask,
    )

def run_training(config_path: str) -> dict:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    config = load_config(config_path)
    setup_logging(config["logging"]["log_dir"], config["logging"].get("level", "INFO"))
    set_seed(config["data"]["seed"])

    device = resolve_device(config["train"].get("device", "auto"))
    logger.info(f"Using device: {device}")

    # Scene-level split:
    # Train and validation manifests are created BEFORE tiling,
    # so tiles from the same scene never appear in both sets.

    if "train_manifest_path" in config["data"] and "val_manifest_path" in config["data"]:
        train_manifest_path = config["data"]["train_manifest_path"]
        val_manifest_path = config["data"]["val_manifest_path"]
    else:
        # Backward compatibility for older tests/configs
        train_manifest_path = config["data"]["manifest_path"]
        val_manifest_path = config["data"]["manifest_path"]

    with open(train_manifest_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(val_manifest_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    train_entries = train_data["entries"]
    val_entries = val_data["entries"]

    logger.info(
        f"Scene-level split: {len(train_entries)} train tiles, "
        f"{len(val_entries)} validation tiles"
    )
    
    norm_cfg = NormalizationConfig.from_dict(config["data"].get("normalization"))
    if norm_cfg.method != "none":
        logger.warning(
            f"normalization.method='{norm_cfg.method}' is active. This is marked PENDING "
            f"verification in the spec (§7) — confirm it matches what Ishita's "
            f"preprocessing actually applies before trusting results on real data."
        )

    if "train_manifest_path" in config["data"] and "val_manifest_path" in config["data"]:
        train_ds = _build_dataset(train_entries, config, require_mask=True, augmentation=config["data"].get("augmentation"))
        val_ds = _build_dataset(
            val_entries if val_entries else train_entries,
            config,
            require_mask=True,
            augmentation={},
        )
    else:
        train_ds = SARSegmentationDataset(
            train_entries,
            norm_cfg,
            config["data"]["bands"],
            patch_size=config["data"]["patch_size"],
        )
        val_ds = SARSegmentationDataset(
            val_entries if val_entries else train_entries,
            norm_cfg,
            config["data"]["bands"],
            patch_size=config["data"]["patch_size"],
        )

    # Compute the actual positive-tile fraction from the training manifest.
    positive_tiles = 0
    for entry in train_entries:
        mask_path = entry.get("mask_path")

        if not mask_path:
            continue

        if str(mask_path).endswith(".pt"):
            mask = torch.load(mask_path, weights_only=False)
            has_oil = mask.float().mean().item() > 0
        else:
            import rasterio

            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            has_oil = np.any(mask > 0)

        if has_oil:
            positive_tiles += 1

    train_positive_fraction = positive_tiles / max(len(train_entries), 1)

    logger.info(
        f"Fraction of training tiles with oil pixels: "
        f"{train_positive_fraction:.3f}"
    )
    class_imbalance = config["data"].get("class_imbalance", {})
    weighted_sampling = class_imbalance.get("weighted_sampling", False)

    if weighted_sampling:
        sample_weights = []

        for entry in train_entries:
            mask_path = entry.get("mask_path")

            if not mask_path:
                sample_weights.append(1.0)
                continue

            if str(mask_path).endswith(".pt"):
                mask = torch.load(mask_path, weights_only=False)
                has_oil = mask.float().mean().item() > 0
            else:
                import rasterio

                with rasterio.open(mask_path) as src:
                    mask = src.read(1)

                has_oil = np.any(mask > 0)

            sample_weights.append(2.0 if has_oil else 1.0)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config["train"]["batch_size"],
            sampler=sampler,
            num_workers=config["train"].get("num_workers", 0),
        )

        logger.info("Weighted sampling enabled for training.")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=config["train"].get("num_workers", 0),
        )

    val_loader = DataLoader(
        val_ds, batch_size=config["train"]["batch_size"], shuffle=False,
        num_workers=config["train"].get("num_workers", 0),
    )

    model = build_model(config).to(device)
    loss_fn = build_loss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    checkpoint_path = Path(config["train"]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_dice = -1.0
    epochs_without_improvement = 0
    patience = config["train"].get("early_stopping_patience", 8)
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    for epoch in range(config["train"]["epochs"]):
        model.train()
        train_losses = []
        for images, masks, _scene_ids in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, batch_metrics = [], []
        with torch.no_grad():
            for images, masks, _scene_ids in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = loss_fn(logits, masks)
                val_losses.append(loss.item())
                batch_metrics.append(compute_metrics(logits, masks))

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        val_metrics = aggregate_metrics(batch_metrics) if batch_metrics else {"dice": 0.0}

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        logger.info(
            f"Epoch {epoch+1}/{config['train']['epochs']} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_metrics.get('dice', 0):.4f} "
            f"val_iou={val_metrics.get('iou', 0):.4f}"
        )

        if val_metrics.get("dice", 0) > best_val_dice:
            best_val_dice = val_metrics.get("dice", 0)
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config,
                 "epoch": epoch, "val_dice": best_val_dice},
                checkpoint_path,
            )
            logger.info(f"New best model saved (val_dice={best_val_dice:.4f}) -> {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

    if not checkpoint_path.exists():
        raise RuntimeError(
            "Training finished but no checkpoint was ever saved — val_dice never "
            "improved from its initial -1.0. This indicates a bug (e.g. an empty "
            "validation set), not a real training result."
        )

    # Spec §3/§29: false-positive check on look-alike/no-oil set, if configured
    lookalike_report = None
    lookalike_path = config["train"].get("lookalike_manifest_path")
    if lookalike_path and Path(lookalike_path).exists():
        lookalike_entries = load_manifest_entries(lookalike_path)
        lookalike_ds = _build_dataset(lookalike_entries, config, require_mask=True, augmentation={})
        lookalike_loader = DataLoader(lookalike_ds, batch_size=config["train"]["batch_size"])
        positive_fractions = []
        model.eval()
        with torch.no_grad():
            for images, _masks, _scene_ids in lookalike_loader:
                images = images.to(device)
                logits = model(images)
                preds = (torch.sigmoid(logits) > config["inference"]["threshold"]).float()
                for p in preds:
                    positive_fractions.append(p.mean().item())
        lookalike_report = false_positive_rate_on_lookalike(positive_fractions)
        logger.info(f"Look-alike false-positive check: {lookalike_report}")
    else:
        logger.warning(
            "No look-alike manifest found — false-positive check (spec §3/§29) was "
            "skipped. Set train.lookalike_manifest_path in config.yaml and generate "
            "data before treating this model as demo-ready."
        )

    report = {
        "history": history,
        "best_val_dice": best_val_dice,
        "checkpoint_path": str(checkpoint_path),
        "train_positive_fraction": train_positive_fraction,
        "lookalike_false_positive_check": lookalike_report,
    }
    metrics_path = Path(config["inference"]["output_dir"]) / "metrics" / "training_metrics.json"
    write_json(metrics_path, report)
    logger.info(f"Training complete. Best val_dice={best_val_dice:.4f}. Report: {metrics_path}")

    return {"best_ckpt_path": str(checkpoint_path), "best_val_dice": best_val_dice, "report": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the U-Net segmentation model.")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_training(args.config)
