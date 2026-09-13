"""IoU, Dice, Precision, Recall — spec §3 ("Evaluates on a held-out split
(IoU, Dice, Precision, Recall)")."""
from __future__ import annotations

import torch


def _binarize(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    preds = _binarize(logits, threshold)
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1 - targets_flat)).sum().item()
    fn = ((1 - preds_flat) * targets_flat).sum().item()

    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return {
        "iou": iou, "dice": dice, "precision": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
    }


def aggregate_metrics(metric_dicts: list[dict]) -> dict:
    """Aggregate across batches by re-deriving from summed tp/fp/fn — correct
    for imbalanced data; do not naively average per-batch IoU/Dice."""
    if not metric_dicts:
        raise ValueError("Cannot aggregate an empty list of metrics.")
    tp = sum(m["tp"] for m in metric_dicts)
    fp = sum(m["fp"] for m in metric_dicts)
    fn = sum(m["fn"] for m in metric_dicts)
    eps = 1e-7
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }


def false_positive_rate_on_lookalike(all_preds_positive_fraction: list[float]) -> dict:
    """
    Spec §3/§29: FP check on a look-alike/no-oil set — every sample here has
    NO real oil, so any positive prediction is by definition a false positive.
    Returns the fraction of look-alike samples that triggered a false detection.
    """
    if not all_preds_positive_fraction:
        raise ValueError("Cannot compute false-positive rate on an empty look-alike set.")
    n_false_positive = sum(1 for f in all_preds_positive_fraction if f > 0.0)
    return {
        "n_samples": len(all_preds_positive_fraction),
        "n_false_positive": n_false_positive,
        "false_positive_rate": n_false_positive / len(all_preds_positive_fraction),
    }
