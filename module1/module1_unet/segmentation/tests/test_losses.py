"""spec §28: test_losses — combined loss is non-negative and decreases when
overfitting one batch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


def test_dice_loss_perfect_prediction_near_zero():
    import torch

    from src.losses import DiceLoss

    logits = torch.full((1, 1, 4, 4), 10.0)
    targets = torch.ones((1, 1, 4, 4))
    loss = DiceLoss()(logits, targets)
    assert loss.item() < 0.05


def test_combined_loss_is_nonnegative():
    import torch

    from src.losses import BCEDiceLoss

    logits = torch.randn(2, 1, 8, 8)
    targets = torch.randint(0, 2, (2, 1, 8, 8)).float()
    loss = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)(logits, targets)
    assert loss.item() >= 0.0


def test_loss_decreases_when_overfitting_one_batch():
    import torch

    from src.losses import BCEDiceLoss

    torch.manual_seed(0)
    logits = torch.randn(1, 1, 8, 8, requires_grad=True)
    targets = torch.randint(0, 2, (1, 1, 8, 8)).float()
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.Adam([logits], lr=0.1)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_both_weights_zero_raises():
    from src.losses import BCEDiceLoss

    with pytest.raises(ValueError):
        BCEDiceLoss(bce_weight=0.0, dice_weight=0.0)
