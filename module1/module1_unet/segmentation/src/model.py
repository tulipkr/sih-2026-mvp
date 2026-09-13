"""
U-Net with a pretrained encoder, per spec §13.

Uses segmentation-models-pytorch, which internally handles the "input
channel mismatch" requirement (pretrained encoders expect 3-channel RGB,
this input is 2-channel SAR) via its `in_channels` parameter — when
in_channels != 3, smp replaces the encoder's first conv layer and
initializes it from the mean of the pretrained 3-channel weights,
replicated across the new channel count. That is exactly the behavior
spec §13 describes ("Replace the first conv layer, initializing from the
pretrained weights' mean across channels"), so this file does not
re-implement it manually — it relies on and verifies smp's own behavior
(see build_model's assertion below).

Output activation: NOT applied here — the model returns raw logits.
Sigmoid is applied at inference time (infer.py) and BCEWithLogits is used
in losses.py, both for numerical stability (spec §13).
"""
from __future__ import annotations


def build_model(config: dict):
    """
    Builds the U-Net per config['model']. Raises a clear error (not a silent
    fallback) if segmentation-models-pytorch or its backbone weights aren't
    available — e.g. no internet access to fetch pretrained weights.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise ImportError(
            "segmentation-models-pytorch is required (spec §13/§23) but is not "
            "installed. Run: pip install segmentation-models-pytorch"
        ) from e

    m = config["model"]
    in_channels = m.get("in_channels", 2)
    out_channels = m.get("out_channels", 1)
    backbone = m.get("backbone", "efficientnet-b3")
    pretrained = m.get("pretrained", True)

    encoder_weights = "imagenet" if pretrained else None

    try:
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,  # raw logits — sigmoid applied at inference time only
        )
    except Exception as e:
        if pretrained:
            raise RuntimeError(
                f"Failed to build model with backbone='{backbone}', "
                f"encoder_weights='imagenet'. If this is a network/download error "
                f"(no internet access to fetch pretrained weights), set "
                f"model.pretrained: false in config.yaml and retry — you'll lose "
                f"the pretrained-weight benefit but the architecture will still build. "
                f"Original error: {e}"
            ) from e
        raise

    # Sanity check the exact requirement from spec §13: the first conv layer
    # must actually accept in_channels, not silently fall back to 3.
    first_conv = _find_first_conv(model)
    if first_conv is not None and first_conv.in_channels != in_channels:
        raise RuntimeError(
            f"Model's first conv layer has in_channels={first_conv.in_channels}, "
            f"expected {in_channels}. segmentation-models-pytorch did not adapt the "
            f"input layer as expected — do not proceed with training on this model."
        )

    return model


def _find_first_conv(model):
    """Walk the encoder to find its first Conv2d layer, for the channel-count sanity check."""
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module
    return None
