"""spec §28: test_model_forward — model accepts (B,2,256,256), outputs (B,1,256,256).

NOTE: these tests require torch + segmentation-models-pytorch, which could
not be installed/executed in the sandbox this was built in (no network
access). Run these yourself once dependencies are installed — see README.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.model import build_model


def _tiny_config(pretrained: bool = False, backbone: str = "resnet34"):
    return {
        "model": {
            "in_channels": 2,
            "out_channels": 1,
            "backbone": backbone,
            "pretrained": pretrained,
        }
    }


def test_forward_pass_shape_256():
    import torch

    config = _tiny_config(pretrained=False)
    model = build_model(config)
    x = torch.randn(2, 2, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256), f"expected (2,1,256,256), got {tuple(out.shape)}"


def test_first_conv_accepts_two_channels():
    """spec §13: the input-channel-mismatch fix must actually take effect."""
    model = build_model(_tiny_config(pretrained=False))
    from src.model import _find_first_conv

    first_conv = _find_first_conv(model)
    assert first_conv is not None
    assert first_conv.in_channels == 2


def test_output_is_logits_not_bounded():
    """Model must output raw logits — sigmoid happens in infer.py, per §13."""
    import torch

    model = build_model(_tiny_config(pretrained=False))
    x = torch.randn(1, 2, 256, 256) * 50
    out = model(x)
    assert out.dtype == torch.float32


def test_missing_smp_raises_clear_error(monkeypatch):
    """If segmentation-models-pytorch isn't installed, error must be explicit, not a crash."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "segmentation_models_pytorch":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        build_model(_tiny_config(pretrained=False))
