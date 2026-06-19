from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import VBSNet


def test_vbsnet_forward_shape() -> None:
    model = VBSNet(
        input_size=96,
        num_classes=4,
        encoder_channels=(8, 16, 32, 64, 128),
        sam_channels=32,
        use_sam=False,
    )
    x = torch.randn(2, 1, 96, 96)
    y = model(x)
    assert tuple(y.shape) == (2, 4, 96, 96)
