from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import CTSliceDataset
from losses import HybridLoss


def test_dataset_loads_paired_npy_files(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    np.save(image_dir / "case_001.npy", np.random.rand(64, 64).astype(np.float32))
    np.save(mask_dir / "case_001.npy", np.zeros((64, 64), dtype=np.uint8))

    dataset = CTSliceDataset(image_dir, mask_dir, input_size=96, augment=False)
    sample = dataset[0]

    assert tuple(sample["image"].shape) == (1, 96, 96)
    assert tuple(sample["mask"].shape) == (96, 96)


def test_hybrid_loss_returns_scalar() -> None:
    criterion = HybridLoss(num_classes=4)
    logits = torch.randn(2, 4, 96, 96, requires_grad=True)
    target = torch.randint(0, 4, (2, 96, 96))
    loss = criterion(logits, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
