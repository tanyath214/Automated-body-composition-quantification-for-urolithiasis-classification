"""Preprocessing helpers for 2D CT slices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_image_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        key = "image" if "image" in data.files else data.files[0]
        return data[key]
    return np.asarray(Image.open(path))


def normalize_to_unit(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    min_value = float(np.nanmin(image))
    max_value = float(np.nanmax(image))
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - min_value) / (max_value - min_value), 0.0, 1.0)


def preprocess_slice(image: np.ndarray, input_size: int = 96) -> torch.Tensor:
    image = torch.from_numpy(normalize_to_unit(image)).float().unsqueeze(0).unsqueeze(0)
    image = F.interpolate(image, size=(input_size, input_size), mode="bilinear", align_corners=False)
    return image


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, mask.astype(np.uint8))
    else:
        Image.fromarray(mask.astype(np.uint8)).save(path)
