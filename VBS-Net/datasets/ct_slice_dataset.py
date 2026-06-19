"""Dataset for 2D L1/L3 non-contrast CT slices and segmentation masks."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        key = "image" if "image" in data.files else data.files[0]
        return data[key]
    return np.asarray(Image.open(path))


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    min_value = float(np.nanmin(image))
    max_value = float(np.nanmax(image))
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return np.clip(image, 0.0, 1.0)


def _prepare_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int64)


def _resize_pair(image: torch.Tensor, mask: torch.Tensor, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    image = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    mask = F.interpolate(mask[None, None].float(), size=size, mode="nearest").squeeze(0).squeeze(0).long()
    return image, mask


def _random_affine(
    image: torch.Tensor,
    mask: torch.Tensor,
    translate_mm: float,
    pixel_spacing_mm: float,
    scale_range: tuple[float, float],
    rotate_degrees: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    angle = math.radians(random.uniform(-rotate_degrees, rotate_degrees))
    scale = random.uniform(scale_range[0], scale_range[1])
    max_translate_px = translate_mm / max(pixel_spacing_mm, 1e-6)
    tx_px = random.uniform(-max_translate_px, max_translate_px)
    ty_px = random.uniform(-max_translate_px, max_translate_px)

    _, h, w = image.shape
    cos_a = math.cos(angle) / scale
    sin_a = math.sin(angle) / scale
    theta = torch.tensor(
        [
            [cos_a, -sin_a, 2.0 * tx_px / max(w, 1)],
            [sin_a, cos_a, 2.0 * ty_px / max(h, 1)],
        ],
        dtype=torch.float32,
        device=image.device,
    ).unsqueeze(0)

    grid = F.affine_grid(theta, size=(1, 1, h, w), align_corners=False)
    image = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(0)
    mask = F.grid_sample(mask[None, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return image, mask.squeeze(0).squeeze(0).long()


class CTSliceDataset(Dataset):
    """Paired 2D CT slice dataset.

    Expected directory layout:
        images/
          case_001.npy
        masks/
          case_001.npy

    A CSV file with ``image`` and ``mask`` columns can also be used.
    """

    def __init__(
        self,
        images_dir: str | Path | None = None,
        masks_dir: str | Path | None = None,
        csv_file: str | Path | None = None,
        input_size: int | tuple[int, int] = 96,
        augment: bool = False,
        translate_mm: float = 5.0,
        pixel_spacing_mm: float = 1.0,
        scale_range: tuple[float, float] = (0.9, 1.1),
        rotate_degrees: float = 10.0,
    ) -> None:
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else tuple(input_size)
        self.augment = augment
        self.translate_mm = translate_mm
        self.pixel_spacing_mm = pixel_spacing_mm
        self.scale_range = scale_range
        self.rotate_degrees = rotate_degrees
        self.samples = self._collect_samples(images_dir, masks_dir, csv_file)

    def _collect_samples(
        self,
        images_dir: str | Path | None,
        masks_dir: str | Path | None,
        csv_file: str | Path | None,
    ) -> list[tuple[Path, Path]]:
        if csv_file:
            csv_path = Path(csv_file)
            rows: list[tuple[Path, Path]] = []
            with csv_path.open("r", newline="") as f:
                for row in csv.DictReader(f):
                    rows.append((Path(row["image"]), Path(row["mask"])))
            if rows:
                return rows
            raise ValueError(f"No samples found in CSV: {csv_path}")

        if images_dir is None or masks_dir is None:
            raise ValueError("Provide either csv_file or both images_dir and masks_dir.")

        image_root = Path(images_dir)
        mask_root = Path(masks_dir)
        image_paths = sorted(p for p in image_root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        samples = []
        for image_path in image_paths:
            candidates = [mask_root / f"{image_path.stem}{suffix}" for suffix in IMAGE_EXTENSIONS]
            mask_path = next((p for p in candidates if p.exists()), None)
            if mask_path is not None:
                samples.append((image_path, mask_path))
        if not samples:
            raise ValueError(f"No paired image/mask files found under {image_root} and {mask_root}.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.samples[index]
        image = torch.from_numpy(_normalize_image(_load_array(image_path))).float().unsqueeze(0)
        mask = torch.from_numpy(_prepare_mask(_load_array(mask_path))).long()
        image, mask = _resize_pair(image, mask, self.input_size)

        if self.augment:
            image, mask = _random_affine(
                image=image,
                mask=mask,
                translate_mm=self.translate_mm,
                pixel_spacing_mm=self.pixel_spacing_mm,
                scale_range=self.scale_range,
                rotate_degrees=self.rotate_degrees,
            )

        return {"image": image, "mask": mask, "image_path": str(image_path), "mask_path": str(mask_path)}
