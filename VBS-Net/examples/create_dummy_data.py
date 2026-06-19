"""Create synthetic 2D CT-like slices for repository smoke testing.

The generated images are not medical data and must not be used to estimate
clinical performance. They only verify that the dataset, training, and inference
code can run without requiring protected CT data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def ellipse_mask(height: int, width: int, center: tuple[float, float], radius: tuple[float, float]) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    cy, cx = center
    ry, rx = radius
    return (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0


def make_case(rng: np.random.Generator, size: int = 96) -> tuple[np.ndarray, np.ndarray]:
    image = rng.normal(loc=0.08, scale=0.025, size=(size, size)).astype(np.float32)
    mask = np.zeros((size, size), dtype=np.uint8)

    sat = ellipse_mask(size, size, (size * 0.50, size * 0.50), (size * 0.43, size * 0.36))
    muscle = ellipse_mask(size, size, (size * 0.49, size * 0.50), (size * 0.30, size * 0.25))
    vat = ellipse_mask(size, size, (size * 0.51, size * 0.50), (size * 0.20, size * 0.17))

    # Labels follow the default class order: 0 background, 1 muscle, 2 SAT, 3 VAT.
    mask[sat] = 2
    mask[muscle] = 1
    mask[vat] = 3

    image[sat] += 0.15
    image[muscle] += 0.45
    image[vat] += 0.25
    image += rng.normal(loc=0.0, scale=0.02, size=(size, size)).astype(np.float32)
    return np.clip(image, 0.0, 1.0), mask


def write_split(root: Path, split: str, count: int, rng: np.random.Generator, size: int) -> None:
    image_dir = root / split / "images"
    mask_dir = root / split / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        image, mask = make_case(rng, size=size)
        case_id = f"{split}_{index:03d}"
        np.save(image_dir / f"{case_id}.npy", image)
        np.save(mask_dir / f"{case_id}.npy", mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic data for VBS-Net smoke tests.")
    parser.add_argument("--output-dir", default="data/dummy", help="Directory to write synthetic train/val data.")
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.output_dir)
    rng = np.random.default_rng(args.seed)
    write_split(root, "train", args.train_count, rng, args.size)
    write_split(root, "val", args.val_count, rng, args.size)
    print(f"Wrote synthetic data to {root}")


if __name__ == "__main__":
    main()
