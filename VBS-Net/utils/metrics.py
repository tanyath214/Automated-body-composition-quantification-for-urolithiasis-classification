"""Segmentation metrics: Dice, precision, recall, and HD95."""

from __future__ import annotations

import numpy as np
import torch


def _as_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def dice_score(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, num_classes: int, eps: float = 1e-6) -> dict[int, float]:
    pred_np = _as_numpy(pred)
    target_np = _as_numpy(target)
    scores = {}
    for cls in range(num_classes):
        pred_c = pred_np == cls
        target_c = target_np == cls
        inter = np.logical_and(pred_c, target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        scores[cls] = float((2.0 * inter + eps) / (denom + eps))
    return scores


def precision_recall(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    num_classes: int,
    eps: float = 1e-6,
) -> tuple[dict[int, float], dict[int, float]]:
    pred_np = _as_numpy(pred)
    target_np = _as_numpy(target)
    precision = {}
    recall = {}
    for cls in range(num_classes):
        pred_c = pred_np == cls
        target_c = target_np == cls
        tp = np.logical_and(pred_c, target_c).sum()
        fp = np.logical_and(pred_c, ~target_c).sum()
        fn = np.logical_and(~pred_c, target_c).sum()
        precision[cls] = float((tp + eps) / (tp + fp + eps))
        recall[cls] = float((tp + eps) / (tp + fn + eps))
    return precision, recall


def _surface_distances(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    if not mask_a.any() or not mask_b.any():
        return np.asarray([np.inf], dtype=np.float32)
    surface_a = np.logical_xor(mask_a, binary_erosion(mask_a))
    surface_b = np.logical_xor(mask_b, binary_erosion(mask_b))
    dt_b = distance_transform_edt(~surface_b)
    return dt_b[surface_a]


def hd95(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, num_classes: int) -> dict[int, float]:
    """Compute symmetric 95th percentile Hausdorff distance per class."""

    pred_np = _as_numpy(pred)
    target_np = _as_numpy(target)
    values = {}
    for cls in range(num_classes):
        pred_c = pred_np == cls
        target_c = target_np == cls
        if not pred_c.any() and not target_c.any():
            values[cls] = 0.0
            continue
        try:
            distances = np.concatenate([_surface_distances(pred_c, target_c), _surface_distances(target_c, pred_c)])
            values[cls] = float(np.percentile(distances, 95))
        except Exception:
            values[cls] = float("nan")
    return values


def segmentation_metrics(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray, num_classes: int) -> dict[str, dict[int, float]]:
    precision, recall = precision_recall(pred, target, num_classes)
    return {
        "dice": dice_score(pred, target, num_classes),
        "precision": precision,
        "recall": recall,
        "hd95": hd95(pred, target, num_classes),
    }
