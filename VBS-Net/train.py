"""Train the reconstructed VBS-Net segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from datasets import CTSliceDataset
from losses import HybridLoss
from models import build_vbsnet
from utils.metrics import dice_score
from utils.seed import set_seed


def load_config(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def build_dataset(config: dict, split: str, augment: bool) -> CTSliceDataset:
    data_cfg = config["data"]
    split_cfg = data_cfg.get(split, {})
    aug_cfg = config.get("augmentation", {})
    return CTSliceDataset(
        images_dir=split_cfg.get("images_dir"),
        masks_dir=split_cfg.get("masks_dir"),
        csv_file=split_cfg.get("csv_file"),
        input_size=config["model"].get("input_size", 96),
        augment=augment,
        translate_mm=aug_cfg.get("translate_mm", 5.0),
        pixel_spacing_mm=aug_cfg.get("pixel_spacing_mm", 1.0),
        scale_range=tuple(aug_cfg.get("scale_range", [0.9, 1.1])),
        rotate_degrees=aug_cfg.get("rotate_degrees", 10.0),
    )


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    dice_values = []
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        running_loss += loss.item() * images.size(0)
        batch_dice = dice_score(preds, masks, num_classes)
        foreground = [value for cls, value in batch_dice.items() if cls != 0]
        dice_values.append(sum(foreground) / max(len(foreground), 1))
    return running_loss / max(len(loader.dataset), 1), float(sum(dice_values) / max(len(dice_values), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reconstructed VBS-Net.")
    parser.add_argument("--config", default="configs/vbsnet.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42), deterministic=config.get("deterministic", False))

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    output_dir = Path(config["training"].get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = build_dataset(config, "train", augment=True)
    val_dataset = build_dataset(config, "val", augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get("batch_size", 8),
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"].get("batch_size", 8),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )

    model = build_vbsnet(config).to(device)
    criterion = HybridLoss(num_classes=config["model"].get("num_classes", 4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].get("lr", 1e-4),
        weight_decay=config["training"].get("weight_decay", 1e-5),
    )

    best_dice = -1.0
    epochs = config["training"].get("epochs", 100)
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device, config["model"].get("num_classes", 4))
        print(f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "val_dice": val_dice,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
