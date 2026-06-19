"""Inference script for reconstructed VBS-Net."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models import build_vbsnet
from utils.preprocess import load_image_array, preprocess_slice, save_mask


def load_config(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(model, image_path: str | Path, input_size: int, device: torch.device) -> np.ndarray:
    image = load_image_array(image_path)
    original_shape = image.shape[:2]
    tensor = preprocess_slice(image, input_size=input_size).to(device)
    logits = model(tensor)
    logits = F.interpolate(logits, size=original_shape, mode="bilinear", align_corners=False)
    return torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VBS-Net inference on one 2D CT slice.")
    parser.add_argument("--config", default="configs/vbsnet.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint.")
    parser.add_argument("--image", required=True, help="Path to input image (.npy, .npz, PNG, TIFF, etc.).")
    parser.add_argument("--output", required=True, help="Output mask path, usually .npy or .png.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("config", config)
    model = build_vbsnet(model_config).to(device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    input_size = model_config["model"].get("input_size", 96)
    mask = run_inference(model, args.image, input_size, device)
    save_mask(mask, args.output)
    print(f"Saved segmentation mask to {args.output}")


if __name__ == "__main__":
    main()
