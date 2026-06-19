# VBS-Net-Reconstructed

Reconstructed PyTorch implementation of the VBS-Net segmentation framework described in the manuscript. The model is intended for automated segmentation of muscle and adipose tissue on localized L1/L3 non-contrast CT slices.

This repository is not the original proprietary training code. It is a manuscript-consistent reconstruction for research and reproducibility.

## Scope

This repository provides code for the segmentation framework only. It is suitable as a transparent PyTorch reference implementation of the VBS-Net architecture described in the manuscript, but it does not include the original datasets, trained weights, or commercial-platform components.

## Architecture

VBS-Net-Reconstructed includes:

1. A VB-Net-style 2D encoder with four down-sampling blocks.
2. A SAM image encoder wrapper implemented around a ViT encoder interface.
3. Bottleneck-level feature fusion by concatenation.
4. A 1x1 convolution to compress fused features.
5. A VB-Net-style decoder with skip connections.
6. A configurable segmentation head. The default `num_classes` is 4:
   `background`, `muscle`, `SAT`, and `VAT`.

If `segment-anything` and a SAM checkpoint are available, set `use_sam: true` and provide `sam_checkpoint` in `configs/vbsnet.yaml`. Otherwise, the code automatically uses a lightweight ViT-like fallback encoder so the repository can still run.

## Installation

```bash
cd VBS-Net-Reconstructed
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install PyTorch according to your CUDA environment if needed. MONAI is used where helpful for Dice and Focal losses, but the repository also includes fallback loss implementations.

The `segment-anything` package is optional. Install it only if you plan to load an actual SAM checkpoint:

```bash
pip install segment-anything
```

## Data Structure

Input is a 2D non-contrast CT slice already localized to L1 or L3. Images are resized to the configured input size, default `96 x 96`, and normalized to `[0, 1]`.

Default paired directory layout:

```text
data/
  train/
    images/
      case_001.npy
    masks/
      case_001.npy
  val/
    images/
      case_101.npy
    masks/
      case_101.npy
```

Supported formats include `.npy`, `.npz`, PNG, JPG, TIFF, and BMP. Masks should contain integer labels.

Alternatively, provide a CSV with `image` and `mask` columns:

```csv
image,mask
D:/data/case_001.npy,D:/data/case_001_mask.npy
```

Set `csv_file` in `configs/vbsnet.yaml` if using CSV input.

## Synthetic Smoke-Test Data

No real CT data are included. To verify that the repository wiring works without private data, generate toy synthetic arrays:

```bash
python examples/create_dummy_data.py --output-dir data/dummy
```

Then run a one-epoch smoke training job:

```bash
python train.py --config configs/vbsnet_dummy.yaml
```

The dummy data are artificial ellipses with class labels. They are only for testing file I/O, transforms, tensor shapes, loss computation, and checkpoint writing. They are not medical images and cannot be used to estimate segmentation accuracy.

## Training

Edit `configs/vbsnet.yaml` to point to your data, then run:

```bash
python train.py --config configs/vbsnet.yaml
```

Training augmentation follows the manuscript description:

- random translation up to 5 mm
- scaling from 0.9 to 1.1
- rotation from -10 degrees to +10 degrees

The hybrid loss is:

```text
L_hybrid = 0.5 * L_DSC + 0.5 * L_Focal
```

Checkpoints are written to `training.output_dir` as `last.pt` and `best.pt`.

## Inference

```bash
python infer.py ^
  --config configs/vbsnet.yaml ^
  --checkpoint outputs/vbsnet/best.pt ^
  --image data/val/images/case_101.npy ^
  --output outputs/case_101_mask.npy
```

The predicted mask is resized back to the original input slice size before saving.

## Metrics

Utilities are provided for:

- Dice
- precision
- recall
- HD95

See `utils/metrics.py`.

## Tests

The test suite uses synthetic tensors or temporary `.npy` files, so it does not require real CT data:

```bash
pytest tests
```

The tests check the model forward shape, dataset loading, and hybrid loss scalar output. They are engineering smoke tests, not performance validation.

## Model Shape Summary

For the default `96 x 96` input and encoder channels `[32, 64, 128, 256, 512]`:

- Input: `[B, 1, 96, 96]`
- VB-Net bottleneck after four down-sampling blocks: approximately `[B, 512, 6, 6]`
- SAM/fallback ViT feature map is interpolated to the same bottleneck resolution
- Concatenated bottleneck: `[B, 512 + sam_channels, 6, 6]`
- 1x1 fusion compression: `[B, 512, 6, 6]`
- Decoder output logits: `[B, num_classes, 96, 96]`

## Limitations

- This is a reconstructed implementation based on the manuscript, not the original proprietary training code or exact model weights.
- No pretrained VBS-Net weights are included.
- No real CT slices, annotations, or train/validation splits are included.
- Performance depends on preprocessing, CT windowing, annotation consistency, and dataset composition.
- Radiomics feature extraction and classifier construction are intentionally not implemented here. In the manuscript, those steps were performed using the commercial uAI Research Portal platform, so they are outside the scope of this open PyTorch reconstruction.
- Clinical deployment requires independent validation and regulatory review.

## Code Availability Statement Template

This repository contains a reconstructed PyTorch implementation of the VBS-Net segmentation architecture described in the manuscript. It is not the original proprietary implementation and does not include the commercial uAI Research Portal radiomics feature extraction or classifier construction components.
