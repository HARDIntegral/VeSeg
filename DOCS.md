# VeSeg Documentation

## Overview

VeSeg is a Python package for vascular segmentation from microscopy images and visible-light vessel images. The package provides:

- U-Net based vessel segmentation
- Dataset loading for VessMAP
- Image augmentation and preprocessing
- Probability → binary mask conversion
- Postprocessing and mask cleanup
- Mask scaling and export utilities
- Future skeletonization and transport simulation support

Primary usage:

```python
import veseg

result = veseg.predict(
	"image.png",
	mode=veseg.ENHANCED_INVERTED,
)

result.show()
result.save_mask("mask.png")
```

---

# Installation

## User installation

Install directly from GitHub:

```bash
pip install git+https://github.com/HARDIntegral/VeSeg.git
```

Or install from a local clone:

```bash
git clone https://github.com/HARDIntegral/VeSeg
cd VeSeg

pip install .
```

Verify installation:

```bash
python
```

```python
import veseg
```

---

## Development installation

Clone:

```bash
git clone https://github.com/HARDIntegral/VeSeg
cd VeSeg
```

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install in editable mode:

```bash
pip install -e .
```

Editable installation allows local code changes without reinstalling the package.

---

# Public API

## Prediction

### `veseg.predict(...)`

Run vessel segmentation on an image.

Arguments:

| Parameter | Type | Description |
|-----------|------|-------------|
| path | str \| Path | Input image |
| mode | str | Preprocessing mode |
| threshold | float | Probability threshold |
| checkpoint_path | str \| Path | Model weights |
| image_size | int | Resize before inference |

Returns:

```python
PredictionResult
```

Example:

```python
result = veseg.predict(
	"example.png",
	mode=veseg.RED,
	threshold=0.7,
)
```

---

## Modes

### `veseg.GRAYSCALE`

Uses grayscale image directly.

### `veseg.INVERTED`

Grayscale + inversion.

### `veseg.ENHANCED_INVERTED`

Background subtraction + sharpening + contrast windowing + inversion.

Recommended for:

- visible vessel images
- microscopy with uneven illumination

### `veseg.RED`

Extracts red channel emphasis.

Recommended for:

- blood vessel photographs
- retina images
- visible vascular images

---

# PredictionResult

Returned by:

```python
result = veseg.predict(...)
```

Methods:

## `result.show()`

Display:

- original image
- processed image
- probability map
- binary mask

---

## `result.save_mask(path)`

Export:

```python
result.save_mask("mask.png")
```

---

## `result.scaled(scale)`

Scale mask resolution.

Example:

```python
large = result.scaled(4.0)
```

Useful when exporting masks into simulations.

---

## `result.despeckle()`

Remove isolated false positives.

Example:

```python
clean = result.despeckle(min_neighbors=3)
```

---

## `result.skeletonize()`

Status:

```text
Not implemented
```

Planned:

mask → centerline → graph → transport simulation

---

# Postprocessing

Available:

```python
veseg.remove_specks()
veseg.scale_mask()
veseg.resize_mask()
veseg.save_mask()
veseg.threshold_probability()
```

---

# Training

Training entrypoint:

```bash
python scripts/train_vessmap.py
```

Training uses:

Loss:

```text
0.5 BCEWithLogits
+
0.5 Dice loss
```

Optimizer:

```text
AdamW
```

Evaluation:

```text
Dice
IoU
Pixel accuracy
Threshold sweep
```

---

# Dataset

Expected VessMAP structure:

```text
data/
└── VessMAP/
	├── images/
	├── annotator1/
	│   ├── labels/
	│   └── skeletons/
	└── annotator2/
```

---

# Architecture

Current segmentation architecture:

```text
Input
 ↓
Encoder
 ↓
Bottleneck
 ↓
Decoder + skip connections
 ↓
Pixelwise logits
```

Model:

```text
U-Net
```

# Internal Structure

```text
src/veseg/
├── data.py
├── metrics.py
├── model.py
├── predict.py
├── postprocessing.py
├── preprocessing.py
├── train.py
└── skeleton.py
```

Only `predict.py` and exported API functions in `__init__.py` should be considered stable.