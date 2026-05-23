
# VeSeg

VeSeg is a Python package for biomedical vascular segmentation using deep learning. The package generates vessel masks from microscopy, histology, and visible-light vascular images through a pretrained U-Net segmentation pipeline.

The long-term purpose of VeSeg is to convert raw vascular images into representations usable for:

- Oxygen and nutrient transport simulations
- Tissue diffusion modeling
- Vascular topology extraction
- Vessel skeletonization and graph generation
- Vessel diameter estimation
- Biomedical image analysis
- Future blood flow and transport simulations

---

## Features

Current:

- U-Net based vessel segmentation
- Multiple preprocessing modes
	- Grayscale
	- Inverted
	- Enhanced + inverted
	- Red channel extraction
- Automatic postprocessing
- Speck removal
- Binary mask generation
- Mask scaling utilities
- Installable Python package
- Public prediction API
- Dataset augmentation pipeline
- Threshold sweep optimization during validation

Planned:

- Skeletonization
- Vessel graph extraction
- Vessel radius estimation

---

## Installation

### User installation

Install directly from GitHub:

```bash
pip install git+https://github.com/HARDIntegral/VeSeg.git
```

Or:

```bash
git clone https://github.com/HARDIntegral/VeSeg.git
cd VeSeg
pip install .
```

Verify:

```python
import veseg
```

### Development installation

```bash
git clone https://github.com/HARDIntegral/VeSeg.git
cd VeSeg

python -m venv .venv
source .venv/bin/activate

pip install -e '.[dev]'
```

Run tests:

```bash
pytest
```

---

## Quick Start

Predict vessels from an image:

```python
import veseg

result = veseg.predict(
	"test_images/structure1.png",
	mode=veseg.RED,
)

result.show()
result.save_mask("mask.png")
```

Scale masks:

```python
scaled = result.scaled(2.0)
scaled.save_mask("large_mask.png")
```

Remove isolated false positives:

```python
clean = result.despeckle(min_neighbors=3)
```

---

## Public API

Available:

```python
veseg.predict()
veseg.scale_mask()
veseg.resize_mask()
veseg.remove_specks()
veseg.postprocess_prediction()
```

Prediction modes:

```python
veseg.GRAYSCALE
veseg.INVERTED
veseg.ENHANCED_INVERTED
veseg.RED
```

---

## Model

Current architecture:

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

Training:

```text
Loss:
0.5 BCEWithLogits + 0.5 Dice

Optimizer:
AdamW
```

---

## Tests

Run:

```bash
pytest
```

Current tests include:

- dataset loading
- metrics
- package API validation

---

## Documentation

Detailed package documentation:

```text
DOCS.md
```

---

## Status

VeSeg v0.1

Current focus:

```text
segmentation
→ skeletonization
→ vascular graph extraction
→ transport simulation
```

---

## License

MIT License