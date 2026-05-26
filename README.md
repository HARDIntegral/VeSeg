# VeSeg

VeSeg is a Python package for vascular segmentation and vessel geometry extraction. The package combines a pretrained U-Net with a Rust backend to generate vessel masks, skeletons, vascular graphs, and radius estimates from microscopy and visible-light vascular images.

The long-term purpose of VeSeg is to convert raw vascular images into representations usable for:

- Oxygen and nutrient transport simulations
- Tissue diffusion modeling
- Vascular topology extraction
- Vessel skeletonization and graph generation
- Vessel radius and diameter estimation
- Biomedical image analysis
- Future blood flow and coupled transport simulations

---

## Pipeline

```text
Image
↓
Preprocessing
↓
U-Net segmentation
↓
Binary vessel mask
↓
Skeletonization (Rust)
↓
Distance transform
↓
Graph extraction
↓
Radius estimation
↓
Simulation-ready geometry
```

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
- Rust-accelerated skeletonization
- Vessel graph extraction
- Vessel radius estimation
- Bundled geometry extraction backend

- Distance transform generation
- Radius-annotated edge tables
- Vessel reconstruction from skeleton + radius estimates

---

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/HARDIntegral/VeSeg.git
```

> **Note:** Prebuilt wheels and bundled binaries are currently tested primarily on Apple Silicon (ARM64 macOS). Installation on other systems may require compiling the Rust extension locally.
>
> For non-Apple Silicon systems or development workflows, the recommended installation method is the following development installation.

### Development installation

```bash
git clone https://github.com/HARDIntegral/VeSeg.git
cd VeSeg

python -m venv .venv
source .venv/bin/activate

pip install maturin
maturin develop
```

Build wheel:

```bash
maturin build --release --out target/wheels
```

Run tests:

```bash
pytest
```

---

## Quick Start

Predict vessels and extract geometry:

```python
import veseg

result = veseg.predict(
	"test_images/structure1.png",
	mode=veseg.ENHANCED_INVERTED,
)

mask, skeleton, distance_map, nodes, edges = (
	veseg.extract_vessel_geometry(result.mask)
)

print("nodes:", nodes.shape)
print("edges:", edges.shape)
print("mean radii:", edges[:, 4])
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

Radius estimates are returned inside the edge table:

```python
mean_radius = edges[:, 4]
min_radius = edges[:, 5]
max_radius = edges[:, 6]
```

---

## Public API

Available:

```python
veseg.predict()
veseg.load_model()

veseg.extract_vessel_geometry()
veseg.build_vessel_geometry()
veseg.build_vessel_graph()

veseg.skeletonize_mask()
veseg.distance_transform_mask()
veseg.reconstruct_vessel_mask()

veseg.postprocess_prediction()
veseg.scale_mask()
veseg.resize_mask()
veseg.remove_specks()
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

Detailed API documentation, parameters, return values, graph schemas, and examples:

```text
DOCS.md
```

---

## License

MIT License