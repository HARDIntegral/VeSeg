# VeSeg Documentation

## Overview

VeSeg is a Python package for vascular segmentation and vessel geometry extraction.

The package combines:

- pretrained U-Net segmentation
- preprocessing pipelines
- binary mask generation
- Rust-accelerated skeletonization
- Rust distance transforms
- vessel graph extraction
- vessel radius estimation
- vessel reconstruction

The intended workflow:

```text
image
→ preprocessing
→ segmentation
→ binary mask
→ skeleton
→ graph
→ radius estimation
→ transport simulation
```

Typical usage:

```python
import veseg

result = veseg.predict(
    "image.png",
    mode=veseg.ENHANCED_INVERTED,
)

mask, skeleton, distance, nodes, edges = veseg.extract_vessel_geometry(result.mask)
```

---

# Installation

## Install from PyPI

```bash
pip install veseg
```

Verify:

```python
import veseg

print(veseg.predict)

print(veseg.extract_vessel_geometry)
```

---

## Development install

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

Install:

```bash
pip install maturin
maturin develop
```

Build wheel:

```bash
maturin build \
	--release \
	--out target/wheels
```

---

# Public API

Available after:

```python
import veseg
```

Prediction:

```python
veseg.load_model()
veseg.predict()
```

Geometry:

```python
veseg.extract_vessel_geometry()
veseg.build_vessel_geometry()
veseg.build_vessel_graph()

veseg.skeletonize_mask()
veseg.distance_transform_mask()
veseg.reconstruct_vessel_mask()
```

Postprocessing:

```python
veseg.postprocess_prediction()

veseg.threshold_probability()
veseg.remove_specks()

veseg.scale_mask()
veseg.resize_mask()
veseg.save_mask()
```

Constants:

```python
veseg.GRAYSCALE
veseg.INVERTED
veseg.ENHANCED_INVERTED
veseg.RED
```

---

# Prediction Modes

## `veseg.GRAYSCALE`

Use grayscale directly.

Recommended:

- clean microscopy
- already high contrast

Example:

```python
result=veseg.predict(
	"image.png",
	mode=veseg.GRAYSCALE
)
```

---

## `veseg.INVERTED`

Use grayscale + inversion.

Recommended:

- dark vessels

---

## `veseg.ENHANCED_INVERTED`

Pipeline:

```text
grayscale
→ blur
→ local sharpen
→ background subtraction
→ percentile contrast
→ gamma
```

Recommended:

- visible vessels
- uneven illumination

Default:

```python
mode=veseg.ENHANCED_INVERTED
```

---

## `veseg.RED`

Pipeline:

```text
red channel emphasis = red - 0.5 × (green + blue)
```

Recommended:

- retina
- blood vessels
- visible vascular images

---

# Model API

## `veseg.load_model()`

Load pretrained U-Net.

Signature:

```python
load_model(checkpoint_path=DEFAULT_CHECKPOINT_PATH, device=None)
```

Parameters:

| name | type | description |
|------|------|-------------|
| checkpoint_path | str \| Path | custom weights |
| device | torch.device | optional |

Returns:

```python
UNet
```

Notes:

Model cache:

```python
_MODEL_CACHE
```

prevents repeated loading.

Repeated calls:

```python
load_model()
load_model()
```

reuse model.

Raises:

```python
FileNotFoundError
```

if checkpoint missing.

Example:

```python
model=veseg.load_model()

print(type(model))
```

---

## `veseg.predict()`

Signature:

```python
predict(
    path,
    mode=ENHANCED_INVERTED,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    threshold=0.70,
    min_neighbors=2,
    image_size=256,
)
```

Parameters:

| parameter | type | default |
|-----------|------|---------|
| path | str \| Path | required |
| mode | str | ENHANCED_INVERTED |
| checkpoint_path | Path | bundled |
| threshold | float | 0.70 |
| min_neighbors | int | 2 |
| image_size | int | 256 |

Returns:

```python
PredictionResult
```

Contains:

```python
result.original
result.processed

result.probability
result.mask

result.threshold
result.mode
```

Example:

```python
result = veseg.predict(
    "example.png",
    mode=veseg.RED,
    threshold=0.65,
    min_neighbors=3,
)
```

---

# PredictionResult

Returned:

```python
result = veseg.predict(...)
```

Attributes:

```python
result.original

result.processed

result.probability

result.mask

result.threshold

result.mode
```

---

## `result.show()`

Display:

- original
- processed
- probability
- mask

Returns:

```python
None
```

Example:

```python
result.show()
```

---

## `result.save_mask(path)`

Parameters:

| parameter | type |
|-----------|------|
| path | str \| Path |

Returns:

```python
None
```

Example:

```python
result.save_mask("mask.png")
```

---

## `result.scaled(scale)`

Signature:

```python
scaled(scale)
```

Parameters:

| parameter | type |
|-----------|------|
| scale | float |

Returns:

```python
PredictionResult
```

NOT:

```python
numpy.ndarray
```

The object is copied with:

```python
mask = scale_mask(...)
```

Example:

```python
large = result.scaled(4.0)

large.show()
```

---

## `result.despeckle(min_neighbors=2)`

Parameters:

| parameter | default |
|-----------|---------|
| min_neighbors | 2 |

Returns:

```python
PredictionResult
```

Example:

```python
clean = result.despeckle(3)
```

---

# Geometry API

The geometry API converts a binary vessel mask into:

```text
mask
→ skeleton
→ distance transform
→ graph
→ radius estimates
```

Most users should use:

```python
mask, skeleton, distance, nodes, edges = veseg.extract_vessel_geometry(result.mask)
```

---

## `veseg.extract_vessel_geometry(mask)`

Run the full Rust-backed geometry pipeline.

Parameters:

| parameter | type | description |
|-----------|------|-------------|
| mask | ndarray | binary vessel mask |

Returns:

```python
(
    mask,
    skeleton,
    distance_map,
    nodes,
    edges
)
```

Output schemas:

### mask

```python
shape: (H, W)

dtype: bool
```

---

### skeleton

```python
shape: (H, W)

dtype: bool
```

---

### distance_map

```python
shape: (H, W)

dtype: float32
```

Stores:

```text
radius-like distances
```

---

### nodes

Shape:

```python
(N, 4)
```

Columns:

| index | meaning |
|------|----------|
|0|node id|
|1|row|
|2|column|
|3|kind|

---

### edges

Shape:

```python
(M, 8)
```

Columns:

| index | meaning |
|------|----------|
|0|edge id|
|1|start node|
|2|end node|
|3|length px|
|4|mean radius px|
|5|min radius px|
|6|max radius px|
|7|normalized radius|

---

Notes:

Radius estimates:

```python
edges[:, 4]
edges[:, 5]
edges[:, 6]
```

Example:

```python
mask, skeleton, distance, nodes, edges = veseg.extract_vessel_geometry(result.mask)

print(edges[:, 4])
```

---

## `veseg.skeletonize_mask(mask)`

Extract 1-pixel skeleton.

Parameters:

| parameter | type |
|-----------|------|
| mask | ndarray |

Returns:

```python
ndarray
```

dtype:

```python
bool
```

Raises:

```python
ImportError
```

if:

```text
veseg_core missing
```

Example:

```python
skeleton = veseg.skeletonize_mask(mask)
```

---

## `veseg.distance_transform_mask(mask)`

Compute the distance transform of a binary vessel mask.

Parameters:

| parameter |
|-----------|
| mask |

Returns:

```python
float32 ndarray
```

Notes:

Larger values correspond to pixels farther from vessel boundaries. These values are used as local radius estimates during geometry extraction.

Raises:

```python
ImportError
```

if the bundled Rust extension is unavailable.

Example:

```python
distance = veseg.distance_transform_mask(mask)
```

---

## `veseg.build_vessel_graph(skeleton)`

Build graph topology from a skeleton without radius summaries.

Returns:

```python
nodes,
edges
```

Edge table contains:

- edge id
- start node id
- end node id
- edge length

Radius summaries are not included. Use `build_vessel_geometry()` or `extract_vessel_geometry()` if radius estimates are needed.

Raises:

```python
ImportError
```

if the bundled Rust extension is unavailable.

Example:

```python
nodes, edges = veseg.build_vessel_graph(skeleton)
```

---

## `veseg.build_vessel_geometry(skeleton, distance_map)`

Build topology + radii.

Parameters:

| parameter |
|-----------|
| skeleton |
| distance_map |

Returns:

```python
nodes,
edges
```

Shapes:

```python
nodes.shape == (N, 4)
edges.shape == (M, 8)
```

Radius info:

```python
edges[:, 4]
edges[:, 5]
edges[:, 6]
```

Raises:

```python
ImportError
```

if the bundled Rust extension is unavailable.

Example:

```python
nodes, edges = veseg.build_vessel_geometry(skeleton, distance)
```

---

## `veseg.reconstruct_vessel_mask(skeleton, distance_map)`

Reconstruct mask.

Returns:

```python
ndarray
```

Raises:

```python
ImportError
```

if the bundled Rust extension is unavailable.

Example:

```python
mask = veseg.reconstruct_vessel_mask(skeleton, distance)
```

---

# Postprocessing API

## `veseg.threshold_probability(...)`

Signature:

```python
threshold_probability(
	probability,
	threshold=0.70
)
```

Returns:

```python
float32 mask
```

---

## `veseg.remove_specks(...)`

Signature:

```python
remove_specks(
	mask,
	min_neighbors=2
)
```

Returns:

```python
float32 mask
```

Notes:

Higher: `min_neighbors` = more aggressive cleanup

---

## `veseg.resize_mask(...)`

Signature:

```python
resize_mask(
	mask,
	size
)
```

Returns:

```python
float32 mask
```

Interpolation:

```text
nearest neighbor
```

---

## `veseg.scale_mask(...)`

Signature:

```python
scale_mask(
	mask,
	scale
)
```

Raises:

```python
ValueError
```

if:

```python
scale<=0
```

Returns:

```python
float32 mask
```

---

## `veseg.save_mask(...)`

Signature:

```python
save_mask(
	mask,
	path
)
```

Returns:

```python
None
```

---

## `veseg.postprocess_prediction(...)`

Signature:

```python
postprocess_prediction(
    probability,
    threshold=0.70,
    min_neighbors=2,
    output_size=None,
    scale=None,
)
```

Raises:

```python
ValueError
```

if:

```python
output_size
and
scale
both provided
```

Pipeline:

```text
probability
→ resize
→ threshold
→ despeckle
→ mask
```

Returns:

```python
float32 mask
```

---

# Full Pipeline Example

```python
import veseg

result = veseg.predict("test_images/structure1.png")

mask, skeleton, distance, nodes, edges = veseg.extract_vessel_geometry(result.mask)

print(nodes.shape)
print(edges.shape)
print(edges[:, 4])
```

Workflow:

```text
image
→ predict
→ mask
→ skeleton
→ graph
→ radii
→ simulation
```

---

# Training

Training entrypoint:

```bash
python scripts/train_vessmap.py
```

Training uses:

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

Segmentation:

```text
image
↓
preprocessing
↓
U-Net
↓
probability
↓
binary mask
```

Geometry:

```text
mask
↓
Rust skeletonization
↓
distance transform
↓
graph extraction
↓
radius estimation
```

---

# Internal Structure

```text
src/veseg/

predict.py
postprocessing.py
model.py
train.py

checkpoints/
vessmap_unet.pt

veseg_core/
skeletonize.rs
distance_transform.rs
graph.rs
radius.rs
reconstruction.rs
```

Public API:

```python
import veseg
```

Only exported functions in:

```python
__all__
```

are considered stable.

---
