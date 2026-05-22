

# VeSeg

VeSeg is a Python package for biomedical vascular segmentation using deep learning. The goal of VeSeg is to generate accurate vessel masks from microscopy, histology, and other vascular imaging modalities, reducing reliance on traditional thresholding and morphology-based methods.

This project is intended to serve as a reusable preprocessing tool for downstream applications including:

- Oxygen and nutrient transport simulations
- Tissue diffusion modeling
- Vascular topology extraction
- Vessel skeletonization and graph generation
- Vessel diameter estimation
- Angiogenesis modeling
- Biomedical image analysis

## Features (Planned)

- [ ] U-Net-based vessel segmentation
- [ ] Pretrained segmentation models
- [ ] Automatic image preprocessing
	- Contrast enhancement
	- Normalization
	- Denoising
- [ ] Binary vessel mask generation
- [ ] Postprocessing cleanup
- [ ] Skeletonization
- [ ] Vessel graph extraction
- [ ] Vessel width estimation
- [ ] Support for microscopy and histology images
- [ ] CLI support
- [ ] Pip-installable package

## Installation

From GitHub:

```bash
pip install git+https://github.com/HARDIntegral/VeSeg.git
```

Local development:

```bash
git clone https://github.com/HARDIntegral/VeSeg.git
cd VeSeg
pip install -e .
```

## Quick Start

```python
import veseg

mask = veseg.segment("vessel_image.png")

mask.save("mask.png")
```

## Datasets

VeSeg is being developed using publicly available vascular imaging datasets including confocal microscopy and biomedical vessel segmentation benchmarks.

## Status

⚠️ Early development. APIs, models, and package structure may change substantially.

## License

MIT License