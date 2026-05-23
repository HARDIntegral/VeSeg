"""
High-level inference API for VeSeg.

This module is meant to be the package-facing prediction layer. Scripts should
stay small and call these functions instead of duplicating model loading,
preprocessing, inference, and postprocessing logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from importlib.resources.abc import Traversable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter

from veseg.model import UNet
from veseg.postprocessing import (
	postprocess_prediction,
	scale_mask,
	save_mask,
	remove_specks,
)
from veseg.train import get_device


GRAYSCALE = "grayscale"
INVERTED = "inverted"
ENHANCED_INVERTED = "enhanced_inverted"
RED = "red_vessels"

DEFAULT_IMAGE_SIZE = 256
DEFAULT_THRESHOLD = 0.70
DEFAULT_CHECKPOINT_PATH = files("veseg").joinpath("checkpoints/vessmap_unet.pt")



_MODEL_CACHE: dict[tuple[str, str], UNet] = {}


def resolve_checkpoint_path(checkpoint_path: str | Path | Traversable) -> Path | Traversable:
	"""
	Resolve the model checkpoint path.

	The default checkpoint is packaged inside veseg/checkpoints. A user can still
	pass an explicit local path to override the bundled model.
	"""

	if isinstance(checkpoint_path, Path):
		return checkpoint_path

	if isinstance(checkpoint_path, str):
		return Path(checkpoint_path)

	return checkpoint_path

@dataclass
class PredictionResult:
	"""
	Container for VeSeg prediction outputs.

	Attributes:
		original: RGB image used for display.
		processed: Preprocessed grayscale image passed into the model.
		probability: Model probability map with values in [0, 1].
		mask: Cleaned binary vessel mask.
		threshold: Threshold used to convert probability into mask.
		mode: Preprocessing mode used before prediction.
	"""

	original: np.ndarray
	processed: np.ndarray
	probability: np.ndarray
	mask: np.ndarray
	threshold: float
	mode: str

	def show(self) -> None:
		"""
		Display original image, processed image, probability map, and binary mask.
		"""

		plt.figure(figsize=(16, 4))

		plt.subplot(1, 4, 1)
		plt.imshow(self.original)
		plt.title("Original")
		plt.axis("off")

		plt.subplot(1, 4, 2)
		plt.imshow(self.processed, cmap="gray")
		plt.title(f"Processed: {self.mode}")
		plt.axis("off")

		plt.subplot(1, 4, 3)
		plt.imshow(self.probability, cmap="gray")
		plt.title("Probability")
		plt.axis("off")

		plt.subplot(1, 4, 4)
		plt.imshow(self.mask, cmap="gray")
		plt.title(f"Mask\nthreshold={self.threshold}")
		plt.axis("off")

		plt.tight_layout()
		plt.show()

	def save_mask(self, path: str | Path) -> None:
		"""
		Save the current mask.
		"""

		save_mask(self.mask, path)

	def scaled(self, scale: float) -> "PredictionResult":
		"""
		Return a new PredictionResult with a scaled mask.
		Useful when exporting vessel networks into simulations with different resolutions.
		"""

		return PredictionResult(
			original=self.original,
			processed=self.processed,
			probability=self.probability,
			mask=scale_mask(self.mask, scale),
			threshold=self.threshold,
			mode=self.mode,
		)

	def despeckle(self, min_neighbors: int = 2) -> "PredictionResult":
		"""
		Remove isolated vessel specks from the mask.
		"""

		return PredictionResult(
			original=self.original,
			processed=self.processed,
			probability=self.probability,
			mask=remove_specks(self.mask, min_neighbors),
			threshold=self.threshold,
			mode=self.mode,
		)

	def skeletonize(self) -> None:
		"""
		Placeholder for future skeletonization support.
		"""

		raise NotImplementedError("Skeletonization is not implemented yet.")


def normalize(array: np.ndarray) -> np.ndarray:
	"""
	Normalize an array to [0, 1].
	"""

	array = array.astype(np.float32)
	minimum = float(array.min())
	maximum = float(array.max())

	if maximum - minimum < 1e-8:
		return np.zeros_like(array, dtype=np.float32)

	return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def contrast_window(
	array: np.ndarray,
	low_percentile: float = 75,
	high_percentile: float = 99,
) -> np.ndarray:
	"""
	Window contrast using percentile clipping.
	"""

	low = float(np.percentile(array, low_percentile))
	high = float(np.percentile(array, high_percentile))

	if high - low < 1e-8:
		return np.zeros_like(array, dtype=np.float32)

	array = (array - low) / (high - low)
	return np.clip(array, 0.0, 1.0).astype(np.float32)


def preprocess_image(
	path: str | Path,
	mode: str = ENHANCED_INVERTED,
	image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Load and preprocess an image for VeSeg inference.

	Args:
		path: Image path.
		mode: Preprocessing mode.
		image_size: Square image size expected by the model.

	Returns:
		processed: Grayscale model input with shape (height, width).
		original_display: RGB image for visualization.
	"""

	original = Image.open(path).convert("RGB")
	original = original.resize((image_size, image_size))

	if mode == GRAYSCALE:
		processed = original.convert("L")
		image = np.array(processed).astype(np.float32) / 255.0

	elif mode == INVERTED:
		processed = original.convert("L")
		image = np.array(processed).astype(np.float32) / 255.0
		image = 1.0 - image

	elif mode == ENHANCED_INVERTED:
		processed = original.convert("L")
		background_blur = processed.filter(ImageFilter.GaussianBlur(radius=10))
		local_blur = processed.filter(ImageFilter.GaussianBlur(radius=2))

		gray = np.array(processed).astype(np.float32) / 255.0
		background = np.array(background_blur).astype(np.float32) / 255.0
		local = np.array(local_blur).astype(np.float32) / 255.0

		sharpened = np.clip(gray + 1.0 * (gray - local), 0.0, 1.0)
		image = background - sharpened
		image = contrast_window(image, low_percentile=75, high_percentile=99)
		image = np.power(image, 2.0).astype(np.float32)

	elif mode == RED:
		rgb = np.array(original).astype(np.float32) / 255.0
		red = rgb[:, :, 0]
		green = rgb[:, :, 1]
		blue = rgb[:, :, 2]

		image = red - 0.5 * (green + blue)
		image = normalize(image)

	else:
		raise ValueError(f"Unknown preprocessing mode: {mode}")

	original_display = np.array(original).astype(np.float32) / 255.0
	return image.astype(np.float32), original_display.astype(np.float32)


def load_model(
	checkpoint_path: str | Path | Traversable = DEFAULT_CHECKPOINT_PATH,
	device: torch.device | None = None,
) -> UNet:
	"""
	Load a trained VeSeg U-Net checkpoint.
	"""

	device = device or get_device()
	checkpoint_path = resolve_checkpoint_path(checkpoint_path)
	cache_key = (str(checkpoint_path), str(device))

	if not checkpoint_path.exists():
		raise FileNotFoundError(
			f"Could not find checkpoint: {checkpoint_path}. "
			"If you installed VeSeg from GitHub, make sure "
			"src/veseg/checkpoints/vessmap_unet.pt is included in the package."
		)

	if cache_key in _MODEL_CACHE:
		return _MODEL_CACHE[cache_key]

	model = UNet(in_channels=1, out_channels=1).to(device)
	model.load_state_dict(torch.load(checkpoint_path, map_location=device))
	model.eval()

	_MODEL_CACHE[cache_key] = model
	return model


def predict(
	path: str | Path,
	mode: str = ENHANCED_INVERTED,
	checkpoint_path: str | Path | Traversable = DEFAULT_CHECKPOINT_PATH,
	threshold: float = DEFAULT_THRESHOLD,
	min_neighbors: int = 2,
	image_size: int = DEFAULT_IMAGE_SIZE,
) -> PredictionResult:
	"""
	Predict a vessel mask from an image.

	Args:
		path: Image path.
		mode: Preprocessing mode. Use RED for visible-light red vessel images.
		checkpoint_path: Path to trained model weights.
		threshold: Probability threshold used to create the binary mask.
		min_neighbors: Minimum neighborhood support for postprocessing cleanup.
		image_size: Square image size expected by the model.

	Returns:
		PredictionResult containing images, probability map, mask, scaling utilities,
		and future vessel graph/skeleton operations.
	"""

	device = get_device()
	model = load_model(checkpoint_path=checkpoint_path, device=device)
	processed, original_display = preprocess_image(path, mode=mode, image_size=image_size)

	input_tensor = (
		torch.from_numpy(processed)
		.unsqueeze(0)
		.unsqueeze(0)
		.float()
		.to(device)
	)

	with torch.no_grad():
		logits = model(input_tensor)
		probability = torch.sigmoid(logits).cpu().numpy()[0, 0]

	mask = postprocess_prediction(
		probability=probability,
		threshold=threshold,
		min_neighbors=min_neighbors,
	)

	return PredictionResult(
		original=original_display,
		processed=processed,
		probability=probability,
		mask=mask,
		threshold=threshold,
		mode=mode,
	)
