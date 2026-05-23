"""
Postprocessing utilities for VeSeg predictions.

These functions run after model inference. They convert probability maps into
binary vessel masks, remove tiny specks, and optionally resize masks for use in
other pipelines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


MaskSize = tuple[int, int]


def threshold_probability(probability: np.ndarray, threshold: float = 0.70) -> np.ndarray:
	"""
	Convert a probability map into a binary vessel mask.

	Args:
		probability: Model probability output with values in [0, 1].
		threshold: Minimum probability required for a pixel to count as vessel.

	Returns:
		Binary float32 mask where vessel pixels are 1.0 and background pixels are 0.0.
	"""

	return (np.asarray(probability) >= threshold).astype(np.float32)


def remove_specks(mask: np.ndarray, min_neighbors: int = 2) -> np.ndarray:
	"""
	Remove isolated foreground pixels from a binary vessel mask.

	This is a small dependency-free cleanup step. A vessel pixel is kept only if it
	has enough neighboring vessel pixels in its 3x3 neighborhood.

	Args:
		mask: Binary vessel mask.
		min_neighbors: Minimum neighboring vessel pixels required to keep a pixel.

	Returns:
		Cleaned binary float32 mask.
	"""

	mask = np.asarray(mask).astype(bool)

	if min_neighbors <= 0:
		return mask.astype(np.float32)

	padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
	neighbor_count = np.zeros(mask.shape, dtype=np.uint8)

	for y_offset in range(3):
		for x_offset in range(3):
			if y_offset == 1 and x_offset == 1:
				continue

			neighbor_count += padded[
				y_offset:y_offset + mask.shape[0],
				x_offset:x_offset + mask.shape[1],
			]

	cleaned = mask & (neighbor_count >= min_neighbors)
	return cleaned.astype(np.float32)


def clean_binary_mask(mask: np.ndarray, min_neighbors: int = 2) -> np.ndarray:
	"""
	Clean a binary mask after prediction.

	This is kept as a readable alias around remove_specks because most cleanup
	currently means removing isolated false-positive pixels.
	"""

	return remove_specks(mask, min_neighbors=min_neighbors)


def resize_mask(mask: np.ndarray, size: MaskSize) -> np.ndarray:
	"""
	Resize a binary mask to a new size.

	Nearest-neighbor resizing is used so the mask stays binary instead of creating
	gray interpolation values around vessel edges.

	Args:
		mask: Binary vessel mask.
		size: Output size as (width, height), matching Pillow's convention.

	Returns:
		Resized binary float32 mask.
	"""

	mask_image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
	resized = mask_image.resize(size, resample=Image.Resampling.NEAREST)

	return (np.asarray(resized) > 0).astype(np.float32)


def scale_mask(mask: np.ndarray, scale: float) -> np.ndarray:
	"""
	Scale a binary mask up or down by a multiplier.

	Args:
		mask: Binary vessel mask.
		scale: Scale multiplier. For example, 2.0 doubles width and height.

	Returns:
		Scaled binary float32 mask.
	"""

	if scale <= 0:
		raise ValueError("scale must be greater than 0")

	height, width = np.asarray(mask).shape[-2:]
	new_width = max(1, int(round(width * scale)))
	new_height = max(1, int(round(height * scale)))

	return resize_mask(mask, size=(new_width, new_height))


def save_mask(mask: np.ndarray, path: str | Path) -> None:
	"""
	Save a binary mask as a PNG-compatible image.

	Args:
		mask: Binary vessel mask.
		path: Output image path.
	"""

	output = (np.asarray(mask) > 0).astype(np.uint8) * 255
	Image.fromarray(output).save(path)


def postprocess_prediction(
	probability: np.ndarray,
	threshold: float = 0.70,
	min_neighbors: int = 2,
	output_size: MaskSize | None = None,
	scale: float | None = None,
) -> np.ndarray:
	"""
	Convert a probability map into a cleaned binary vessel mask.

	Args:
		probability: Model probability output with values in [0, 1].
		threshold: Minimum probability required for a pixel to count as vessel.
		min_neighbors: Minimum neighborhood support needed to keep a vessel pixel.
		output_size: Optional output size as (width, height).
		scale: Optional scale multiplier for the final mask.

	Returns:
		Cleaned binary float32 vessel mask.
	"""

	if output_size is not None and scale is not None:
		raise ValueError("Use either output_size or scale, not both")

	mask = threshold_probability(probability, threshold=threshold)
	mask = remove_specks(mask, min_neighbors=min_neighbors)

	if output_size is not None:
		return resize_mask(mask, size=output_size)

	if scale is not None:
		return scale_mask(mask, scale=scale)

	return mask