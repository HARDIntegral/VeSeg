"""
Postprocessing utilities for VeSeg predictions.

These functions run after model inference. They convert probability maps into
binary vessel masks, remove tiny specks, resize masks, and optionally call the
Rust geometry backend for skeletons, distance maps, reconstructions, and vessel
graph extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
	from . import veseg_core
except ImportError:
	veseg_core = None


MaskSize = tuple[int, int]
VesselGeometry = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _require_core():
	"""Return the Rust backend or raise a clear install error."""
	if veseg_core is None:
		raise ImportError(
			"veseg_core is required for geometry extraction. "
			"Run `maturin develop` from the veseg_core directory."
		)

	return veseg_core


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
	"""Convert any binary-like mask into a C-contiguous boolean mask."""
	return np.ascontiguousarray(np.asarray(mask) > 0)


def _as_float_map(values: np.ndarray) -> np.ndarray:
	"""Convert a numeric map into a C-contiguous float32 array for Rust."""
	return np.ascontiguousarray(np.asarray(values, dtype=np.float32))


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

	# Count 8-connected support around each foreground pixel.
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


def _resize_probability(probability: np.ndarray, size: MaskSize) -> np.ndarray:
	"""
	Resize a soft probability map to a new size before thresholding.

	This is an internal helper used by postprocess_prediction. Probability maps are
	continuous model outputs, so bilinear resizing avoids the blocky artifacts that
	come from upscaling an already-thresholded binary mask.

	Args:
		probability: Model probability output with values in [0, 1].
		size: Output size as (width, height), matching Pillow's convention.

	Returns:
		Resized float32 probability map clipped to [0, 1].
	"""

	probability = np.asarray(probability, dtype=np.float32)
	probability_image = Image.fromarray(probability)
	resized = probability_image.resize(size, resample=Image.Resampling.BILINEAR)

	return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 1.0)


def scale_mask(mask: np.ndarray, scale: float) -> np.ndarray:
	"""
	Scale an existing binary vessel mask up or down by a multiplier.

	This function preserves binary values using nearest-neighbor interpolation.
	Because resizing happens after thresholding, large upscaling factors may create
	blocky or stair-step vessel edges.

	For smoother prediction-time scaling, use postprocess_prediction(..., scale=...)
	or postprocess_prediction(..., output_size=...). Those paths resize the model's
	soft output before thresholding and generally produce smoother vessel geometry.

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


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
	"""
	Extract a 1-pixel-wide vessel skeleton using the Rust backend.

	Args:
		mask: Binary vessel mask.

	Returns:
		Boolean skeleton mask.
	"""

	core = _require_core()
	return core.skeletonize_mask(_as_bool_mask(mask))


def distance_transform_mask(mask: np.ndarray) -> np.ndarray:
	"""
	Estimate local vessel radius at each mask pixel using the Rust backend.

	Args:
		mask: Binary vessel mask.

	Returns:
		float32 distance transform where vessel pixels store radius-like distances.
	"""

	core = _require_core()
	return core.distance_transform_mask(_as_bool_mask(mask))


def reconstruct_vessel_mask(skeleton: np.ndarray, distance_map: np.ndarray) -> np.ndarray:
	"""
	Reconstruct a smoother binary vessel mask from skeleton and radius map.

	Args:
		skeleton: Boolean skeleton mask.
		distance_map: Distance transform from distance_transform_mask.

	Returns:
		Boolean reconstructed vessel mask.
	"""

	core = _require_core()
	return core.reconstruct_vessel_mask(
		_as_bool_mask(skeleton),
		_as_float_map(distance_map),
	)


def build_vessel_graph(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""
	Build graph nodes and edges from a skeleton using the Rust backend.

	Args:
		skeleton: Boolean skeleton mask.

	Returns:
		(nodes, edges), where nodes stores id/row/col/kind and edges stores
		id/start/end/length.
	"""

	core = _require_core()
	return core.build_vessel_graph_mask(_as_bool_mask(skeleton))


def build_vessel_geometry(
	skeleton: np.ndarray,
	distance_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Build graph nodes and radius-annotated edges using the Rust backend.

	Args:
		skeleton: Boolean skeleton mask.
		distance_map: Distance transform from distance_transform_mask.

	Returns:
		(nodes, edges), where edge columns are:
		id, start_node, end_node, length_px, mean_radius_px,
		min_radius_px, max_radius_px, mean_radius_norm.
	"""

	core = _require_core()
	return core.build_vessel_geometry_mask(
		_as_bool_mask(skeleton),
		_as_float_map(distance_map),
	)


def extract_vessel_geometry(mask: np.ndarray) -> VesselGeometry:
	"""
	Run the full Rust-backed geometry pipeline from a binary vessel mask.

	Args:
		mask: Binary vessel mask.

	Returns:
		(mask, skeleton, distance_map, nodes, edges). The returned mask is boolean,
		and edges include radius summaries for downstream transport solvers.
	"""

	mask = _as_bool_mask(mask)
	skeleton = skeletonize_mask(mask)
	distance_map = distance_transform_mask(mask)
	nodes, edges = build_vessel_geometry(skeleton, distance_map)

	return mask, skeleton, distance_map, nodes, edges


def postprocess_prediction(
	probability: np.ndarray,
	threshold: float = 0.70,
	min_neighbors: int = 2,
	output_size: MaskSize | None = None,
	scale: float | None = None,
) -> np.ndarray:
	"""
	Convert a probability map into a cleaned binary vessel mask.

	If resizing is requested, the soft probability map is resized before
	thresholding. This avoids blocky stair-step artifacts that happen when a small
	binary mask is upscaled directly.

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

	probability = np.asarray(probability, dtype=np.float32)

	if scale is not None:
		if scale <= 0:
			raise ValueError("scale must be greater than 0")

		height, width = probability.shape[-2:]
		output_size = (
			max(1, int(round(width * scale))),
			max(1, int(round(height * scale))),
		)

	if output_size is not None:
		probability = _resize_probability(probability, size=output_size)

	mask = threshold_probability(probability, threshold=threshold)
	mask = remove_specks(mask, min_neighbors=min_neighbors)

	return mask
