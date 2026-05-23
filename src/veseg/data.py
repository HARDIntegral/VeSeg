

"""
Dataset loading utilities for VeSeg.

This module currently focuses on the VessMAP dataset. VessMAP gives VeSeg a
simple supervised segmentation layout:

	image -> vessel mask -> vessel skeleton

The loader stays intentionally lightweight. It only depends on NumPy and Pillow
for image loading, while PyTorch conversion happens later in the training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterator, Literal

import numpy as np
from PIL import Image


SplitName = Literal["all", "train", "val", "test"]
IMAGE_EXTENSIONS = (".png", ".tif", ".tiff")


@dataclass(frozen=True)
class VesselSample:
	"""
	File paths for one vessel segmentation sample.

	Attributes:
		image_path: Path to the raw microscopy image.
		mask_path: Path to the binary vessel mask.
		skeleton_path: Optional path to the binary vessel skeleton.
	"""

	image_path: Path
	mask_path: Path
	skeleton_path: Path | None = None


def load_grayscale_image(path: str | Path, normalize: bool = True) -> np.ndarray:
	"""
	Load an image as a grayscale NumPy array.

	Args:
		path: Image file path.
		normalize: If True, return float32 values in [0, 1].

	Returns:
		A 2D grayscale array with shape (height, width).
	"""

	array = np.asarray(Image.open(path).convert("L"))

	if normalize:
		return array.astype(np.float32) / 255.0

	return array


def load_binary_mask(path: str | Path) -> np.ndarray:
	"""
	Load a binary vessel mask.

	Any nonzero pixel is treated as vessel.

	Args:
		path: Mask image file path.

	Returns:
		A boolean mask with shape (height, width).
	"""

	return load_grayscale_image(path, normalize=False) > 0


def add_channel_axis(array: np.ndarray) -> np.ndarray:
	"""
	Add the channel dimension expected by the model.

	Args:
		array: 2D image or mask with shape (height, width).

	Returns:
		Array with shape (1, height, width).
	"""

	return array[np.newaxis, :, :]


class VessMAPDataset:
	"""
	Lightweight loader for the VessMAP vascular microscopy dataset.

	Expected folder structure:

		data/VessMAP/
		├── images/
		├── annotator1/
		│   ├── labels/
		│   └── skeletons/
		└── annotator2/
		    └── labels/

	By default, annotator1 is used because it contains the full label set and the
	skeletons. Annotator2 can still be loaded for inter-annotator comparisons.
	"""

	def __init__(
		self,
		root: str | Path,
		annotator: str = "annotator1",
		split: SplitName = "all",
		train_ratio: float = 0.7,
		val_ratio: float = 0.15,
		seed: int = 42,
	) -> None:
		"""
		Create a VessMAP dataset split.

		Args:
			root: Path to the VessMAP dataset directory.
			annotator: Annotator folder to load labels from.
			split: Dataset split to expose.
			train_ratio: Fraction of samples used for training.
			val_ratio: Fraction of samples used for validation.
			seed: Seed used for deterministic splitting.
		"""

		self.root = Path(root)
		self.annotator = annotator
		self.split = split
		self.train_ratio = train_ratio
		self.val_ratio = val_ratio
		self.seed = seed

		self.images_dir = self.root / "images"
		self.labels_dir = self.root / annotator / "labels"
		self.skeletons_dir = self.root / annotator / "skeletons"

		self._validate_directories()
		self._validate_split_settings()

		self.samples = self._select_split(self._collect_samples())

	def __len__(self) -> int:
		"""
		Return the number of samples in the selected split.
		"""

		return len(self.samples)

	def __iter__(self) -> Iterator[VesselSample]:
		"""
		Iterate over sample file paths without loading image arrays.
		"""

		return iter(self.samples)

	def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
		"""
		Load one sample as model-ready arrays.

		Args:
			index: Sample index in the selected split.

		Returns:
			image: Float32 grayscale image with shape (1, height, width).
			mask: Float32 binary mask with shape (1, height, width).
			skeleton: Optional float32 binary skeleton with shape (1, height, width).
		"""

		sample = self.samples[index]

		image = load_grayscale_image(sample.image_path, normalize=True).astype(np.float32)
		mask = load_binary_mask(sample.mask_path).astype(np.float32)
		skeleton = self._load_optional_skeleton(sample.skeleton_path)

		return add_channel_axis(image), add_channel_axis(mask), skeleton

	def _validate_directories(self) -> None:
		"""
		Validate that the required VessMAP folders exist.
		"""

		if not self.root.exists():
			raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

		if not self.images_dir.exists():
			raise FileNotFoundError(f"Missing VessMAP images directory: {self.images_dir}")

		if not self.labels_dir.exists():
			raise FileNotFoundError(f"Missing VessMAP labels directory: {self.labels_dir}")

	def _validate_split_settings(self) -> None:
		"""
		Validate split name and split ratios.
		"""

		if self.split not in {"all", "train", "val", "test"}:
			raise ValueError("split must be one of: 'all', 'train', 'val', or 'test'")

		if not 0.0 < self.train_ratio < 1.0:
			raise ValueError("train_ratio must be between 0 and 1")

		if not 0.0 <= self.val_ratio < 1.0:
			raise ValueError("val_ratio must be between 0 and 1")

		if self.train_ratio + self.val_ratio >= 1.0:
			raise ValueError("train_ratio + val_ratio must be less than 1")

	def _collect_samples(self) -> list[VesselSample]:
		"""
		Collect image, mask, and optional skeleton paths.
		"""

		image_paths = self._find_image_paths()
		samples: list[VesselSample] = []

		for image_path in image_paths:
			stem = image_path.stem
			mask_path = self._find_matching_file(self.labels_dir, stem)

			if mask_path is None:
				raise FileNotFoundError(f"Could not find mask for image: {image_path.name}")

			samples.append(
				VesselSample(
					image_path=image_path,
					mask_path=mask_path,
					skeleton_path=self._find_skeleton_path(stem),
				)
			)

		return samples

	def _find_image_paths(self) -> list[Path]:
		"""
		Find all image files in the VessMAP image directory.
		"""

		image_paths: list[Path] = []

		for extension in IMAGE_EXTENSIONS:
			image_paths.extend(self.images_dir.glob(f"*{extension}"))

		image_paths = sorted(set(image_paths))

		if not image_paths:
			raise FileNotFoundError(f"No supported images found in: {self.images_dir}")

		return image_paths

	def _find_skeleton_path(self, stem: str) -> Path | None:
		"""
		Find a matching skeleton file if skeletons are available.
		"""

		if not self.skeletons_dir.exists():
			return None

		return self._find_matching_file(self.skeletons_dir, stem)

	def _select_split(self, samples: list[VesselSample]) -> list[VesselSample]:
		"""
		Select the requested deterministic dataset split.
		"""

		if self.split == "all":
			return samples

		shuffled_samples = samples.copy()
		Random(self.seed).shuffle(shuffled_samples)

		train_end, val_end = self._split_bounds(len(shuffled_samples))

		if self.split == "train":
			return shuffled_samples[:train_end]

		if self.split == "val":
			return shuffled_samples[train_end:val_end]

		return shuffled_samples[val_end:]

	def _split_bounds(self, sample_count: int) -> tuple[int, int]:
		"""
		Compute train and validation split boundaries.
		"""

		train_end = int(sample_count * self.train_ratio)
		val_end = train_end + int(sample_count * self.val_ratio)
		return train_end, val_end

	@staticmethod
	def _load_optional_skeleton(skeleton_path: Path | None) -> np.ndarray | None:
		"""
		Load a skeleton file if one exists.
		"""

		if skeleton_path is None:
			return None

		skeleton = load_binary_mask(skeleton_path).astype(np.float32)
		return add_channel_axis(skeleton)

	@staticmethod
	def _find_matching_file(directory: Path, stem: str) -> Path | None:
		"""
		Find a file with the requested stem and any supported image extension.
		"""

		for extension in IMAGE_EXTENSIONS:
			candidate = directory / f"{stem}{extension}"
			if candidate.exists():
				return candidate

		matches = sorted(directory.glob(f"{stem}.*"))
		if matches:
			return matches[0]

		return None