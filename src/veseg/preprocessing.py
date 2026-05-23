"""
Preprocessing and augmentation utilities for VeSeg.

These functions operate on NumPy arrays before conversion to PyTorch tensors.
Augmentations are applied consistently to images and masks so labels stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AugmentationConfig:
	"""
	Configuration for random augmentation.
	"""

	horizontal_flip_probability: float = 0.5
	vertical_flip_probability: float = 0.5
	rotate_probability: float = 0.5
	brightness_probability: float = 0.3
	contrast_probability: float = 0.3
	noise_probability: float = 0.2
	invert_probability: float = 0.03

	brightness_range: tuple[float, float] = (0.8, 1.2)
	contrast_range: tuple[float, float] = (0.8, 1.2)
	noise_std_range: tuple[float, float] = (0.0, 0.04)


def normalize_image(image: np.ndarray) -> np.ndarray:
	"""
	Normalize image intensities into [0, 1].
	"""

	image = np.asarray(image).astype(np.float32)

	if image.max() > 1.0:
		image /= 255.0

	return np.clip(image, 0.0, 1.0)


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
	"""
	Convert any mask-like array into {0,1}.
	"""

	return (np.asarray(mask) > 0).astype(np.float32)


def random_flip_pair(
	image: np.ndarray,
	mask: np.ndarray,
	rng: np.random.Generator,
	horizontal_probability: float,
	vertical_probability: float,
) -> tuple[np.ndarray, np.ndarray]:

	if rng.random() < horizontal_probability:
		image = np.flip(image, axis=-1)
		mask = np.flip(mask, axis=-1)

	if rng.random() < vertical_probability:
		image = np.flip(image, axis=-2)
		mask = np.flip(mask, axis=-2)

	return image.copy(), mask.copy()


def random_rotate_pair(
	image: np.ndarray,
	mask: np.ndarray,
	rng: np.random.Generator,
	probability: float,
) -> tuple[np.ndarray, np.ndarray]:

	if rng.random() >= probability:
		return image, mask

	k = int(rng.integers(0, 4))
	return (
		np.rot90(image, k=k, axes=(-2, -1)).copy(),
		np.rot90(mask, k=k, axes=(-2, -1)).copy(),
	)


def random_brightness(image, rng, probability, factor_range):
	if rng.random() >= probability:
		return image

	factor = rng.uniform(*factor_range)
	return np.clip(image * factor, 0.0, 1.0).astype(np.float32)


def random_contrast(image, rng, probability, factor_range):
	if rng.random() >= probability:
		return image

	mean = float(image.mean())
	factor = rng.uniform(*factor_range)

	# Stretch intensities around the image mean.
	return np.clip((image - mean) * factor + mean, 0.0, 1.0).astype(np.float32)


def random_gaussian_noise(image, rng, probability, std_range):
	if rng.random() >= probability:
		return image

	std = rng.uniform(*std_range)
	noise = rng.normal(0.0, std, image.shape)
	return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


def random_invert_contrast(image, rng, probability):
	if rng.random() >= probability:
		return image

	# Helps prevent overfitting to bright-vessel vs dark-vessel datasets.
	return (1.0 - image).astype(np.float32)


def augment_pair(
	image: np.ndarray,
	mask: np.ndarray,
	config: AugmentationConfig | None = None,
	seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Apply spatial + intensity augmentation.
	"""

	config = config or AugmentationConfig()
	rng = np.random.default_rng(seed)

	image = normalize_image(image)
	mask = ensure_binary_mask(mask)

	image, mask = random_flip_pair(
		image,
		mask,
		rng,
		config.horizontal_flip_probability,
		config.vertical_flip_probability,
	)

	image, mask = random_rotate_pair(
		image,
		mask,
		rng,
		config.rotate_probability,
	)

	image = random_brightness(image, rng, config.brightness_probability, config.brightness_range)
	image = random_contrast(image, rng, config.contrast_probability, config.contrast_range)
	image = random_gaussian_noise(image, rng, config.noise_probability, config.noise_std_range)
	image = random_invert_contrast(image, rng, config.invert_probability)

	return image.astype(np.float32), mask.astype(np.float32)
