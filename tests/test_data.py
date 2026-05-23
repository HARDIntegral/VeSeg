from pathlib import Path

import numpy as np

from veseg.data import VessMAPDataset, load_binary_mask, load_grayscale_image


DATASET_ROOT = Path("data/VessMAP")


def test_vessmap_dataset_loads_all_samples() -> None:
	dataset = VessMAPDataset(DATASET_ROOT)
	assert len(dataset) == 100


def test_vessmap_sample_shapes_are_correct() -> None:
	dataset = VessMAPDataset(DATASET_ROOT)
	image, mask, skeleton = dataset[0]

	assert image.shape == (1, 256, 256)
	assert mask.shape == (1, 256, 256)
	assert skeleton is not None
	assert skeleton.shape == (1, 256, 256)


def test_vessmap_default_split_lengths() -> None:
	assert len(VessMAPDataset(DATASET_ROOT, split="train")) == 70
	assert len(VessMAPDataset(DATASET_ROOT, split="val")) == 15
	assert len(VessMAPDataset(DATASET_ROOT, split="test")) == 15


def test_vessmap_splits_do_not_overlap() -> None:
	train = VessMAPDataset(DATASET_ROOT, split="train")
	val = VessMAPDataset(DATASET_ROOT, split="val")
	test = VessMAPDataset(DATASET_ROOT, split="test")

	train_names = {sample.image_path.name for sample in train.samples}
	val_names = {sample.image_path.name for sample in val.samples}
	test_names = {sample.image_path.name for sample in test.samples}

	assert train_names.isdisjoint(val_names)
	assert train_names.isdisjoint(test_names)
	assert val_names.isdisjoint(test_names)


def test_vessmap_masks_are_binary() -> None:
	dataset = VessMAPDataset(DATASET_ROOT)
	_, mask, skeleton = dataset[0]

	assert set(np.unique(mask)).issubset({0.0, 1.0})
	assert skeleton is not None
	assert set(np.unique(skeleton)).issubset({0.0, 1.0})


def test_load_grayscale_image_returns_2d_array() -> None:
	dataset = VessMAPDataset(DATASET_ROOT)
	image = load_grayscale_image(dataset.samples[0].image_path)

	assert image.shape == (256, 256)
	assert image.dtype == np.float32


def test_load_binary_mask_returns_bool_array() -> None:
	dataset = VessMAPDataset(DATASET_ROOT)
	mask = load_binary_mask(dataset.samples[0].mask_path)

	assert mask.shape == (256, 256)
	assert mask.dtype == bool