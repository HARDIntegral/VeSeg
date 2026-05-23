import numpy as np

from veseg.metrics import (
	as_binary_mask,
	dice_score,
	iou_score,
	pixel_accuracy,
)


def test_as_binary_mask_thresholds_values() -> None:
	mask = np.array([[0.1, 0.5], [0.7, 0.0]])
	binary = as_binary_mask(mask)

	expected = np.array([
		[False, True],
		[True, False]
	])

	assert binary.dtype == bool
	assert np.array_equal(binary, expected)


def test_dice_score_perfect_overlap() -> None:
	mask = np.array([
		[1, 0],
		[1, 1]
	])

	assert dice_score(mask, mask) == 1.0


def test_dice_score_no_overlap() -> None:
	prediction = np.array([
		[1, 0],
		[0, 0]
	])

	target = np.array([
		[0, 0],
		[0, 1]
	])

	assert dice_score(prediction, target) == 0.0


def test_dice_score_partial_overlap() -> None:
	prediction = np.array([
		[1, 1],
		[0, 0]
	])

	target = np.array([
		[1, 0],
		[1, 0]
	])

	assert dice_score(prediction, target) == 0.5


def test_dice_score_empty_masks() -> None:
	prediction = np.zeros((2, 2))
	target = np.zeros((2, 2))

	assert dice_score(prediction, target) == 1.0


def test_iou_score_perfect_overlap() -> None:
	mask = np.array([
		[1, 0],
		[1, 1]
	])

	assert iou_score(mask, mask) == 1.0


def test_iou_score_partial_overlap() -> None:
	prediction = np.array([
		[1, 1],
		[0, 0]
	])

	target = np.array([
		[1, 0],
		[1, 0]
	])

	assert iou_score(prediction, target) == (1.0 / 3.0)


def test_iou_score_empty_masks() -> None:
	prediction = np.zeros((2, 2))
	target = np.zeros((2, 2))

	assert iou_score(prediction, target) == 1.0


def test_pixel_accuracy_perfect_match() -> None:
	mask = np.array([
		[1, 0],
		[1, 1]
	])

	assert pixel_accuracy(mask, mask) == 1.0


def test_pixel_accuracy_partial_match() -> None:
	prediction = np.array([
		[1, 1],
		[0, 0]
	])

	target = np.array([
		[1, 0],
		[1, 0]
	])

	assert pixel_accuracy(prediction, target) == 0.5


def test_pixel_accuracy_shape_mismatch_raises() -> None:
	prediction = np.zeros((2, 2))
	target = np.zeros((3, 3))

	try:
		pixel_accuracy(prediction, target)
	except ValueError:
		return

	raise AssertionError(
		"pixel_accuracy should raise ValueError for shape mismatch"
	)