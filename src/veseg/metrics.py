"""
Segmentation metrics for VeSeg.

This module is mainly used internally during training and testing.
"""

from __future__ import annotations

import numpy as np


def as_binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
	mask = np.asarray(mask)
	return mask >= threshold


def prepare_binary_masks(
	prediction: np.ndarray,
	target: np.ndarray,
	threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
	pred_mask = as_binary_mask(prediction, threshold)
	target_mask = as_binary_mask(target, threshold)

	if pred_mask.shape != target_mask.shape:
		raise ValueError(
			f"prediction and target must have the same shape, got "
			f"{pred_mask.shape} and {target_mask.shape}"
		)

	return pred_mask, target_mask


def dice_score(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
	pred_mask, target_mask = prepare_binary_masks(prediction, target, threshold)

	intersection = np.logical_and(pred_mask, target_mask).sum()
	denominator = pred_mask.sum() + target_mask.sum()

	# If both masks are empty, they agree perfectly.
	if denominator == 0:
		return 1.0

	return float((2.0 * intersection) / denominator)


def iou_score(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
	pred_mask, target_mask = prepare_binary_masks(prediction, target, threshold)

	intersection = np.logical_and(pred_mask, target_mask).sum()
	union = np.logical_or(pred_mask, target_mask).sum()

	# If both masks are empty, there is no disagreement.
	if union == 0:
		return 1.0

	return float(intersection / union)


def pixel_accuracy(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
	pred_mask, target_mask = prepare_binary_masks(prediction, target, threshold)

	# Pixel accuracy is usually inflated for vessel masks because most pixels are background,
	# but it is still useful as a quick sanity check.
	return float(np.mean(pred_mask == target_mask))
