"""
Reusable training utilities for VeSeg.

This module keeps the training logic inside the package so scripts can stay
small. It supports both the baseline model and the U-Net as long as the model
accepts tensors shaped like:

	(batch, channels, height, width)

and returns one logit map per image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn

from veseg.data import VessMAPDataset
from veseg.metrics import dice_score, iou_score, pixel_accuracy
from veseg.preprocessing import AugmentationConfig, augment_pair


@dataclass(frozen=True)
class TrainingConfig:
	"""
	Configuration for a segmentation training run.

	Attributes:
		dataset_root: Path to the VessMAP dataset root.
		checkpoint_path: Path where the trained model weights should be saved.
		batch_size: Number of samples per gradient update.
		epochs: Number of times to iterate over the training split.
		learning_rate: AdamW optimizer learning rate.
		positive_weight: Weight applied to vessel pixels in BCEWithLogitsLoss.
		use_augmentation: Whether to apply augmentation during training.
		augmentation_config: Optional augmentation configuration.
		thresholds: Probability thresholds tested during validation.
	"""

	dataset_root: Path
	checkpoint_path: Path
	batch_size: int = 8
	epochs: int = 50
	learning_rate: float = 1e-3
	positive_weight: float = 2.0
	use_augmentation: bool = False
	augmentation_config: AugmentationConfig | None = None
	thresholds: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


@dataclass(frozen=True)
class EvaluationMetrics:
	"""
	Validation metrics for a segmentation model.
	"""

	loss: float
	dice: float
	iou: float
	pixel_accuracy: float
	threshold: float


def get_device() -> torch.device:
	"""
	Choose the best available PyTorch device.

	Returns:
		MPS on Apple Silicon if available, CUDA if available, otherwise CPU.
	"""

	if torch.backends.mps.is_available():
		return torch.device("mps")

	if torch.cuda.is_available():
		return torch.device("cuda")

	return torch.device("cpu")


def make_batches(
	dataset: VessMAPDataset,
	batch_size: int,
	shuffle: bool = False,
	seed: int = 42,
	augment: bool = False,
	augmentation_config: AugmentationConfig | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
	"""
	Yield image/mask batches from a VessMAP dataset.

	Args:
		dataset: Dataset split to batch.
		batch_size: Number of samples per batch.
		shuffle: Whether to shuffle sample order before batching.
		seed: Random seed used when shuffling and augmenting.
		augment: Whether to apply image/mask augmentations.
		augmentation_config: Optional augmentation settings.

	Yields:
		image_batch: Float tensor with shape (batch, 1, height, width).
		mask_batch: Float tensor with shape (batch, 1, height, width).
	"""

	indices = np.arange(len(dataset))

	if shuffle:
		rng = np.random.default_rng(seed)
		rng.shuffle(indices)

	for start in range(0, len(indices), batch_size):
		batch_indices = indices[start:start + batch_size]
		images: list[np.ndarray] = []
		masks: list[np.ndarray] = []

		for index in batch_indices:
			image, mask, _ = dataset[int(index)]

			if augment:
				image, mask = augment_pair(
					image=image,
					mask=mask,
					config=augmentation_config,
					seed=seed + int(index),
				)

			images.append(image)
			masks.append(mask)

		yield (
			torch.from_numpy(np.stack(images)).float(),
			torch.from_numpy(np.stack(masks)).float(),
		)


def dice_loss(
	logits: torch.Tensor,
	targets: torch.Tensor,
	smooth: float = 1e-6,
) -> torch.Tensor:
	"""
	Compute soft Dice loss from model logits and binary targets.

	Args:
		logits: Raw model outputs with shape (batch, 1, height, width).
		targets: Binary target masks with shape (batch, 1, height, width).
		smooth: Small value that prevents division by zero.

	Returns:
		Dice loss. Lower is better.
	"""

	probabilities = torch.sigmoid(logits)

	intersection = (probabilities * targets).sum()
	denominator = probabilities.sum() + targets.sum()
	dice = (2.0 * intersection + smooth) / (denominator + smooth)

	return 1.0 - dice


def segmentation_loss(
	logits: torch.Tensor,
	targets: torch.Tensor,
	bce_loss: nn.Module,
	bce_weight: float = 0.5,
	dice_weight: float = 0.5,
) -> torch.Tensor:
	"""
	Combine BCE loss with soft Dice loss.

	BCE handles pixel-level classification, while Dice loss directly rewards mask
	overlap. Combining them usually works better for thin vessel masks than using
	BCE alone.
	"""

	return (
		bce_weight * bce_loss(logits, targets)
		+ dice_weight * dice_loss(logits, targets)
	)


def evaluate_model(
	model: nn.Module,
	dataset: VessMAPDataset,
	loss_function: nn.Module,
	device: torch.device,
	batch_size: int,
	thresholds: tuple[float, ...],
) -> EvaluationMetrics:
	"""
	Evaluate a segmentation model on a dataset split.

	Args:
		model: Segmentation model that outputs logits.
		dataset: Dataset split to evaluate.
		loss_function: Loss function used for reporting validation loss.
		device: PyTorch device.
		batch_size: Evaluation batch size.
		thresholds: Thresholds tested when turning probabilities into masks.

	Returns:
		EvaluationMetrics containing average loss, Dice, IoU, and pixel accuracy.
	"""

	if not thresholds:
		raise ValueError("thresholds must contain at least one value")

	model.eval()

	losses: list[float] = []
	probability_maps: list[np.ndarray] = []
	target_masks: list[np.ndarray] = []

	with torch.no_grad():
		for image_batch, mask_batch in make_batches(dataset, batch_size):
			image_batch = image_batch.to(device)
			mask_batch = mask_batch.to(device)

			logits = model(image_batch)
			loss = segmentation_loss(
				logits=logits,
				targets=mask_batch,
				bce_loss=loss_function,
			)
			losses.append(float(loss.item()))

			probabilities = torch.sigmoid(logits).cpu().numpy()
			targets = mask_batch.cpu().numpy()

			for probability, target in zip(probabilities, targets):
				probability_maps.append(probability)
				target_masks.append(target)

	best_metrics = EvaluationMetrics(
		loss=float(np.mean(losses)),
		dice=-1.0,
		iou=0.0,
		pixel_accuracy=0.0,
		threshold=thresholds[0],
	)

	# Try several thresholds each epoch because thin vessels may score better away from 0.5.
	for threshold in thresholds:
		dice_scores: list[float] = []
		iou_scores: list[float] = []
		pixel_accuracies: list[float] = []

		for probability, target in zip(probability_maps, target_masks):
			prediction = probability >= threshold
			dice_scores.append(dice_score(prediction, target))
			iou_scores.append(iou_score(prediction, target))
			pixel_accuracies.append(pixel_accuracy(prediction, target))

		mean_dice = float(np.mean(dice_scores))

		if mean_dice > best_metrics.dice:
			best_metrics = EvaluationMetrics(
				loss=float(np.mean(losses)),
				dice=mean_dice,
				iou=float(np.mean(iou_scores)),
				pixel_accuracy=float(np.mean(pixel_accuracies)),
				threshold=threshold,
			)

	return best_metrics


def train_model(model: nn.Module, config: TrainingConfig) -> nn.Module:
	"""
	Train a segmentation model on VessMAP.

	Args:
		model: Model to train. It should output logits, not sigmoid probabilities.
		config: Training configuration.

	Returns:
		The trained model.
	"""

	device = get_device()
	print(f"Using device: {device}")

	train_dataset = VessMAPDataset(config.dataset_root, split="train")
	val_dataset = VessMAPDataset(config.dataset_root, split="val")

	model = model.to(device)
	positive_weight = torch.tensor([config.positive_weight], device=device)
	bce_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=config.learning_rate,
		weight_decay=1e-4,
	)
	best_dice = -1.0

	for epoch in range(1, config.epochs + 1):
		model.train()
		train_losses: list[float] = []

		for image_batch, mask_batch in make_batches(
			train_dataset,
			config.batch_size,
			shuffle=True,
			seed=epoch,
			augment=config.use_augmentation,
			augmentation_config=config.augmentation_config,
		):
			image_batch = image_batch.to(device)
			mask_batch = mask_batch.to(device)

			optimizer.zero_grad()
			logits = model(image_batch)
			loss = segmentation_loss(
				logits=logits,
				targets=mask_batch,
				bce_loss=bce_loss,
			)
			loss.backward()
			optimizer.step()

			train_losses.append(float(loss.item()))

		val_metrics = evaluate_model(
			model=model,
			dataset=val_dataset,
			loss_function=bce_loss,
			device=device,
			batch_size=config.batch_size,
			thresholds=config.thresholds,
		)

		print(
			f"Epoch {epoch:02d} | "
			f"train_loss={float(np.mean(train_losses)):.4f} | "
			f"val_loss={val_metrics.loss:.4f} | "
			f"dice={val_metrics.dice:.4f} | "
			f"iou={val_metrics.iou:.4f} | "
			f"pixel_acc={val_metrics.pixel_accuracy:.4f} | "
			f"threshold={val_metrics.threshold:.2f}"
		)

		# Save only the best validation checkpoint instead of blindly saving the last epoch.
		if val_metrics.dice > best_dice:
			best_dice = val_metrics.dice
			config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
			torch.save(model.state_dict(), config.checkpoint_path)
			print(
				f"Saved new best checkpoint: {config.checkpoint_path} "
				f"(Dice={best_dice:.4f}, threshold={val_metrics.threshold:.2f})"
			)

	print(f"Best validation Dice: {best_dice:.4f}")

	return model