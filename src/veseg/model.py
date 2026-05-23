"""
Segmentation models for VeSeg.

This file contains the baseline model used for quick pipeline checks and the
U-Net used for actual vessel segmentation.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TinySegmenter(nn.Module):
	"""
	Small baseline model for sanity-checking training.
	"""

	def __init__(self, in_channels: int = 1, hidden_channels: int = 16) -> None:
		super().__init__()

		self.network = nn.Sequential(
			nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_channels, 1, kernel_size=1),
		)

	def forward(self, image: torch.Tensor) -> torch.Tensor:
		return self.network(image)


class DoubleConv(nn.Module):
	"""
	Two Conv-BatchNorm-ReLU blocks used throughout the U-Net.
	"""

	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()

		self.network = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, image: torch.Tensor) -> torch.Tensor:
		return self.network(image)


class UNet(nn.Module):
	"""

	Compact 2D U-Net for vessel segmentation.
	Architecture:
		Input
		  ↓
		Encoder (feature extraction + downsampling)
		  ↓
		Bottleneck
		  ↓
		Decoder (upsampling + skip connections)
		  ↓
		Pixelwise logits
		The model returns logits rather than probabilities. Use
		BCEWithLogitsLoss during training and torch.sigmoid during inference.
	"""

	def __init__(
		self,
		in_channels: int = 1,
		out_channels: int = 1,
		features: tuple[int, ...] = (32, 64, 128, 256),
	) -> None:
		super().__init__()

		if not features:
			raise ValueError("features must contain at least one channel size")

		self.down_blocks = nn.ModuleList()
		self.up_transposes = nn.ModuleList()
		self.up_blocks = nn.ModuleList()
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		current_channels = in_channels

		# Encoder: repeatedly extract features, then downsample in forward().
		for feature_count in features:
			self.down_blocks.append(DoubleConv(current_channels, feature_count))
			current_channels = feature_count

		self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
		current_channels = features[-1] * 2

		# Decoder: upsample, concatenate the matching skip connection, then refine.
		for feature_count in reversed(features):
			self.up_transposes.append(
				nn.ConvTranspose2d(
					current_channels,
					feature_count,
					kernel_size=2,
					stride=2,
				)
			)
			self.up_blocks.append(DoubleConv(feature_count * 2, feature_count))
			current_channels = feature_count

		self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

	def forward(self, image: torch.Tensor) -> torch.Tensor:
		skip_connections: list[torch.Tensor] = []

		for down_block in self.down_blocks:
			image = down_block(image)
			skip_connections.append(image)
			image = self.pool(image)

		image = self.bottleneck(image)

		for up_transpose, up_block, skip_connection in zip(
			self.up_transposes,
			self.up_blocks,
			reversed(skip_connections),
		):
			image = up_transpose(image)

			# Odd image sizes can create a one-pixel mismatch after pooling/upsampling.
			if image.shape[2:] != skip_connection.shape[2:]:
				image = F.interpolate(
					image,
					size=skip_connection.shape[2:],
					mode="bilinear",
					align_corners=False,
				)

			image = torch.cat((skip_connection, image), dim=1)
			image = up_block(image)

		return self.output_conv(image)