"""
Train a U-Net on the VessMAP dataset.

This script is the main training entrypoint for VeSeg's VessMAP experiments.
The reusable training loop lives in `veseg.train`, while this file defines the
model and run-specific configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from veseg.model import UNet
from veseg.preprocessing import AugmentationConfig
from veseg.train import TrainingConfig, train_model


DATASET_ROOT = PROJECT_ROOT / "data" / "VessMAP"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "vessmap_unet.pt"


def main() -> None:
	model = UNet(
		in_channels=1,
		out_channels=1,
		features=(32, 64, 128, 256),
	)

	config = TrainingConfig(
		dataset_root=DATASET_ROOT,
		checkpoint_path=CHECKPOINT_PATH,
		batch_size=4,
		epochs=50,
		learning_rate=1e-3,
		positive_weight=2.0,
		use_augmentation=True,
		augmentation_config=AugmentationConfig(),
	)

	train_model(model, config)


if __name__ == "__main__":
	main()