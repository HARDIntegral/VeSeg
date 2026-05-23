"""
VeSeg public API.

Most users should only need:

	import veseg

	result = veseg.predict("image.png", mode=veseg.ENHANCED_INVERTED)
	result.show()
	result.save_mask("mask.png")
"""

from veseg.predict import (
	ENHANCED_INVERTED,
	GRAYSCALE,
	INVERTED,
	RED,
	PredictionResult,
	predict,
)
from veseg.postprocessing import (
	postprocess_prediction,
	remove_specks,
	resize_mask,
	save_mask,
	scale_mask,
	threshold_probability,
)

__all__ = [
	"ENHANCED_INVERTED",
	"GRAYSCALE",
	"INVERTED",
	"RED",
	"PredictionResult",
	"predict",
	"postprocess_prediction",
	"remove_specks",
	"resize_mask",
	"save_mask",
	"scale_mask",
	"threshold_probability",
]