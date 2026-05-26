"""
VeSeg public API.

Most users should only need:

	import veseg

	result = veseg.predict("image.png", mode=veseg.ENHANCED_INVERTED)
	result.show()
	result.save_mask("mask.png")

For most users, the binary mask is the main output. Soft probability maps are
kept inside PredictionResult for advanced workflows, but probability-resizing
helpers are intentionally not part of the public API.
"""

from veseg.predict import (
	ENHANCED_INVERTED,
	GRAYSCALE,
	INVERTED,
	RED,
	PredictionResult,
	load_model,
	predict,
)
from veseg.postprocessing import (
	build_vessel_geometry,
	build_vessel_graph,
	distance_transform_mask,
	extract_vessel_geometry,
	postprocess_prediction,
	reconstruct_vessel_mask,
	remove_specks,
	resize_mask,
	save_mask,
	scale_mask,
	skeletonize_mask,
	threshold_probability,
)

__all__ = [
	"ENHANCED_INVERTED",
	"GRAYSCALE",
	"INVERTED",
	"RED",
	"PredictionResult",
	"load_model",
	"predict",
	"build_vessel_geometry",
	"build_vessel_graph",
	"distance_transform_mask",
	"extract_vessel_geometry",
	"reconstruct_vessel_mask",
	"skeletonize_mask",
	"postprocess_prediction",
	"remove_specks",
	"resize_mask",
	"save_mask",
	"scale_mask",
	"threshold_probability",
]