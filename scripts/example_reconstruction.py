

import matplotlib.pyplot as plt
import numpy as np
import veseg


result = veseg.predict(
	"test_images/structure1.png",
	mode=veseg.RED,
)

# Resize the soft model output before thresholding so the large mask is smoother.
mask = veseg.postprocess_prediction(
	result.probability,
	threshold=0.70,
	min_neighbors=2,
	output_size=(1000, 1000),
).astype(bool)

skeleton = veseg.skeletonize_mask(mask)
distance_map = veseg.distance_transform_mask(mask)
reconstruction = veseg.reconstruct_vessel_mask(skeleton, distance_map)

# Show skeleton centerlines in red on top of the reconstructed vessel mask.
reconstruction_overlay = np.zeros((*reconstruction.shape, 3), dtype=float)
reconstruction_overlay[..., 0] = reconstruction.astype(float)
reconstruction_overlay[..., 1] = reconstruction.astype(float)
reconstruction_overlay[..., 2] = reconstruction.astype(float)
reconstruction_overlay[skeleton] = [1.0, 0.0, 0.0]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

axes[0].imshow(mask, cmap="gray")
axes[0].set_title("Resized mask")
axes[0].axis("off")

axes[1].imshow(skeleton, cmap="gray")
axes[1].set_title("Skeleton")
axes[1].axis("off")

axes[2].imshow(distance_map)
axes[2].set_title("Distance transform")
axes[2].axis("off")

axes[3].imshow(reconstruction, cmap="gray")
axes[3].set_title("Reconstruction")
axes[3].axis("off")

axes[4].imshow(reconstruction_overlay)
axes[4].set_title("Reconstruction + skeleton")
axes[4].axis("off")

plt.tight_layout()
plt.show()