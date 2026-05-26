import matplotlib.pyplot as plt
import numpy as np
import veseg


result = veseg.predict(
	"test_images/structure1.png",
	mode=veseg.RED,
)

# Resize probability map before thresholding for smoother large masks.
mask = veseg.postprocess_prediction(
	result.probability,
	threshold=0.70,
	min_neighbors=2,
	output_size=(1000, 1000),
).astype(bool)


skeleton = veseg.skeletonize_mask(mask)
distance_map = veseg.distance_transform_mask(mask)

# Sample local vessel radius only along skeleton pixels.
skeleton_radii = np.where(skeleton, distance_map, 0.0)

# Overlay skeleton centerlines in red on top of the resized mask.
overlay = np.zeros((*mask.shape, 3), dtype=float)
overlay[..., 0] = mask.astype(float)
overlay[..., 1] = mask.astype(float)
overlay[..., 2] = mask.astype(float)
overlay[skeleton] = [1.0, 0.0, 0.0]

fig, axes = plt.subplots(1, 6, figsize=(24, 4))

axes[0].imshow(result.mask)
axes[0].set_title(f"Original mask\n{result.mask.shape}")
axes[0].axis("off")

axes[1].imshow(mask)
axes[1].set_title(f"Resized mask\n{mask.shape}")
axes[1].axis("off")

axes[2].imshow(skeleton)
axes[2].set_title("Skeleton")
axes[2].axis("off")


axes[3].imshow(overlay)
axes[3].set_title("Skeleton overlay")
axes[3].axis("off")

axes[4].imshow(distance_map)
axes[4].set_title("Distance transform")
axes[4].axis("off")

radius_plot = axes[5].imshow(skeleton_radii)
axes[5].set_title("Skeleton radii")
axes[5].axis("off")
fig.colorbar(radius_plot, ax=axes[5], fraction=0.046)

plt.tight_layout()
plt.show()