import matplotlib.pyplot as plt
import veseg


result = veseg.predict(
	"test_images/structure1.png",
	mode=veseg.RED,
)

# Smooth scaling: resize probability map before thresholding.
resized_mask = veseg.postprocess_prediction(
	result.probability,
	threshold=0.70,
	min_neighbors=2,
	output_size=(1000, 1000),
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(result.mask, cmap="gray")
axes[0].set_title(f"Original mask\n{result.mask.shape}")
axes[0].axis("off")

axes[1].imshow(resized_mask, cmap="gray")
axes[1].set_title(f"Resized mask\n{resized_mask.shape}")
axes[1].axis("off")

plt.tight_layout()
plt.show()