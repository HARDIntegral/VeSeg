import matplotlib.pyplot as plt
import numpy as np
import veseg


# Build an RGB image so endpoints and junctions are easy to see.
def make_graph_overlay(reconstruction, skeleton, endpoints, junctions):
	"""Overlay graph node candidates on top of reconstructed vessels."""
	overlay = np.zeros((*reconstruction.shape, 3), dtype=float)
	overlay[..., 0] = reconstruction.astype(float)
	overlay[..., 1] = reconstruction.astype(float)
	overlay[..., 2] = reconstruction.astype(float)

	# Skeleton centerlines stay red.
	overlay[skeleton] = [1.0, 0.0, 0.0]

	# Rust graph nodes: endpoints are blue and junctions are green.
	for row, col in endpoints:
		overlay[row, col] = [0.0, 0.35, 1.0]

	for row, col in junctions:
		overlay[row, col] = [0.0, 1.0, 0.0]

	return overlay


result = veseg.predict(
	"test_images/structure1.png",
	mode=veseg.RED,
)

# Resize probability first so skeletonization works on smoother vessel geometry.
mask = veseg.postprocess_prediction(
	result.probability,
	threshold=0.70,
	min_neighbors=2,
	output_size=(1000, 1000),
).astype(bool)

skeleton = veseg.skeletonize_mask(mask)
distance_map = veseg.distance_transform_mask(mask)
reconstruction = veseg.reconstruct_vessel_mask(skeleton, distance_map)

nodes, edges = veseg.build_vessel_graph(skeleton)
endpoints = nodes[nodes[:, 3] == 0][:, 1:3].astype(int)
junctions = nodes[nodes[:, 3] == 1][:, 1:3].astype(int)

graph_overlay = make_graph_overlay(
	reconstruction,
	skeleton,
	endpoints,
	junctions,
)

print(f"Endpoints: {len(endpoints)}")
print(f"Junctions: {len(junctions)}")
print(f"Total graph nodes: {len(nodes)}")
print(f"Graph edges: {len(edges)}")

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(reconstruction, cmap="gray")
axes[0].set_title("Reconstructed vessels")
axes[0].axis("off")

axes[1].imshow(skeleton, cmap="gray")
axes[1].set_title("Skeleton")
axes[1].axis("off")

axes[2].imshow(reconstruction, cmap="gray")

# Draw simplified graph edges from Rust before graph nodes.
for _, start_node, end_node, _ in edges:
	start_node = int(start_node)
	end_node = int(end_node)
	start = nodes[start_node]
	end = nodes[end_node]
	axes[2].plot(
		[start[2], end[2]],
		[start[1], end[1]],
		linewidth=0.8,
	)

if len(endpoints) > 0:
	axes[2].scatter(endpoints[:, 1], endpoints[:, 0], s=18, label="Endpoints")

if len(junctions) > 0:
	axes[2].scatter(junctions[:, 1], junctions[:, 0], s=18, label="Junctions")

axes[2].set_title("Graph nodes")
axes[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2)
axes[2].axis("off")

axes[3].imshow(graph_overlay)
axes[3].set_title("Graph overlay")
axes[3].axis("off")

plt.tight_layout()
plt.show()