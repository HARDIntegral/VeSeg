import matplotlib.pyplot as plt
import numpy as np
import veseg


# Build vessel geometry and visualize estimated radii per edge.
result = veseg.predict(
	"test_images/structure1.png",
	mode=veseg.RED,
)

# Resize probability first so skeletonization preserves smoother vessels.
mask = veseg.postprocess_prediction(
	result.probability,
	threshold=0.70,
	min_neighbors=2,
	output_size=(1000, 1000),
).astype(bool)

skeleton = veseg.skeletonize_mask(mask)
distance_map = veseg.distance_transform_mask(mask)
reconstruction = veseg.reconstruct_vessel_mask(
	skeleton,
	distance_map,
)

nodes, edges = veseg.build_vessel_geometry(
	skeleton,
	distance_map,
)

print(f"Nodes: {len(nodes)}")
print(f"Edges: {len(edges)}")
print(f"Mean edge radius (px): {edges[:,4].mean():.2f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Show reconstructed vessels.
axes[0].imshow(reconstruction, cmap="gray")
axes[0].set_title("Reconstructed vessels")
axes[0].axis("off")

# Show raw distance map.
axes[1].imshow(distance_map)
axes[1].set_title("Distance transform\n(local radius field)")
axes[1].axis("off")

# Overlay graph colored by estimated radius.
axes[2].imshow(reconstruction, cmap="gray")

max_radius = max(edges[:,4].max(), 1.0)

for edge in edges:
	edge_id = int(edge[0])
	start_node = int(edge[1])
	end_node = int(edge[2])
	mean_radius = edge[4]
	normalized_radius = mean_radius / max_radius

	start = nodes[start_node]
	end = nodes[end_node]

	# Larger vessels appear brighter.
	axes[2].plot(
		[start[2], end[2]],
		[start[1], end[1]],
		linewidth=0.8 + normalized_radius * 4,
		alpha=0.8,
	)

endpoints = nodes[nodes[:,3] == 0]
junctions = nodes[nodes[:,3] == 1]

if len(endpoints):
	axes[2].scatter(
		endpoints[:,2],
		endpoints[:,1],
		s=20,
		label="Endpoints"
	)

if len(junctions):
	axes[2].scatter(
		junctions[:,2],
		junctions[:,1],
		s=20,
		label="Junctions"
	)

axes[2].set_title("Graph + estimated edge radii")
axes[2].legend()
axes[2].axis("off")

plt.tight_layout()
plt.show()
