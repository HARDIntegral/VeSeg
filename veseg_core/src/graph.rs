use ndarray::Array2;
use std::collections::HashSet;

const ENDPOINT_CLUSTER_RADIUS: isize = 2;
const JUNCTION_CLUSTER_RADIUS: isize = 4;
const NODE_CAPTURE_RADIUS: f32 = 8.0;

// Graph nodes are places where vessel paths either start/end or branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Endpoint,
    Junction,
}

// Store one important skeleton region as a graph node.
// Edge tracing connects these nodes through normal path pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNode {
    pub id: usize,
    pub row: usize,
    pub col: usize,
    pub kind: NodeKind,
}

// Store one traced vessel segment between two graph nodes.
// The path is needed later for sampling vessel radius along the segment.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphEdge {
    pub id: usize,
    pub start_node: usize,
    pub end_node: usize,
    pub length_px: f32,
    pub path: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VesselGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

/// Build a simplified graph from a 1-pixel-wide skeleton.
///
/// The graph is built in three stages: detect graph nodes, trace edges through
/// skeleton path pixels, then suppress degree-2 nodes that only split one vessel
/// into two artificial segments.
pub fn build_vessel_graph(skeleton: &Array2<bool>) -> VesselGraph {
    let nodes = detect_graph_nodes(skeleton);
    let edges = trace_graph_edges(skeleton, &nodes);

    simplify_degree_two_nodes(nodes, edges)
}

/// Detect graph nodes from a 1-pixel-wide skeleton.
///
/// Raw endpoint and junction pixels are first detected by local connectivity.
/// Nearby raw nodes are then clustered into one representative node so a thick
/// junction blob does not become dozens of graph nodes.
pub fn detect_graph_nodes(skeleton: &Array2<bool>) -> Vec<GraphNode> {
    let raw_nodes = detect_raw_graph_nodes(skeleton);
    let endpoints = filter_nodes_by_kind(&raw_nodes, NodeKind::Endpoint);
    let junctions = filter_nodes_by_kind(&raw_nodes, NodeKind::Junction);

    let mut nodes = Vec::new();

    // Endpoints are usually compact, so they get a smaller clustering radius.
    nodes.extend(cluster_nodes_by_components(
        &endpoints,
        skeleton.dim(),
        NodeKind::Endpoint,
        ENDPOINT_CLUSTER_RADIUS,
    ));

    // Junctions often appear as blobs of high-degree pixels around one branch point.
    nodes.extend(cluster_nodes_by_components(
        &junctions,
        skeleton.dim(),
        NodeKind::Junction,
        JUNCTION_CLUSTER_RADIUS,
    ));

    // Reassign ids after clustering so Python receives compact node ids.
    for (id, node) in nodes.iter_mut().enumerate() {
        node.id = id;
    }

    nodes
}

fn detect_raw_graph_nodes(skeleton: &Array2<bool>) -> Vec<GraphNode> {
    let (height, width) = skeleton.dim();
    let mut nodes = Vec::new();

    // Scan every skeleton pixel and decide whether it should become a raw graph node.
    for row in 0..height {
        for col in 0..width {
            if !skeleton[(row, col)] {
                continue;
            }

            // Neighbor count tells us whether this pixel is an endpoint, path, or junction.
            let neighbor_count = count_skeleton_neighbors(skeleton, row, col);
            let kind = classify_node(neighbor_count);

            // Ignore normal path pixels for now. They will become edge paths later.
            if let Some(kind) = kind {
                nodes.push(GraphNode {
                    id: nodes.len(),
                    row,
                    col,
                    kind,
                });
            }
        }
    }

    nodes
}

// Trace vessel segments between detected nodes.
// Node-region pixels are cached to avoid rescanning the entire image for every node.
fn trace_graph_edges(skeleton: &Array2<bool>, nodes: &[GraphNode]) -> Vec<GraphEdge> {
    let (node_labels, pixels_by_node) = label_node_regions(skeleton, nodes);
    let mut edges = Vec::new();
    let mut seen_edges = HashSet::<(usize, usize)>::new();

    for node in nodes {
        for &(row, col) in &pixels_by_node[node.id] {
            for (neighbor_row, neighbor_col) in valid_neighbors(row, col, skeleton.dim()) {
                if !skeleton[(neighbor_row, neighbor_col)] {
                    continue;
                }

                if node_labels[(neighbor_row, neighbor_col)] == Some(node.id) {
                    continue;
                }

                if let Some((end_node, length_px, path)) = trace_edge_from_pixel(
                    skeleton,
                    &node_labels,
                    node.id,
                    (row, col),
                    (neighbor_row, neighbor_col),
                ) {
                    if node.id == end_node {
                        continue;
                    }

                    let key = ordered_pair(node.id, end_node);

                    if seen_edges.insert(key) {
                        edges.push(GraphEdge {
                            id: edges.len(),
                            start_node: key.0,
                            end_node: key.1,
                            length_px,
                            path,
                        });
                    }
                }
            }
        }
    }

    edges
}

fn trace_edge_from_pixel(
    skeleton: &Array2<bool>,
    node_labels: &Array2<Option<usize>>,
    start_node: usize,
    previous_pixel: (usize, usize),
    current_pixel: (usize, usize),
) -> Option<(usize, f32, Vec<(usize, usize)>)> {
    let mut previous = previous_pixel;
    let mut current = current_pixel;
    let mut length_px = pixel_distance(previous, current);
    let mut path = vec![previous, current];
    let mut visited = HashSet::<(usize, usize)>::new();

    loop {
        if !visited.insert(current) {
            return None;
        }

        if let Some(node_id) = node_labels[(current.0, current.1)] {
            if node_id != start_node {
                return Some((node_id, length_px, path));
            }
        }

        let next_pixels: Vec<(usize, usize)> =
            valid_neighbors(current.0, current.1, skeleton.dim())
                .into_iter()
                .filter(|&pixel| pixel != previous)
                .filter(|&(row, col)| skeleton[(row, col)])
                .collect();

        if next_pixels.is_empty() {
            return None;
        }

        // Once the trace reaches a branch-like ambiguity, stop at any labeled node if possible.
        // Full path disambiguation can be tightened later when pruning is added.
        let next = next_pixels[0];
        length_px += pixel_distance(current, next);
        path.push(next);
        previous = current;
        current = next;
    }
}

fn simplify_degree_two_nodes(nodes: Vec<GraphNode>, edges: Vec<GraphEdge>) -> VesselGraph {
    let mut active_nodes = vec![true; nodes.len()];
    let mut active_edges = edges;
    let mut changed = true;

    while changed {
        changed = false;
        let adjacency = build_adjacency(nodes.len(), &active_edges, &active_nodes);

        for node_id in 0..nodes.len() {
            if !active_nodes[node_id] {
                continue;
            }

            let connected = &adjacency[node_id];

            if connected.len() != 2 {
                continue;
            }

            let first_edge_index = connected[0];
            let second_edge_index = connected[1];
            let first_neighbor = other_node(&active_edges[first_edge_index], node_id);
            let second_neighbor = other_node(&active_edges[second_edge_index], node_id);

            if first_neighbor == second_neighbor {
                continue;
            }

            if edge_exists_between(&active_edges, first_neighbor, second_neighbor) {
                continue;
            }

            let merged_length = active_edges[first_edge_index].length_px
                + active_edges[second_edge_index].length_px;
            let merged_path = merge_edge_paths(
                &active_edges[first_edge_index],
                &active_edges[second_edge_index],
                node_id,
            );

            active_nodes[node_id] = false;
            remove_edges_at_node(&mut active_edges, node_id);
            active_edges.push(GraphEdge {
                id: active_edges.len(),
                start_node: first_neighbor,
                end_node: second_neighbor,
                length_px: merged_length,
                path: merged_path,
            });

            changed = true;
            break;
        }
    }

    let (new_nodes, id_map) = compact_nodes(nodes, active_nodes);
    let new_edges = compact_edges(active_edges, &id_map);

    VesselGraph {
        nodes: new_nodes,
        edges: new_edges,
    }
}

fn label_node_regions(
    skeleton: &Array2<bool>,
    nodes: &[GraphNode],
) -> (Array2<Option<usize>>, Vec<Vec<(usize, usize)>>) {
    let (height, width) = skeleton.dim();
    let mut labels = Array2::<Option<usize>>::from_elem((height, width), None);
    let mut pixels_by_node = vec![Vec::new(); nodes.len()];

    for row in 0..height {
        for col in 0..width {
            if !skeleton[(row, col)] {
                continue;
            }

            let mut best = None;
            let mut best_dist = f32::MAX;

            for node in nodes {
                let dist = distance_to_point(row, col, node.row, node.col);
                if dist <= NODE_CAPTURE_RADIUS && dist < best_dist {
                    best = Some(node.id);
                    best_dist = dist;
                }
            }

            labels[(row, col)] = best;
            if let Some(id) = best {
                pixels_by_node[id].push((row, col));
            }
        }
    }

    (labels, pixels_by_node)
}

fn build_adjacency(
    node_count: usize,
    edges: &[GraphEdge],
    active_nodes: &[bool],
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); node_count];

    for (idx, edge) in edges.iter().enumerate() {
        if active_nodes[edge.start_node] && active_nodes[edge.end_node] {
            adjacency[edge.start_node].push(idx);
            adjacency[edge.end_node].push(idx);
        }
    }

    adjacency
}

fn compact_nodes(
    nodes: Vec<GraphNode>,
    active_nodes: Vec<bool>,
) -> (Vec<GraphNode>, Vec<Option<usize>>) {
    let mut new_nodes = Vec::new();
    let mut id_map = vec![None; active_nodes.len()];

    for node in nodes {
        if !active_nodes[node.id] {
            continue;
        }

        let old_id = node.id;
        let new_id = new_nodes.len();
        id_map[old_id] = Some(new_id);
        new_nodes.push(GraphNode { id: new_id, ..node });
    }

    (new_nodes, id_map)
}

fn compact_edges(edges: Vec<GraphEdge>, id_map: &[Option<usize>]) -> Vec<GraphEdge> {
    let mut compacted = Vec::new();
    let mut seen = HashSet::<(usize, usize)>::new();

    for edge in edges {
        let Some(start_node) = id_map.get(edge.start_node).copied().flatten() else {
            continue;
        };
        let Some(end_node) = id_map.get(edge.end_node).copied().flatten() else {
            continue;
        };

        if start_node == end_node {
            continue;
        }

        let key = ordered_pair(start_node, end_node);

        if seen.insert(key) {
            compacted.push(GraphEdge {
                id: compacted.len(),
                start_node: key.0,
                end_node: key.1,
                length_px: edge.length_px,
                path: edge.path,
            });
        }
    }

    compacted
}

fn merge_edge_paths(
    first: &GraphEdge,
    second: &GraphEdge,
    shared_node: usize,
) -> Vec<(usize, usize)> {
    let mut first_path = oriented_path_away_from_node(first, shared_node);
    let mut second_path = oriented_path_away_from_node(second, shared_node);

    first_path.reverse();

    if !first_path.is_empty() && !second_path.is_empty() && first_path.last() == second_path.first()
    {
        second_path.remove(0);
    }

    first_path.extend(second_path);
    first_path
}

fn oriented_path_away_from_node(edge: &GraphEdge, node_id: usize) -> Vec<(usize, usize)> {
    if edge.start_node == node_id {
        edge.path.clone()
    } else {
        let mut reversed = edge.path.clone();
        reversed.reverse();
        reversed
    }
}

fn remove_edges_at_node(edges: &mut Vec<GraphEdge>, node_id: usize) {
    edges.retain(|edge| edge.start_node != node_id && edge.end_node != node_id);
}

fn other_node(edge: &GraphEdge, node_id: usize) -> usize {
    if edge.start_node == node_id {
        edge.end_node
    } else {
        edge.start_node
    }
}

fn edge_exists_between(edges: &[GraphEdge], first: usize, second: usize) -> bool {
    edges.iter().any(|edge| {
        (edge.start_node == first && edge.end_node == second)
            || (edge.start_node == second && edge.end_node == first)
    })
}

fn ordered_pair(first: usize, second: usize) -> (usize, usize) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

fn filter_nodes_by_kind(nodes: &[GraphNode], kind: NodeKind) -> Vec<GraphNode> {
    nodes
        .iter()
        .filter(|node| node.kind == kind)
        .cloned()
        .collect()
}

fn cluster_nodes_by_components(
    nodes: &[GraphNode],
    shape: (usize, usize),
    kind: NodeKind,
    cluster_radius: isize,
) -> Vec<GraphNode> {
    if nodes.is_empty() {
        return Vec::new();
    }

    let mut node_mask = build_node_cluster_mask(nodes, shape, cluster_radius);
    let mut clusters = Vec::<Vec<(usize, usize)>>::new();

    for node in nodes {
        if !node_mask[(node.row, node.col)] {
            continue;
        }

        clusters.push(flood_fill_node_cluster(&mut node_mask, node.row, node.col));
    }

    clusters
        .iter()
        .enumerate()
        .map(|(id, cluster)| pixel_cluster_centroid(id, cluster, kind))
        .collect()
}

fn build_node_cluster_mask(
    nodes: &[GraphNode],
    shape: (usize, usize),
    cluster_radius: isize,
) -> Array2<bool> {
    let (height, width) = shape;
    let mut mask = Array2::<bool>::from_elem((height, width), false);

    // Dilate raw node pixels so nearby pieces of the same junction become one component.
    for node in nodes {
        for_each_disk_pixel(
            node.row,
            node.col,
            height,
            width,
            cluster_radius,
            |row, col| {
                mask[(row, col)] = true;
            },
        );
    }

    mask
}

fn flood_fill_node_cluster(
    mask: &mut Array2<bool>,
    start_row: usize,
    start_col: usize,
) -> Vec<(usize, usize)> {
    let mut stack = vec![(start_row, start_col)];
    let mut cluster = Vec::new();

    mask[(start_row, start_col)] = false;

    while let Some((row, col)) = stack.pop() {
        cluster.push((row, col));

        for (neighbor_row, neighbor_col) in valid_neighbors(row, col, mask.dim()) {
            if !mask[(neighbor_row, neighbor_col)] {
                continue;
            }

            mask[(neighbor_row, neighbor_col)] = false;
            stack.push((neighbor_row, neighbor_col));
        }
    }

    cluster
}

fn pixel_cluster_centroid(id: usize, cluster: &[(usize, usize)], kind: NodeKind) -> GraphNode {
    let mut row_total = 0.0;
    let mut col_total = 0.0;

    for &(row, col) in cluster {
        row_total += row as f32;
        col_total += col as f32;
    }

    let count = cluster.len() as f32;

    GraphNode {
        id,
        row: (row_total / count).round() as usize,
        col: (col_total / count).round() as usize,
        kind,
    }
}

fn for_each_disk_pixel<F>(
    center_row: usize,
    center_col: usize,
    height: usize,
    width: usize,
    radius: isize,
    mut visit: F,
) where
    F: FnMut(usize, usize),
{
    let center_row = center_row as isize;
    let center_col = center_col as isize;
    let radius_squared = radius * radius;

    for row_offset in -radius..=radius {
        for col_offset in -radius..=radius {
            if row_offset * row_offset + col_offset * col_offset > radius_squared {
                continue;
            }

            let row = center_row + row_offset;
            let col = center_col + col_offset;

            if row < 0 || col < 0 {
                continue;
            }

            let row = row as usize;
            let col = col as usize;

            if row < height && col < width {
                visit(row, col);
            }
        }
    }
}

fn distance_to_point(row: usize, col: usize, target_row: usize, target_col: usize) -> f32 {
    let row_delta = row as f32 - target_row as f32;
    let col_delta = col as f32 - target_col as f32;

    (row_delta * row_delta + col_delta * col_delta).sqrt()
}

fn pixel_distance(first: (usize, usize), second: (usize, usize)) -> f32 {
    distance_to_point(first.0, first.1, second.0, second.1)
}

// Classify skeleton pixels by local connectivity.
fn classify_node(neighbor_count: usize) -> Option<NodeKind> {
    match neighbor_count {
        // One neighbor means the vessel terminates here.
        1 => Some(NodeKind::Endpoint),
        // Three or more neighbors means multiple vessel paths meet here.
        3..=8 => Some(NodeKind::Junction),
        // Zero-neighbor noise and two-neighbor path pixels are not graph nodes yet.
        _ => None,
    }
}

// Count how many nearby pixels are part of the skeleton.
fn count_skeleton_neighbors(skeleton: &Array2<bool>, row: usize, col: usize) -> usize {
    valid_neighbors(row, col, skeleton.dim())
        .iter()
        .filter(|&&(neighbor_row, neighbor_col)| skeleton[(neighbor_row, neighbor_col)])
        .count()
}

// Return valid 8-connected neighbors.
// This allocates; future optimization is replacing with visitor pattern.
// Return valid 8-connected neighbors without going outside image bounds.
fn valid_neighbors(row: usize, col: usize, shape: (usize, usize)) -> Vec<(usize, usize)> {
    let (height, width) = shape;
    let mut neighbors = Vec::with_capacity(8);

    // Use 8-connectivity so diagonal skeleton segments stay connected.
    for row_offset in -1..=1 {
        for col_offset in -1..=1 {
            if row_offset == 0 && col_offset == 0 {
                continue;
            }

            let neighbor_row = row as isize + row_offset;
            let neighbor_col = col as isize + col_offset;

            // Skip pixels above or left of the image.
            if neighbor_row < 0 || neighbor_col < 0 {
                continue;
            }

            let neighbor_row = neighbor_row as usize;
            let neighbor_col = neighbor_col as usize;

            // Keep only pixels inside the image.
            if neighbor_row < height && neighbor_col < width {
                neighbors.push((neighbor_row, neighbor_col));
            }
        }
    }

    neighbors
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // A long straight line should only have two endpoints and one traced edge.
    // The line needs to be longer than the node merge/capture radius used for real images.
    #[test]
    fn builds_line_graph() {
        let mut skeleton = Array2::<bool>::from_elem((5, 25), false);

        for col in 2..=22 {
            skeleton[(2, col)] = true;
        }

        let graph = build_vessel_graph(&skeleton);
        let endpoints = graph
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Endpoint)
            .count();
        let junctions = graph
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Junction)
            .count();

        assert_eq!(endpoints, 2);
        assert_eq!(junctions, 0);
        assert_eq!(graph.edges.len(), 1);
    }

    // A cross-shaped skeleton should collapse clustered junction pixels into one node.
    #[test]
    fn detects_cross_junction() {
        let skeleton = array![
            [false, false, true, false, false],
            [false, false, true, false, false],
            [true, true, true, true, true],
            [false, false, true, false, false],
            [false, false, true, false, false],
        ];

        let graph = build_vessel_graph(&skeleton);
        let junctions = graph
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Junction)
            .count();

        assert_eq!(junctions, 1);
    }
}
