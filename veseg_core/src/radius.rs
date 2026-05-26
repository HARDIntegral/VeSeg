use crate::graph::GraphEdge;
use ndarray::Array2;

#[derive(Debug, Clone, PartialEq)]
pub struct EdgeRadius {
    pub edge_id: usize,
    pub mean_radius_px: f32,
    pub min_radius_px: f32,
    pub max_radius_px: f32,
    pub mean_radius_norm: f32,
}

/// Estimate vessel radius statistics for each graph edge.
///
/// Edge paths already store traced skeleton pixels. The distance transform gives
/// an approximate local vessel radius at each pixel. Sampling the distance map
/// along an edge path produces edge-level geometry summaries.
pub fn estimate_edge_radii(edges: &[GraphEdge], distance_map: &Array2<f32>) -> Vec<EdgeRadius> {
    let (height, width) = distance_map.dim();
    let normalization = height.max(width) as f32;

    edges
        .iter()
        .map(|edge| estimate_edge_radius(edge, distance_map, normalization))
        .collect()
}

fn estimate_edge_radius(
    edge: &GraphEdge,
    distance_map: &Array2<f32>,
    normalization: f32,
) -> EdgeRadius {
    let mut sum = 0.0_f32;
    let mut count = 0usize;
    let mut min_radius = f32::MAX;
    let mut max_radius = 0.0_f32;

    // Sample radius along the traced vessel centerline.
    for &(row, col) in &edge.path {
        let radius = distance_map[(row, col)];

        if radius <= 0.0 {
            continue;
        }

        sum += radius;
        count += 1;
        min_radius = min_radius.min(radius);
        max_radius = max_radius.max(radius);
    }

    let mean_radius = match count {
        0 => 0.0,
        _ => sum / count as f32,
    };

    EdgeRadius {
        edge_id: edge.id,
        mean_radius_px: mean_radius,
        min_radius_px: if count > 0 { min_radius } else { 0.0 },
        max_radius_px: max_radius,
        mean_radius_norm: if normalization > 0.0 {
            mean_radius / normalization
        } else {
            0.0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphEdge;
    use ndarray::Array2;

    #[test]
    fn estimates_radius_from_edge_path() {
        let edge = GraphEdge {
            id: 0,
            start_node: 0,
            end_node: 1,
            length_px: 2.0,
            path: vec![(1, 1), (1, 2), (1, 3)],
        };

        let mut distance_map = Array2::<f32>::zeros((5, 5));
        distance_map[(1, 1)] = 2.0;
        distance_map[(1, 2)] = 4.0;
        distance_map[(1, 3)] = 6.0;

        let radii = estimate_edge_radii(&[edge], &distance_map);

        assert_eq!(radii[0].mean_radius_px, 4.0);
        assert_eq!(radii[0].min_radius_px, 2.0);
        assert_eq!(radii[0].max_radius_px, 6.0);
    }
}
