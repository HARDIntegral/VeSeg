use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::distance_transform::distance_transform;
use crate::graph::{NodeKind, build_vessel_graph, detect_graph_nodes};
use crate::radius::estimate_edge_radii;
use crate::reconstruction::reconstruct_vessels;
use crate::skeletonize::skeletonize;

#[pyfunction]
pub fn skeletonize_mask(py: Python<'_>, mask: PyReadonlyArray2<'_, bool>) -> Py<PyArray2<bool>> {
    let rust_mask = mask.as_array().to_owned();
    let skeleton = skeletonize(&rust_mask);

    PyArray2::from_owned_array(py, skeleton).into()
}

#[pyfunction]
pub fn distance_transform_mask(
    py: Python<'_>,
    mask: PyReadonlyArray2<'_, bool>,
) -> Py<PyArray2<f32>> {
    let rust_mask = mask.as_array().to_owned();
    let distances = distance_transform(&rust_mask);

    PyArray2::from_owned_array(py, distances).into()
}

#[pyfunction]
pub fn reconstruct_vessel_mask(
    py: Python<'_>,
    skeleton: PyReadonlyArray2<'_, bool>,
    distance_map: PyReadonlyArray2<'_, f32>,
) -> Py<PyArray2<bool>> {
    let rust_skeleton = skeleton.as_array().to_owned();
    let rust_distance_map = distance_map.as_array().to_owned();
    let reconstruction = reconstruct_vessels(&rust_skeleton, &rust_distance_map);

    PyArray2::from_owned_array(py, reconstruction).into()
}

#[pyfunction]
pub fn detect_graph_nodes_mask(
    py: Python<'_>,
    skeleton: PyReadonlyArray2<'_, bool>,
) -> Py<PyArray2<i64>> {
    let rust_skeleton = skeleton.as_array().to_owned();
    let nodes = detect_graph_nodes(&rust_skeleton);
    let mut output = ndarray::Array2::<i64>::zeros((nodes.len(), 4));

    for (index, node) in nodes.iter().enumerate() {
        output[(index, 0)] = node.id as i64;
        output[(index, 1)] = node.row as i64;
        output[(index, 2)] = node.col as i64;
        output[(index, 3)] = match node.kind {
            NodeKind::Endpoint => 0,
            NodeKind::Junction => 1,
        };
    }

    PyArray2::from_owned_array(py, output).into()
}

#[pyfunction]
pub fn build_vessel_graph_mask(
    py: Python<'_>,
    skeleton: PyReadonlyArray2<'_, bool>,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let rust_skeleton = skeleton.as_array().to_owned();
    let graph = build_vessel_graph(&rust_skeleton);

    let mut node_output = ndarray::Array2::<i64>::zeros((graph.nodes.len(), 4));
    for (index, node) in graph.nodes.iter().enumerate() {
        node_output[(index, 0)] = node.id as i64;
        node_output[(index, 1)] = node.row as i64;
        node_output[(index, 2)] = node.col as i64;
        node_output[(index, 3)] = match node.kind {
            NodeKind::Endpoint => 0,
            NodeKind::Junction => 1,
        };
    }

    let mut edge_output = ndarray::Array2::<f32>::zeros((graph.edges.len(), 4));
    for (index, edge) in graph.edges.iter().enumerate() {
        edge_output[(index, 0)] = edge.id as f32;
        edge_output[(index, 1)] = edge.start_node as f32;
        edge_output[(index, 2)] = edge.end_node as f32;
        edge_output[(index, 3)] = edge.length_px;
    }

    Ok((
        PyArray2::from_owned_array(py, node_output).into(),
        PyArray2::from_owned_array(py, edge_output).into(),
    ))
}

#[pyfunction]
pub fn build_vessel_geometry_mask(
    py: Python<'_>,
    skeleton: PyReadonlyArray2<'_, bool>,
    distance_map: PyReadonlyArray2<'_, f32>,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let rust_skeleton = skeleton.as_array().to_owned();
    let rust_distance_map = distance_map.as_array().to_owned();
    let graph = build_vessel_graph(&rust_skeleton);
    let radii = estimate_edge_radii(&graph.edges, &rust_distance_map);

    let mut node_output = ndarray::Array2::<i64>::zeros((graph.nodes.len(), 4));
    for (index, node) in graph.nodes.iter().enumerate() {
        node_output[(index, 0)] = node.id as i64;
        node_output[(index, 1)] = node.row as i64;
        node_output[(index, 2)] = node.col as i64;
        node_output[(index, 3)] = match node.kind {
            NodeKind::Endpoint => 0,
            NodeKind::Junction => 1,
        };
    }

    // Edge columns:
    // id, start_node, end_node, length_px, mean_radius_px,
    // min_radius_px, max_radius_px, mean_radius_norm
    let mut edge_output = ndarray::Array2::<f32>::zeros((graph.edges.len(), 8));
    for (index, edge) in graph.edges.iter().enumerate() {
        let radius = &radii[index];

        edge_output[(index, 0)] = edge.id as f32;
        edge_output[(index, 1)] = edge.start_node as f32;
        edge_output[(index, 2)] = edge.end_node as f32;
        edge_output[(index, 3)] = edge.length_px;
        edge_output[(index, 4)] = radius.mean_radius_px;
        edge_output[(index, 5)] = radius.min_radius_px;
        edge_output[(index, 6)] = radius.max_radius_px;
        edge_output[(index, 7)] = radius.mean_radius_norm;
    }

    Ok((
        PyArray2::from_owned_array(py, node_output).into(),
        PyArray2::from_owned_array(py, edge_output).into(),
    ))
}
