mod distance_transform;
mod graph;
mod python;
mod radius;
mod reconstruction;
mod skeletonize;

use pyo3::prelude::*;
use python::{
    build_vessel_geometry_mask, build_vessel_graph_mask, detect_graph_nodes_mask,
    distance_transform_mask, reconstruct_vessel_mask, skeletonize_mask,
};

#[pymodule]
fn veseg_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(skeletonize_mask, m)?)?;
    m.add_function(wrap_pyfunction!(distance_transform_mask, m)?)?;
    m.add_function(wrap_pyfunction!(reconstruct_vessel_mask, m)?)?;
    m.add_function(wrap_pyfunction!(detect_graph_nodes_mask, m)?)?;
    m.add_function(wrap_pyfunction!(build_vessel_graph_mask, m)?)?;
    m.add_function(wrap_pyfunction!(build_vessel_geometry_mask, m)?)?;
    Ok(())
}
