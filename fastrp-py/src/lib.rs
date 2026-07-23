use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use fastrp::FastRPBuilder;

#[pyfunction]
#[pyo3(signature = (adj_matrix, dim, weights, seed=None, undirected=true, node_features=None, feature_weight=1.0))]
fn fit_dense<'py>(
    py: Python<'py>,
    adj_matrix: PyReadonlyArray2<'py, f64>,
    dim: usize,
    weights: Vec<f64>,
    seed: Option<u64>,
    undirected: bool,
    node_features: Option<PyReadonlyArray2<'py, f64>>,
    feature_weight: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let adj_matrix_owned = adj_matrix.as_array().to_owned();
    let mut builder = FastRPBuilder::new(dim)
        .with_weights(weights)
        .with_undirected(undirected)
        .with_feature_weight(feature_weight);
    if let Some(s) = seed {
        builder = builder.with_seed(s);
    }
    if let Some(nf) = node_features {
        builder = builder.with_node_features(nf.as_array().to_owned());
    }

    let result = py.detach(|| {
        builder.fit_dense(&adj_matrix_owned)
    })
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (row_ptrs, col_indices, values, num_nodes, dim, weights, seed=None, undirected=true, node_features=None, feature_weight=1.0))]
fn fit_csr<'py>(
    py: Python<'py>,
    row_ptrs: PyReadonlyArray1<'py, i64>,
    col_indices: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
    num_nodes: usize,
    dim: usize,
    weights: Vec<f64>,
    seed: Option<u64>,
    undirected: bool,
    node_features: Option<PyReadonlyArray2<'py, f64>>,
    feature_weight: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Borrow numpy buffers as slices — zero-copy, no Python object protocol
    let rp = row_ptrs.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("row_ptrs must be contiguous: {e}")))?;
    let ci = col_indices.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("col_indices must be contiguous: {e}")))?;
    let vals = values.as_slice()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("values must be contiguous: {e}")))?;

    // Single native Rust loop: i64 → usize. No Python object overhead.
    let row_ptrs_vec: Vec<usize> = rp.iter().map(|&x| x as usize).collect();
    let col_indices_vec: Vec<usize> = ci.iter().map(|&x| x as usize).collect();
    let values_vec: Vec<f64> = vals.to_vec();

    let mut builder = FastRPBuilder::new(dim)
        .with_weights(weights)
        .with_undirected(undirected)
        .with_feature_weight(feature_weight);
    if let Some(s) = seed {
        builder = builder.with_seed(s);
    }
    if let Some(nf) = node_features {
        builder = builder.with_node_features(nf.as_array().to_owned());
    }

    let result = py.detach(|| {
        builder.from_csr(row_ptrs_vec, col_indices_vec, values_vec, num_nodes)
    })
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (adj_list, dim, iter_weights, seed=None, undirected=true, node_features=None, feature_weight=1.0))]
fn fit_adj_list<'py>(
    py: Python<'py>,
    adj_list: Vec<Vec<(usize, f64)>>,
    dim: usize,
    iter_weights: Vec<f64>,
    seed: Option<u64>,
    undirected: bool,
    node_features: Option<PyReadonlyArray2<'py, f64>>,
    feature_weight: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut builder = FastRPBuilder::new(dim)
        .with_weights(iter_weights)
        .with_undirected(undirected)
        .with_feature_weight(feature_weight);
    if let Some(s) = seed {
        builder = builder.with_seed(s);
    }
    if let Some(nf) = node_features {
        builder = builder.with_node_features(nf.as_array().to_owned());
    }

    let result = py.detach(|| {
        builder.fit_adj_list(adj_list)
    })
    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(result.into_pyarray(py))
}


#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_dense, m)?)?;
    m.add_function(wrap_pyfunction!(fit_csr, m)?)?;
    m.add_function(wrap_pyfunction!(fit_adj_list, m)?)?;
    Ok(())
}

