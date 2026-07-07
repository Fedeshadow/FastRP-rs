use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use fastrp::FastRPBuilder;

#[pyfunction]
#[pyo3(signature = (adj_matrix, dim, weights, seed=None))]
fn fit_dense<'py>(
    py: Python<'py>,
    adj_matrix: PyReadonlyArray2<'py, f64>,
    dim: usize,
    weights: Vec<f64>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let adj_matrix_owned = adj_matrix.as_array().to_owned();
    let builder = FastRPBuilder::new(dim)
        .with_weights(weights);
    let builder = if let Some(s) = seed {
        builder.with_seed(s)
    } else {
        builder
    };

    let result = builder.fit_dense(&adj_matrix_owned)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (row_ptrs, col_indices, values, num_nodes, dim, weights, seed=None))]
fn fit_csr<'py>(
    py: Python<'py>,
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
    num_nodes: usize,
    dim: usize,
    weights: Vec<f64>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let builder = FastRPBuilder::new(dim)
        .with_weights(weights);
    let builder = if let Some(s) = seed {
        builder.with_seed(s)
    } else {
        builder
    };

    let result = builder.from_csr(row_ptrs, col_indices, values, num_nodes)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (adj_list, dim, weights, seed=None))]
fn fit_adj_list<'py>(
    py: Python<'py>,
    adj_list: Vec<Vec<(usize, f64)>>,
    dim: usize,
    weights: Vec<f64>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let builder = FastRPBuilder::new(dim)
        .with_weights(weights);
    let builder = if let Some(s) = seed {
        builder.with_seed(s)
    } else {
        builder
    };

    let result = builder.fit_adj_list(adj_list)
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
