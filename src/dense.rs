use ndarray::Array2;
use crate::algorithm::initialize_h0;
use crate::error::FastRPError;

/// A wrapper for Dense networks utilizing straightforward matrix multiplication.
/// Suitable mostly for smaller subgraphs.
pub struct DenseMatrix {
    pub adj_matrix: Array2<f64>,
    pub num_nodes: usize,
}

impl DenseMatrix {
    pub fn build_from_adj_list(num_nodes: usize, adj_list: Vec<Vec<(usize, f64)>>) -> Result<Self, FastRPError> {
        let capacity = num_nodes.checked_mul(num_nodes).ok_or_else(|| FastRPError::ShapeMismatch("Capacity overflow".into()))?;
        let mut vec = Vec::new();
        vec.try_reserve_exact(capacity)?;
        vec.resize(capacity, 0.0);
        let mut adj_matrix = Array2::from_shape_vec((num_nodes, num_nodes), vec)
            .map_err(|e| FastRPError::ShapeMismatch(e.to_string()))?;
        
        for (i, neighbors) in adj_list.into_iter().enumerate() {
            let degree = neighbors.len() as f64;
            let norm_factor = if degree > 0.0 { 1.0 / degree } else { 0.0 };
            for (target, _weight) in neighbors {
                adj_matrix[[i, target]] = norm_factor;
            }
        }
        Ok(Self { adj_matrix, num_nodes })
    }
}

/// Executes FastRP algorithm using traditional dense matrix multiply.
pub fn compute_fastrp_dense(
    dense: &DenseMatrix,
    dim: usize,
    iteration_weights: &[f64],
    seed: Option<u64>
) -> Result<Array2<f64>, FastRPError> {
    let mut h_curr = initialize_h0(dense.num_nodes, dim, seed)?;
    
    let capacity = dense.num_nodes.checked_mul(dim).ok_or_else(|| FastRPError::ShapeMismatch("Capacity overflow".into()))?;
    let mut vec_final = Vec::new();
    vec_final.try_reserve_exact(capacity)?;
    vec_final.resize(capacity, 0.0);
    let mut final_embeddings = Array2::from_shape_vec((dense.num_nodes, dim), vec_final)
        .map_err(|e| FastRPError::ShapeMismatch(e.to_string()))?;

    for &weight in iteration_weights {
        let h_next = dense.adj_matrix.dot(&h_curr);

        if weight != 0.0 {
            ndarray::Zip::from(final_embeddings.rows_mut())
                .and(h_next.rows())
                // dense dot paths run natively sequential inner
                .for_each(|mut final_row, next_row| {
                    for j in 0..dim {
                        final_row[j] += weight * next_row[j];
                    }
                });
        }
        h_curr = h_next;
    }

    Ok(final_embeddings)
}
