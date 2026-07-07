use ndarray::Array2;
use crate::error::FastRPError;

/// A simple Compressed Sparse Row (CSR) matrix representation.
pub struct CsrMatrix {
    pub(crate) row_ptrs: Vec<usize>,
    pub(crate) col_indices: Vec<usize>,
    pub(crate) values: Vec<f64>,
    pub(crate) num_nodes: usize,
}

impl CsrMatrix {
    /// Builds a row-normalized CSR matrix from a node adjacency list.
    /// The input is expected to be a `Vec<Vec<(target_node, weight)>>`, 
    /// where the index is the source node ID.
    pub fn build_from_adj_list(adj_list: Vec<Vec<(usize, f64)>>) -> Result<Self, FastRPError> {
        let num_nodes = adj_list.len();
        
        let mut row_ptrs = Vec::new();
        row_ptrs.try_reserve_exact(num_nodes + 1)?;
        
        let total_edges: usize = adj_list.iter().map(|n| n.len()).sum();
        
        let mut col_indices = Vec::new();
        col_indices.try_reserve_exact(total_edges)?;
        
        let mut values = Vec::new();
        values.try_reserve_exact(total_edges)?;
        
        let mut current_ptr = 0;
        row_ptrs.push(current_ptr);

        for neighbors in adj_list {
            // Degree normalization: 1.0 / degree
            // For weighted graphs, divide by the sum of out-weights
            let degree = neighbors.len() as f64;
            let norm_factor = if degree > 0.0 { 1.0 / degree } else { 0.0 };

            for (target, _weight) in neighbors {
                if target >= num_nodes {
                    return Err(FastRPError::ShapeMismatch(format!(
                        "Target node ID {} is out of bounds for graph with {} nodes",
                        target, num_nodes
                    )));
                }

                col_indices.push(target);
                values.push(norm_factor); // Store normalized weight
                current_ptr += 1;
            }
            row_ptrs.push(current_ptr);
        }

        Ok(Self { row_ptrs, col_indices, values, num_nodes })
    }

    /// Builds a CSR matrix from a dense adjacency matrix.
    /// Non-zero entries are treated as edges with their respective values as weights.
    pub fn build_from_dense(adj_matrix: &Array2<f64>) -> Result<Self, FastRPError> {
        let (num_nodes, cols) = adj_matrix.dim();
        if num_nodes != cols {
            return Err(FastRPError::ShapeMismatch("Adjacency matrix must be square".into()));
        }

        let mut row_ptrs = Vec::new();
        row_ptrs.try_reserve_exact(num_nodes + 1)?;

        // Count non-zero elements to pre-allocate correctly and avoid implicit OOM panics from Vec::push
        let mut nnz = 0;
        for &val in adj_matrix.iter() {
            if val != 0.0 {
                nnz += 1;
            }
        }

        let mut col_indices = Vec::new();
        col_indices.try_reserve_exact(nnz)?;
        
        let mut values = Vec::new();
        values.try_reserve_exact(nnz)?;
        
        let mut current_ptr = 0;
        row_ptrs.push(current_ptr);

        for row in adj_matrix.rows() {
            for (target, &val) in row.into_iter().enumerate() {
                if val != 0.0 {
                    col_indices.push(target);
                    values.push(val); // Keep original matrix value
                    current_ptr += 1;
                }
            }
            row_ptrs.push(current_ptr);
        }

        Ok(Self { row_ptrs, col_indices, values, num_nodes })
    }

    /// Builds a CSR matrix directly from raw CSR components.
    pub fn build_from_csr(
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        num_nodes: usize,
    ) -> Result<Self, FastRPError> {
        if row_ptrs.len() != num_nodes + 1 {
            return Err(FastRPError::ShapeMismatch(format!(
                "row_ptrs length {} must be num_nodes + 1 ({})",
                row_ptrs.len(),
                num_nodes + 1
            )));
        }
        if col_indices.len() != values.len() {
            return Err(FastRPError::ShapeMismatch(
                "col_indices and values must have the same length".into(),
            ));
        }

        Ok(Self {
            row_ptrs,
            col_indices,
            values,
            num_nodes,
        })
    }
}
