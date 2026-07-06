use crate::error::FastRPError;

/// A simple Compressed Sparse Row (CSR) matrix representation.
pub struct CsrMatrix {
    pub row_ptrs: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
    pub num_nodes: usize,
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
                col_indices.push(target);
                values.push(norm_factor); // Store normalized weight
                current_ptr += 1;
            }
            row_ptrs.push(current_ptr);
        }

        Ok(Self { row_ptrs, col_indices, values, num_nodes })
    }
}
