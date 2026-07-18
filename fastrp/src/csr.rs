use crate::error::FastRPError;
use ndarray::Array2;
use sprs::{CsMat, CsMatView, errors::StructureError};

/// A simple Compressed Sparse Row (CSR) matrix representation.
#[derive(Debug)]
pub struct CsrMatrix {
    pub(crate) row_ptrs: Vec<usize>,
    pub(crate) col_indices: Vec<usize>,
    pub(crate) values: Vec<f64>,
    pub(crate) num_nodes: usize,
}

impl CsrMatrix {
    /// Returns a slice of the row pointers of the matrix.
    pub fn row_ptrs(&self) -> &[usize] {
        &self.row_ptrs
    }

    /// Returns a slice of the column indices of the matrix.
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Returns a slice of the values of the matrix.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Returns the number of nodes (rows/columns) in the matrix.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Creates a zero-copy borrowed view of the sparse matrix.
    /// This verifies structure constraints without allocating memory.
    pub fn view(&self) -> Result<CsMatView<'_, f64>, StructureError> {
        let shape = (self.num_nodes, self.num_nodes);

        CsMatView::try_new(shape, &self.row_ptrs, &self.col_indices, &self.values)
            .map_err(|(_, _, _, err)| err)
    }

    /// Builds a CSR matrix from a node adjacency list.
    /// The input is expected to be a `Vec<Vec<(target_node, weight)>>`,
    /// where the index is the source node ID.
    ///
    /// Raw edge weights are stored as-is; row-normalization is applied
    /// during the algorithm's SpMM step, not here.
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
            for (target, weight) in neighbors {
                if target >= num_nodes {
                    return Err(FastRPError::ShapeMismatch(format!(
                        "Target node ID {} is out of bounds for graph with {} nodes",
                        target, num_nodes
                    )));
                }

                col_indices.push(target);
                values.push(weight); // Store original weight
                current_ptr += 1;
            }
            row_ptrs.push(current_ptr);
        }

        let mut matrix = Self {
            row_ptrs,
            col_indices,
            values,
            num_nodes,
        };
        matrix.sort_indices();
        Ok(matrix)
    }

    /// Builds a CSR matrix from a dense adjacency matrix.
    /// Non-zero entries are treated as edges with their respective values as weights.
    pub fn build_from_dense(adj_matrix: &Array2<f64>) -> Result<Self, FastRPError> {
        let (num_nodes, cols) = adj_matrix.dim();
        if num_nodes != cols {
            return Err(FastRPError::ShapeMismatch(
                "Adjacency matrix must be square".into(),
            ));
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

        Ok(Self {
            row_ptrs,
            col_indices,
            values,
            num_nodes,
        })
    }

    /// Builds a CSR matrix directly from raw CSR components.
    ///
    /// This method validates structural invariants to guarantee that downstream
    /// computation cannot panic due to out-of-bounds indexing:
    /// - `row_ptrs` must have exactly `num_nodes + 1` elements.
    /// - `row_ptrs[0]` must be `0`.
    /// - `row_ptrs` must be monotonically non-decreasing.
    /// - The last element of `row_ptrs` must equal `col_indices.len()`.
    /// - `col_indices` and `values` must have the same length.
    /// - Every value in `col_indices` must be less than `num_nodes`.
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
        if row_ptrs[0] != 0 {
            return Err(FastRPError::ShapeMismatch(format!(
                "row_ptrs[0] must be 0, got {}",
                row_ptrs[0]
            )));
        }
        let last_ptr = *row_ptrs.last().ok_or_else(|| {
            FastRPError::ShapeMismatch("row_ptrs cannot be empty".into())
        })?;
        if last_ptr != col_indices.len() {
            return Err(FastRPError::ShapeMismatch(format!(
                "row_ptrs last element ({}) must equal col_indices length ({})",
                last_ptr,
                col_indices.len()
            )));
        }
        for window in row_ptrs.windows(2) {
            if window[0] > window[1] {
                return Err(FastRPError::ShapeMismatch(format!(
                    "row_ptrs must be monotonically non-decreasing, found {} followed by {}",
                    window[0], window[1]
                )));
            }
        }
        for (i, &col) in col_indices.iter().enumerate() {
            if col >= num_nodes {
                return Err(FastRPError::ShapeMismatch(format!(
                    "col_indices[{}] = {} is out of bounds for graph with {} nodes",
                    i, col, num_nodes
                )));
            }
        }

        let mut matrix = Self {
            row_ptrs,
            col_indices,
            values,
            num_nodes,
        };
        matrix.sort_indices();
        Ok(matrix)
    }

    /// Returns a new undirected version (M + M^T) of this matrix using `sprs`.
    ///
    /// # Errors
    /// Returns `Err(FastRPError::...)` if matrix structure is invalid during addition.
    pub fn make_undirected(&self) -> Result<CsrMatrix, FastRPError> {
        // 1. Create a lightweight reference view (Zero allocations)
        let a_view = self
            .view()
            .map_err(|e| FastRPError::ShapeMismatch(format!("Invalid CSR structure: {}", e)))?;

        // 2. Transpose the view
        // (Creates A^T structure; index allocations are structurally unavoidable here)
        let a_t = a_view.transpose_view();

        // 3. Compute A + A^T via reference addition
        // The underlying data values are only read from `self` here
        let res_mat: CsMat<f64> = &a_view + &a_t;

        // 4. Dismantle the newly produced matrix into your custom struct layout
        let num_nodes = res_mat.rows();
        let (row_ptrs, col_indices, values) = res_mat.into_raw_storage();

        Ok(CsrMatrix {
            row_ptrs,
            col_indices,
            values,
            num_nodes,
        })
    }

    /// Sorts the column indices in each row and sums the weights of duplicate edges.
    /// This ensures the matrix strictly adheres to CSR structural invariants required
    /// by downstream algorithms and external crates (like `sprs`).
    pub fn sort_indices(&mut self) {
        let mut needs_sort = false;
        for row in 0..self.num_nodes {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            if end > start {
                for i in start + 1..end {
                    if self.col_indices[i - 1] >= self.col_indices[i] {
                        needs_sort = true;
                        break;
                    }
                }
            }
            if needs_sort {
                break;
            }
        }

        if !needs_sort {
            return;
        }

        let mut write_idx = 0;
        let mut new_row_ptrs = Vec::with_capacity(self.num_nodes + 1);
        new_row_ptrs.push(0);

        let mut row_edges = Vec::new();

        for row in 0..self.num_nodes {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            row_edges.clear();
            for i in start..end {
                row_edges.push((self.col_indices[i], self.values[i]));
            }

            row_edges.sort_unstable_by(|a, b| a.0.cmp(&b.0));

            let mut last_col = None;
            for (col, val) in row_edges.drain(..) {
                if Some(col) == last_col {
                    self.values[write_idx - 1] += val;
                } else {
                    self.col_indices[write_idx] = col;
                    self.values[write_idx] = val;
                    write_idx += 1;
                    last_col = Some(col);
                }
            }
            new_row_ptrs.push(write_idx);
        }

        self.row_ptrs = new_row_ptrs;
        self.col_indices.truncate(write_idx);
        self.values.truncate(write_idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_undirected_wrong_path() {
        // Create an invalid CsrMatrix by bypassing the normal constructors
        let invalid_matrix = CsrMatrix {
            row_ptrs: vec![0, 1, 3], // Should be length 4 for 3 nodes
            col_indices: vec![1, 2],
            values: vec![1.0, 2.0],
            num_nodes: 3,
        };

        // make_undirected should fail because view() fails
        let result = invalid_matrix.make_undirected();
        assert!(result.is_err());
        match result.unwrap_err() {
            FastRPError::ShapeMismatch(msg) => {
                assert!(msg.contains("Invalid CSR structure"));
            }
            _ => panic!("Expected ShapeMismatch error"),
        }
    }
}
