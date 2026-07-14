use crate::error::FastRPError;
use ndarray::Array2;
use rayon::prelude::*;

const PARALLEL_HISTOGRAM_NNZ_THRESHOLD: usize = 100_000;

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

        Ok(Self {
            row_ptrs,
            col_indices,
            values,
            num_nodes,
        })
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
        // row_ptrs.last() is safe here: we verified len == num_nodes + 1 >= 1
        if *row_ptrs.last().unwrap() != col_indices.len() {
            return Err(FastRPError::ShapeMismatch(format!(
                "row_ptrs last element ({}) must equal col_indices length ({})",
                row_ptrs.last().unwrap(),
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

        Ok(Self {
            row_ptrs,
            col_indices,
            values,
            num_nodes,
        })
    }

    /// Returns the transpose of this matrix.
    ///
    /// # Assumptions
    /// - The matrix is **square**: `num_nodes` is used as both row and
    ///   column count, so every entry of `col_indices` is assumed `< num_nodes`.
    /// - `row_ptrs.len() == num_nodes + 1`, `row_ptrs` is non-decreasing, and
    ///   `row_ptrs.last() == Some(col_indices.len()) == Some(values.len())`.
    ///
    /// Both are guaranteed by `build_from_adj_list`, `build_from_dense`, and
    /// `build_from_csr` — the only sanctioned ways to construct `Self`. This
    /// method trusts those invariants rather than re-validating them (see the
    /// `debug_assert!`s below, which catch violations in debug builds only).
    /// If `CsrMatrix` fields are ever constructed directly elsewhere in the
    /// crate, that's the place to fix, not here.
    ///
    /// # Errors
    /// Returns `Err(FastRPError::...)` if any output buffer allocation fails,
    /// instead of aborting the process (as `vec![0; n]` would on OOM).
    pub fn transpose(&self) -> Result<Self, FastRPError> {
        debug_assert_eq!(self.row_ptrs.len(), self.num_nodes + 1);
        debug_assert_eq!(self.row_ptrs.last().copied(), Some(self.values.len()));
        debug_assert_eq!(self.col_indices.len(), self.values.len());
        debug_assert!(self.col_indices.iter().all(|&c| c < self.num_nodes));

        let n = self.num_nodes;
        let nnz = self.values.len(); // Non zero elements 

        // `counts` plays three roles in sequence: histogram, prefix-sum
        // (= row_ptrs), then scatter cursor.
        let mut counts: Vec<usize> = Vec::new();
        counts.try_reserve_exact(n + 1)?;
        counts.resize(n + 1, 0);

        // 1. Histogram: nnz per column of the original == row length in the
        //    transpose. Parallelized above a size threshold; below it, the
        //    per-thread buffer + reduce merge isn't worth it.
        if nnz >= PARALLEL_HISTOGRAM_NNZ_THRESHOLD {
            counts = self
                .col_indices
                .par_iter()
                .fold(
                    || vec![0usize; n + 1],
                    |mut local, &col| {
                        local[col + 1] += 1;
                        local
                    },
                )
                .reduce(
                    || vec![0usize; n + 1],
                    |mut a, b| {
                        for i in 0..=n {
                            a[i] += b[i];
                        }
                        a
                    },
                );
        } else {
            for &col in &self.col_indices {
                counts[col + 1] += 1;
            }
        }

        // 2. Prefix sum -> counts[i] becomes the start offset of row i.
        for i in 0..n {
            counts[i + 1] += counts[i];
        }

        // 3. Scatter, using `counts` itself as the moving insertion cursor.
        //    Serial: each iteration depends on the previous write position
        //    for that column, so this isn't trivially parallelizable without
        //    atomics (see note below).
        let mut new_col_indices: Vec<usize> = Vec::new();
        new_col_indices.try_reserve_exact(nnz)?;
        new_col_indices.resize(nnz, 0);

        let mut new_values: Vec<f64> = Vec::new();
        new_values.try_reserve_exact(nnz)?;
        new_values.resize(nnz, 0.0);

        for row in 0..n {
            for idx in self.row_ptrs[row]..self.row_ptrs[row + 1] {
                let col = self.col_indices[idx];
                let dest = counts[col];
                new_col_indices[dest] = row;
                new_values[dest] = self.values[idx];
                counts[col] += 1;
            }
        }

        // 4. Undo the shift to recover proper row_ptrs (counts[i] currently
        //    holds what should be counts[i+1]).
        let mut last = 0;
        for c in counts.iter_mut() {
            let prev = *c;
            *c = last;
            last = prev;
        }

        Ok(Self {
            row_ptrs: counts,
            col_indices: new_col_indices,
            values: new_values,
            num_nodes: n,
        })
    }

    pub fn add(&self, other: &CsrMatrix) -> Result<CsrMatrix, FastRPError> {
        if self.num_nodes != other.num_nodes {
            return Err(FastRPError::ShapeMismatch(format!(
                "Dimension mismatch: {} != {}",
                self.num_nodes, other.num_nodes
            )));
        }

        let rows = self.num_nodes;

        // Helper closure to map indexing failures to our error type
        let err_oob = || {
            FastRPError::ShapeMismatch(
                "Matrix structure corrupted: index out of bounds".to_string(),
            )
        };

        // ---------------------------------------------------------
        // PASS 1: Symbolic Phase (Parallel)
        // ---------------------------------------------------------

        let mut row_nnz = Vec::new();
        row_nnz.try_reserve_exact(rows)?;
        for _ in 0..rows {
            row_nnz.push(0_usize);
        }

        row_nnz.par_iter_mut().enumerate().try_for_each(
            |(i, nnz_count)| -> Result<(), FastRPError> {
                let start_a = *self.row_ptrs.get(i).ok_or_else(err_oob)?;
                let end_a = *self.row_ptrs.get(i + 1).ok_or_else(err_oob)?;
                let start_b = *other.row_ptrs.get(i).ok_or_else(err_oob)?;
                let end_b = *other.row_ptrs.get(i + 1).ok_or_else(err_oob)?;

                let cols_a = self.col_indices.get(start_a..end_a).ok_or_else(err_oob)?;
                let cols_b = other.col_indices.get(start_b..end_b).ok_or_else(err_oob)?;
                let vals_a = self.values.get(start_a..end_a).ok_or_else(err_oob)?;
                let vals_b = other.values.get(start_b..end_b).ok_or_else(err_oob)?;

                let mut count = 0;
                let mut ptr_a = 0;
                let mut ptr_b = 0;

                while ptr_a < cols_a.len() && ptr_b < cols_b.len() {
                    let col_a = *cols_a.get(ptr_a).ok_or_else(err_oob)?;
                    let col_b = *cols_b.get(ptr_b).ok_or_else(err_oob)?;

                    if col_a < col_b {
                        count += 1;
                        ptr_a += 1;
                    } else if col_b < col_a {
                        count += 1;
                        ptr_b += 1;
                    } else {
                        let val_a = *vals_a.get(ptr_a).ok_or_else(err_oob)?;
                        let val_b = *vals_b.get(ptr_b).ok_or_else(err_oob)?;
                        if val_a + val_b != 0.0 {
                            count += 1;
                        }
                        ptr_a += 1;
                        ptr_b += 1;
                    }
                }

                count += cols_a.len() - ptr_a;
                count += cols_b.len() - ptr_b;
                *nnz_count = count;

                Ok(())
            },
        )?;

        // ---------------------------------------------------------
        // PREFIX SUM: Build row_ptrs for Matrix C (Sequential)
        // ---------------------------------------------------------

        let mut row_ptrs_c = Vec::new();
        row_ptrs_c.try_reserve_exact(rows + 1)?;
        row_ptrs_c.push(0);

        let mut total_nnz = 0;
        for &nnz in &row_nnz {
            total_nnz += nnz;
            row_ptrs_c.push(total_nnz);
        }

        // ---------------------------------------------------------
        // PASS 2: Numeric Phase (Parallel)
        // ---------------------------------------------------------

        let mut col_indices_c = Vec::new();
        col_indices_c.try_reserve_exact(total_nnz)?;
        for _ in 0..total_nnz {
            col_indices_c.push(0_usize);
        }

        let mut values_c = Vec::new();
        values_c.try_reserve_exact(total_nnz)?;
        for _ in 0..total_nnz {
            values_c.push(0.0_f64);
        }

        let mut col_slices = Vec::new();
        col_slices.try_reserve_exact(rows)?;
        let mut val_slices = Vec::new();
        val_slices.try_reserve_exact(rows)?;

        let mut rem_cols = &mut col_indices_c[..];
        let mut rem_vals = &mut values_c[..];

        for &nnz in &row_nnz {
            // Explicit check prevents split_at_mut from ever panicking
            if nnz > rem_cols.len() || nnz > rem_vals.len() {
                return Err(FastRPError::ShapeMismatch(
                    "Internal total nnz mismatch".to_string(),
                ));
            }
            let (c_chunk, c_rest) = rem_cols.split_at_mut(nnz);
            let (v_chunk, v_rest) = rem_vals.split_at_mut(nnz);

            col_slices.push(c_chunk);
            val_slices.push(v_chunk);

            rem_cols = c_rest;
            rem_vals = v_rest;
        }

        col_slices
            .into_par_iter()
            .zip(val_slices.into_par_iter())
            .enumerate()
            .try_for_each(|(i, (col_out, val_out))| -> Result<(), FastRPError> {
                let start_a = *self.row_ptrs.get(i).ok_or_else(err_oob)?;
                let end_a = *self.row_ptrs.get(i + 1).ok_or_else(err_oob)?;
                let start_b = *other.row_ptrs.get(i).ok_or_else(err_oob)?;
                let end_b = *other.row_ptrs.get(i + 1).ok_or_else(err_oob)?;

                let cols_a = self.col_indices.get(start_a..end_a).ok_or_else(err_oob)?;
                let cols_b = other.col_indices.get(start_b..end_b).ok_or_else(err_oob)?;
                let vals_a = self.values.get(start_a..end_a).ok_or_else(err_oob)?;
                let vals_b = other.values.get(start_b..end_b).ok_or_else(err_oob)?;

                let mut ptr_a = 0;
                let mut ptr_b = 0;
                let mut out_idx = 0;

                while ptr_a < cols_a.len() && ptr_b < cols_b.len() {
                    let col_a = *cols_a.get(ptr_a).ok_or_else(err_oob)?;
                    let col_b = *cols_b.get(ptr_b).ok_or_else(err_oob)?;

                    if col_a < col_b {
                        *col_out.get_mut(out_idx).ok_or_else(err_oob)? = col_a;
                        *val_out.get_mut(out_idx).ok_or_else(err_oob)? =
                            *vals_a.get(ptr_a).ok_or_else(err_oob)?;
                        ptr_a += 1;
                        out_idx += 1;
                    } else if col_b < col_a {
                        *col_out.get_mut(out_idx).ok_or_else(err_oob)? = col_b;
                        *val_out.get_mut(out_idx).ok_or_else(err_oob)? =
                            *vals_b.get(ptr_b).ok_or_else(err_oob)?;
                        ptr_b += 1;
                        out_idx += 1;
                    } else {
                        let val_a = *vals_a.get(ptr_a).ok_or_else(err_oob)?;
                        let val_b = *vals_b.get(ptr_b).ok_or_else(err_oob)?;
                        let sum = val_a + val_b;

                        if sum != 0.0 {
                            *col_out.get_mut(out_idx).ok_or_else(err_oob)? = col_a;
                            *val_out.get_mut(out_idx).ok_or_else(err_oob)? = sum;
                            out_idx += 1;
                        }
                        ptr_a += 1;
                        ptr_b += 1;
                    }
                }

                while ptr_a < cols_a.len() {
                    *col_out.get_mut(out_idx).ok_or_else(err_oob)? =
                        *cols_a.get(ptr_a).ok_or_else(err_oob)?;
                    *val_out.get_mut(out_idx).ok_or_else(err_oob)? =
                        *vals_a.get(ptr_a).ok_or_else(err_oob)?;
                    ptr_a += 1;
                    out_idx += 1;
                }

                while ptr_b < cols_b.len() {
                    *col_out.get_mut(out_idx).ok_or_else(err_oob)? =
                        *cols_b.get(ptr_b).ok_or_else(err_oob)?;
                    *val_out.get_mut(out_idx).ok_or_else(err_oob)? =
                        *vals_b.get(ptr_b).ok_or_else(err_oob)?;
                    ptr_b += 1;
                    out_idx += 1;
                }

                Ok(())
            })?;

        Ok(CsrMatrix {
            row_ptrs: row_ptrs_c,
            col_indices: col_indices_c,
            values: values_c,
            num_nodes: self.num_nodes,
        })
    }
}
