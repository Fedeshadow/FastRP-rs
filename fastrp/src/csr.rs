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

    /// Computes `self * other`.
    ///
    /// # Assumptions
    /// Same as before: both matrices square, `self.num_nodes == other.num_nodes`
    /// (checked below, not assumed), and each matrix's internal CSR invariants
    /// hold as guaranteed by its constructor (spot-checked via `debug_assert!`).
    pub fn multiply(&self, other: &Self) -> Result<Self, FastRPError> {
        if self.num_nodes != other.num_nodes {
            return Err(FastRPError::ShapeMismatch(format!(
                "cannot multiply: self has {} nodes but other has {}",
                self.num_nodes, other.num_nodes
            )));
        }
        debug_assert_eq!(self.row_ptrs.len(), self.num_nodes + 1);
        debug_assert_eq!(other.row_ptrs.len(), other.num_nodes + 1);
        debug_assert!(self.col_indices.iter().all(|&c| c < self.num_nodes));
        debug_assert!(other.col_indices.iter().all(|&c| c < other.num_nodes));

        let num_nodes = self.num_nodes;
        if num_nodes == 0 {
            return Ok(Self {
                row_ptrs: vec![0],
                col_indices: Vec::new(),
                values: Vec::new(),
                num_nodes: 0,
            });
        }

        // Split rows into contiguous chunks, one per worker thread. Doing
        // this ourselves (rather than relying on rayon's default splitting,
        // or on `map_init`) means we control exactly how many workspaces get
        // allocated: `num_threads`, not one per task and not one per row.
        let num_threads = rayon::current_num_threads().max(1);
        let chunk_len = num_nodes.div_ceil(num_threads);

        let mut chunk_ranges: Vec<(usize, usize)> = Vec::new();
        chunk_ranges.try_reserve_exact(num_threads)?;
        let mut start = 0;
        while start < num_nodes {
            let end = (start + chunk_len).min(num_nodes);
            chunk_ranges.push((start, end));
            start = end;
        }

        // Allocate one workspace per chunk *here*, sequentially, before any
        // parallel work starts. This is the only place large (O(num_nodes))
        // allocations happen, and each one goes through `try_reserve_exact`,
        // so a failure returns an error instead of aborting the process —
        // unlike `map_init`, whose `init` closure has no `Result` path.
        //
        // `accum`/`marker` are Gustavson's-algorithm dense workspaces (see
        // prior comments): `marker[j] == i` means "accum[j] already holds a
        // partial dot product for row i," so no per-row clear is needed.
        let mut chunks: Vec<(usize, usize, Vec<f64>, Vec<usize>)> = Vec::new();
        chunks.try_reserve_exact(chunk_ranges.len())?;
        for (start, end) in chunk_ranges {
            let mut accum = Vec::new();
            accum.try_reserve_exact(num_nodes)?;
            accum.resize(num_nodes, 0.0);

            let mut marker = Vec::new();
            marker.try_reserve_exact(num_nodes)?;
            marker.resize(num_nodes, usize::MAX);

            chunks.push((start, end, accum, marker));
        }

        // Each chunk is processed by exactly one thread, owning its
        // workspace outright — no Mutex, no atomics, no cross-thread
        // contention at all.
        let per_chunk_results: Vec<Vec<(Vec<usize>, Vec<f64>)>> = chunks
            .into_par_iter()
            .map(|(start, end, mut accum, mut marker)| -> Result<Vec<(Vec<usize>, Vec<f64>)>, FastRPError> {
                let mut rows_out = Vec::new();
                rows_out.try_reserve_exact(end - start)?;
                let mut touched: Vec<usize> = Vec::new();

                for i in start..end {
                    touched.clear();

                    for a_idx in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                        let k = self.col_indices[a_idx];
                        let a_val = self.values[a_idx];

                        for b_idx in other.row_ptrs[k]..other.row_ptrs[k + 1] {
                            let j = other.col_indices[b_idx];
                            let contribution = a_val * other.values[b_idx];

                            if marker[j] == i {
                                accum[j] += contribution;
                            } else {
                                marker[j] = i;
                                accum[j] = contribution;
                                touched.push(j);
                            }
                        }
                    }

                    touched.sort_unstable();

                    let mut cols = Vec::new();
                    cols.try_reserve_exact(touched.len())?;
                    let mut vals = Vec::new();
                    vals.try_reserve_exact(touched.len())?;
                    for &j in touched.iter() {
                        cols.push(j);
                        vals.push(accum[j]);
                    }
                    rows_out.push((cols, vals));
                }

                Ok(rows_out)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Sequential stitch, with buffers reserved exactly once up front.
        let total_nnz: usize = per_chunk_results
            .iter()
            .flat_map(|rows| rows.iter())
            .map(|(cols, _)| cols.len())
            .sum();

        let mut new_row_ptrs = Vec::new();
        new_row_ptrs.try_reserve_exact(num_nodes + 1)?;
        let mut new_col_indices = Vec::new();
        new_col_indices.try_reserve_exact(total_nnz)?;
        let mut new_values = Vec::new();
        new_values.try_reserve_exact(total_nnz)?;

        new_row_ptrs.push(0);
        for (cols, vals) in per_chunk_results.into_iter().flatten() {
            new_col_indices.extend(cols);
            new_values.extend(vals);
            new_row_ptrs.push(new_col_indices.len());
        }

        Ok(Self {
            row_ptrs: new_row_ptrs,
            col_indices: new_col_indices,
            values: new_values,
            num_nodes,
        })
    }

    // TODO: add a method that performs let result = m.multiply(&m.transpose());
}
