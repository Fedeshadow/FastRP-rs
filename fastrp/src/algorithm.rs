use ndarray::{Array2, Axis, Zip};
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use crate::csr::CsrMatrix;
use crate::error::FastRPError;

/// Initializes the H0 matrix via very sparse random projection.
///
/// Uses parallel row-level initialization with a fast, non-cryptographic RNG
/// (`Xoshiro256PlusPlus`). Each row derives a deterministic per-row seed from
/// the global seed XOR'd with the row index, ensuring reproducibility
/// regardless of thread scheduling.
pub fn initialize_h0(num_nodes: usize, dim: usize, seed: Option<u64>) -> Result<Array2<f32>, FastRPError> {
    let s: f32 = 3.0;
    let val = s.sqrt();
    let prob_non_zero: f32 = 1.0 / (2.0 * s);

    let capacity = num_nodes.checked_mul(dim).ok_or_else(|| FastRPError::ShapeMismatch("Capacity overflow".into()))?;
    let mut vec = Vec::new();
    vec.try_reserve_exact(capacity)?;
    vec.resize(capacity, 0.0f32);
    let mut h0 = Array2::from_shape_vec((num_nodes, dim), vec)
        .map_err(|e| FastRPError::ShapeMismatch(e.to_string()))?;

    let base_seed = seed.unwrap_or_else(|| {
        let mut rng = rand::thread_rng();
        rng.r#gen()
    });

    // Parallel H0 initialization: each row gets a deterministic per-row seed
    h0.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            // Derive a per-row seed by mixing the base seed with the row index.
            // Using wrapping_add avoids collisions better than XOR for sequential indices.
            let row_seed = base_seed.wrapping_add(i as u64);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(row_seed);

            for j in 0..dim {
                let p: f32 = rng.r#gen();
                if p < prob_non_zero {
                    row[j] = val;
                } else if p < 2.0 * prob_non_zero {
                    row[j] = -val;
                }
            }
        });

    Ok(h0)
}

/// Executes parallel SpMM for the FastRP algorithm using the CSR matrix.
///
/// Embeddings are computed in `f32` for reduced memory usage and improved SIMD throughput.
/// CSR edge weights (f64) are cast to `f32` at the computation boundary.
pub fn compute_fastrp(
    csr: &CsrMatrix, 
    dim: usize, 
    iteration_weights: &[f64],
    seed: Option<u64>
) -> Result<Array2<f32>, FastRPError> {
    let mut h_curr = initialize_h0(csr.num_nodes, dim, seed)?;
    
    let capacity = csr.num_nodes.checked_mul(dim).ok_or_else(|| FastRPError::ShapeMismatch("Capacity overflow".into()))?;
    
    let mut vec_final = Vec::new();
    vec_final.try_reserve_exact(capacity)?;
    vec_final.resize(capacity, 0.0f32);
    let mut final_embeddings = Array2::from_shape_vec((csr.num_nodes, dim), vec_final)
        .map_err(|e| FastRPError::ShapeMismatch(e.to_string()))?;

    let mut vec_next = Vec::new();
    vec_next.try_reserve_exact(capacity)?;
    vec_next.resize(capacity, 0.0f32);
    let mut h_next_storage = Array2::from_shape_vec((csr.num_nodes, dim), vec_next)
        .map_err(|e| FastRPError::ShapeMismatch(e.to_string()))?;

    for &weight in iteration_weights {
        h_next_storage.fill(0.0);

        // 1. Parallelize across all rows (nodes)
        Zip::from(h_next_storage.rows_mut())
            .and(&csr.row_ptrs[..csr.num_nodes])
            .and(&csr.row_ptrs[1..])
            .par_for_each(|mut next_row, &start, &end| {
                // 2. Row-normalize: divide by out-degree to form the transition matrix.
                //    This keeps embedding magnitudes stable across iterations and matches
                //    the Ā (row-stochastic) matrix defined in the FastRP paper.
                let degree = (end - start) as f32;
                let norm = if degree > 0.0 { 1.0 / degree } else { 0.0 };

                // 3. Iterate through the node's neighbors
                for edge_idx in start..end {
                    let neighbor = csr.col_indices[edge_idx];
                    let edge_weight = csr.values[edge_idx] as f32;
                    
                    let neighbor_emb = h_curr.row(neighbor);
                    
                    // 4. BLAS-1 style: next_row += (edge_weight / degree) * neighbor_emb
                    next_row.scaled_add(edge_weight * norm, &neighbor_emb);
                }
            });

        // 4. Accumulate the weighted iteration into the final matrix
        if weight != 0.0 {
            let w = weight as f32;
            Zip::from(final_embeddings.rows_mut())
                .and(h_next_storage.rows())
                .par_for_each(|mut final_row, next_row| {
                    final_row.scaled_add(w, &next_row);
                });
        }

        // 5. Swap matrices alloc-free
        std::mem::swap(&mut h_curr, &mut h_next_storage); 
    }

    Ok(final_embeddings)
}
