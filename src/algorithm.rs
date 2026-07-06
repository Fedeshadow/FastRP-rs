use ndarray::{Array2, Zip};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use crate::csr::CsrMatrix;
use crate::error::FastRPError;

/// Initializes the H0 matrix via very sparse random projection.
pub fn initialize_h0(num_nodes: usize, dim: usize, seed: Option<u64>) -> Result<Array2<f64>, FastRPError> {
    let s: f64 = 3.0;
    let val = s.sqrt();
    let prob_non_zero = 1.0 / (2.0 * s);
    
    let capacity = num_nodes.checked_mul(dim).ok_or_else(|| FastRPError::ShapeMismatch("Capacity overflow".into()))?;
    let mut vec = Vec::new();
    vec.try_reserve_exact(capacity)?;
    vec.resize(capacity, 0.0);
    let mut h0 = Array2::from_shape_vec((num_nodes, dim), vec).unwrap();
    
    match seed {
        Some(s_val) => {
            let mut rng = StdRng::seed_from_u64(s_val);
            fill_h0(&mut h0, &mut rng, prob_non_zero, val);
        },
        None => {
            let mut rng = rand::thread_rng();
            fill_h0(&mut h0, &mut rng, prob_non_zero, val);
        }
    }
    
    Ok(h0)
}

fn fill_h0<R: Rng>(h0: &mut Array2<f64>, rng: &mut R, prob_non_zero: f64, val: f64) {
    let (num_nodes, dim) = h0.dim();
    for i in 0..num_nodes {
        for j in 0..dim {
            let p: f64 = rng.r#gen();
            if p < prob_non_zero {
                h0[[i, j]] = val;
            } else if p < 2.0 * prob_non_zero {
                h0[[i, j]] = -val;
            }
        }
    }
}

/// Executes parallel SpMM for the FastRP algorithm using the CSR matrix.
pub fn compute_fastrp(
    csr: &CsrMatrix, 
    dim: usize, 
    iteration_weights: &[f64],
    seed: Option<u64>
) -> Result<Array2<f64>, FastRPError> {
    let mut h_curr = initialize_h0(csr.num_nodes, dim, seed)?;
    
    let capacity = csr.num_nodes.checked_mul(dim).ok_or_else(|| FastRPError::ShapeMismatch("Capacity overflow".into()))?;
    
    let mut vec_final = Vec::new();
    vec_final.try_reserve_exact(capacity)?;
    vec_final.resize(capacity, 0.0);
    let mut final_embeddings = Array2::from_shape_vec((csr.num_nodes, dim), vec_final).unwrap();

    let mut vec_next = Vec::new();
    vec_next.try_reserve_exact(capacity)?;
    vec_next.resize(capacity, 0.0);
    let mut h_next_storage = Array2::from_shape_vec((csr.num_nodes, dim), vec_next).unwrap();

    for &weight in iteration_weights {
        h_next_storage.fill(0.0);

        // 1. Parallelize across all rows (nodes)
        Zip::from(h_next_storage.rows_mut())
            .and(&csr.row_ptrs[..csr.num_nodes])
            .and(&csr.row_ptrs[1..])
            .par_for_each(|mut next_row, &start, &end| {
                
                // 2. Iterate through the node's neighbors
                for edge_idx in start..end {
                    let neighbor = csr.col_indices[edge_idx];
                    let edge_weight = csr.values[edge_idx];
                    
                    let neighbor_emb = h_curr.row(neighbor);
                    
                    // 3. Vectorized math: next_row += edge_weight * neighbor_emb
                    for j in 0..dim {
                        next_row[j] += edge_weight * neighbor_emb[j];
                    }
                }
            });

        // 4. Accumulate the weighted iteration into the final matrix
        if weight != 0.0 {
            Zip::from(final_embeddings.rows_mut())
                .and(h_next_storage.rows())
                .par_for_each(|mut final_row, next_row| {
                    for j in 0..dim {
                        final_row[j] += weight * next_row[j];
                    }
                });
        }

        // 5. Swap matrices alloc-free
        std::mem::swap(&mut h_curr, &mut h_next_storage); 
    }

    Ok(final_embeddings)
}
