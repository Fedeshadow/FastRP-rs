//! FastRP-rs: A high-performance implementation of the Fast Random Projection algorithm.
//! 
//! This library provides two execution paths:
//! 1. A memory-efficient Compressed Sparse Row (CSR) matrix representation designed for large graphs.
//! 2. A generalized dense execution path for smaller graphs utilizing standard array multiplication.

pub mod algorithm;
pub mod csr;
pub mod dense;
pub mod error;

pub use csr::CsrMatrix;
pub use dense::DenseMatrix;
pub use error::FastRPError;

use ndarray::Array2;

/// Primary builder interface for structuring and running the Fast Random Projection Algorithm.
pub struct FastRPBuilder {
    dim: usize,
    iteration_weights: Vec<f64>,
    seed: Option<u64>,
}

impl FastRPBuilder {
    /// Create a new builder with the target embedding dimension
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            iteration_weights: vec![0.0, 1.0, 1.0], // Default alpha weights per the original paper recommendations
            seed: None,
        }
    }

    /// Set customized iteration weights (e.g., `[0.0, 0.5, 1.0, 1.0]`) which dictates the scale applied from each multi-hop step
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.iteration_weights = weights;
        self
    }
    
    /// Pass a fixed deterministic seed to RNG generation for replicable behavior across runs
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Execute the FastRP algorithm using the memory-efficient Compressed Sparse Row (CSR) structure.
    pub fn fit_csr(&self, adj_list: Vec<Vec<(usize, f64)>>) -> Result<Array2<f64>, FastRPError> {
        let csr = CsrMatrix::build_from_adj_list(adj_list)?;
        algorithm::compute_fastrp(&csr, self.dim, &self.iteration_weights, self.seed)
    }

    /// Execute the FastRP algorithm utilizing a naive Dense Matrix. Prefer `fit_csr` for graphs scaling beyond a thousand nodes.
    pub fn fit_dense(&self, num_nodes: usize, adj_list: Vec<Vec<(usize, f64)>>) -> Result<Array2<f64>, FastRPError> {
        let dense = DenseMatrix::build_from_adj_list(num_nodes, adj_list)?;
        dense::compute_fastrp_dense(&dense, self.dim, &self.iteration_weights, self.seed)
    }
}
