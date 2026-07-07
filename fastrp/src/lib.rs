//! FastRP-rs: A high-performance safe implementation of the Fast Random Projection algorithm.
//!
//! This library provides an efficient graph embedding generation mechanism.
//! Internally, it relies exclusively on a memory-efficient Compressed Sparse Row (CSR) matrix
//! representation designed to scale gracefully to massive networks.
//!
//! Original Paper: [Fast and Accurate Network Embeddings via Very Sparse Random Projection](https://arxiv.org/abs/1908.11512)
//!
//! ```bibtex
//! @inproceedings{chen2019fast,
//!   title = {Fast and accurate network embeddings via very sparse random projection},
//!   author = {Chen, Haochen and Sultan, Syed Fahad and Tian, Yingtao and Chen, Muhao and Skiena, Steven},
//!   booktitle = {International Conference on Information and Knowledge Management},
//!   pages = {399--408},
//!   year = {2019},
//!   doi = {10.1145/3357384.3357879},
//!   journal = {International Conference on Information and Knowledge Management},
//!   publisher = {ACM},
//! }
//! ```
//!
//! # Flexible Input Formats
//!
//! The library exposes multiple ingestion forms to suit your dataset:
//! 1. **Dense Matrices**: Convert small-to-medium graphs from a dense `ndarray::Array2`.
//! 2. **Adjacency Lists**: Input graphs represented as vectors of node-to-weight relationships.
//! 3. **Raw CSR Data**: Pass low-level arrays (row pointers, column indices, values) directly for maximum performance.
//! 4. **Pre-built CsrMatrix**: Utilize the native `CsrMatrix` object directly.
//!
//! # Example
//!
//! ```rust
//! use fastrp::FastRPBuilder;
//!
//! // Prepare your graph using an adjacency list
//! let adj_list = vec![
//!     vec![(1, 1.0), (2, 1.0)], // Node 0 is connected to Node 1 and Node 2
//!     vec![(0, 1.0)],           // Node 1 is connected to Node 0
//!     vec![(0, 1.0)],           // Node 2 is connected to Node 0
//! ];
//!
//! // Configure the FastRPBuilder and generate embeddings
//! let embeddings = FastRPBuilder::new(128)
//!     .with_weights(vec![0.0, 0.5, 1.0])
//!     .with_seed(42)
//!     .fit_adj_list(adj_list)
//!     .expect("Failed to compute embeddings");
//! ```

pub mod algorithm;
pub mod csr;
pub mod error;

pub use csr::CsrMatrix;
pub use error::FastRPError;

use ndarray::Array2;

/// Primary builder interface for structuring and running the Fast Random Projection Algorithm.
///
/// The `FastRPBuilder` uses the builder pattern to configure hyperparameters (such as
/// the embedding dimension, iteration weights, and random seed) before ultimately
/// executing the algorithm via one of the `fit_*` or `from_*` methods.
///
/// # Example
///
/// ```rust
/// use fastrp::FastRPBuilder;
///
/// let builder = FastRPBuilder::new(64)
///     .with_weights(vec![0.0, 1.0, 1.0])
///     .with_seed(12345);
/// ```
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

    /// Execute the FastRP algorithm using a pre-built Compressed Sparse Row (CSR) structure.
    ///
    /// This method is the core execution path of the library. All other input formats
    /// are internally converted to a `CsrMatrix` before being processed by this method.
    ///
    /// Embeddings are returned as `f32` for improved memory efficiency and SIMD throughput.
    pub fn fit_csr(&self, csr: &CsrMatrix) -> Result<Array2<f32>, FastRPError> {
        algorithm::compute_fastrp(csr, self.dim, &self.iteration_weights, self.seed)
    }

    /// Execute the FastRP algorithm from a dense adjacency matrix.
    ///
    /// Internally maps the dense `ndarray::Array2` matrix to a sparse CSR matrix
    /// before computation. This is convenient for small-to-medium graphs, but
    /// may be memory-intensive for large datasets.
    ///
    /// Embeddings are returned as `f32` for improved memory efficiency and SIMD throughput.
    pub fn fit_dense(&self, adj_matrix: &Array2<f64>) -> Result<Array2<f32>, FastRPError> {
        let csr = CsrMatrix::build_from_dense(adj_matrix)?;
        self.fit_csr(&csr)
    }

    /// Execute the FastRP algorithm from raw CSR components.
    ///
    /// This method allows you to directly pass the internal components of a
    /// Compressed Sparse Row matrix. It's the most efficient way to process
    /// data if it is already in CSR format externally.
    ///
    /// Embeddings are returned as `f32` for improved memory efficiency and SIMD throughput.
    pub fn from_csr(
        &self,
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
        num_nodes: usize,
    ) -> Result<Array2<f32>, FastRPError> {
        let csr = CsrMatrix::build_from_csr(row_ptrs, col_indices, values, num_nodes)?;
        self.fit_csr(&csr)
    }

    /// Convenience method to execute the algorithm from an adjacency list format.
    ///
    /// An adjacency list is represented as a `Vec` where each index corresponds
    /// to a node ID, and its value is a list of outgoing edges `(target_node, weight)`.
    /// This format is then internally converted into a memory-efficient CSR matrix.
    ///
    /// Embeddings are returned as `f32` for improved memory efficiency and SIMD throughput.
    pub fn fit_adj_list(
        &self,
        adj_list: Vec<Vec<(usize, f64)>>,
    ) -> Result<Array2<f32>, FastRPError> {
        let csr = CsrMatrix::build_from_adj_list(adj_list)?;
        self.fit_csr(&csr)
    }
}
