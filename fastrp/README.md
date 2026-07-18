# FastRP-rs

A high-performance safe implementation of the **Fast Random Projection (FastRP)** algorithm in Rust.

`fastrp` provides an efficient graph embedding generation mechanism. Internally, it relies exclusively on a memory-efficient Compressed Sparse Row (CSR) matrix representation designed to scale gracefully to massive networks.

[![Crates.io](https://img.shields.io/crates/v/fastrp.svg)](https://crates.io/crates/fastrp)
[![Documentation](https://docs.rs/fastrp/badge.svg)](https://docs.rs/fastrp)
[![License](https://img.shields.io/crates/l/fastrp.svg)](#license)

Original Paper: [Fast and Accurate Network Embeddings via Very Sparse Random Projection](https://arxiv.org/abs/1908.11512)

> **Note**: This is an independent implementation and is not the official implementation of the paper's authors.

---

## Installation

Add `fastrp` to your `Cargo.toml`:

```toml
[dependencies]
fastrp = "0.1.0"
```

---

## Usage Examples

`fastrp` exposes multiple ingestion forms to suit your dataset:

### 1. Adjacency List (Convenient for general graph structures)

```rust
use fastrp::FastRPBuilder;

// Represent the graph as a list of outgoing edges: (target_node_id, weight)
let adj_list = vec![
    vec![(1, 1.0), (2, 1.0)], // Node 0 connects to Node 1 and Node 2
    vec![(0, 1.0)],           // Node 1 connects to Node 0
    vec![(0, 1.0)],           // Node 2 connects to Node 0
];

// Configure the builder and generate embeddings
let embeddings = FastRPBuilder::new(128)
    .with_weights(vec![0.0, 0.5, 1.0])
    .with_seed(42)
    .fit_adj_list(adj_list)
    .expect("Failed to compute embeddings");

assert_eq!(embeddings.nrows(), 3);
assert_eq!(embeddings.ncols(), 128);
```

### 2. Dense Adjacency Matrix (Convenient for small-to-medium datasets)

```rust
use fastrp::FastRPBuilder;
use ndarray::array;

// Represent the graph as a dense 2D adjacency matrix
let adj_matrix = array![
    [0.0, 1.0, 1.0], // Node 0 connects to Node 1 and Node 2
    [1.0, 0.0, 0.0], // Node 1 connects to Node 0
    [1.0, 0.0, 0.0], // Node 2 connects to Node 0
];

// Compute embeddings directly from the dense matrix
let embeddings = FastRPBuilder::new(128)
    .with_weights(vec![0.0, 0.5, 1.0])
    .with_seed(42)
    .fit_dense(&adj_matrix)
    .expect("Failed to compute embeddings");

assert_eq!(embeddings.nrows(), 3);
assert_eq!(embeddings.ncols(), 128);
```

### 3. Compressed Sparse Row (CSR) Format (Most efficient for large-scale graphs)

For large graphs, you can construct a `CsrMatrix` or pass raw components directly using `from_csr` to avoid dense matrix overhead:

```rust
use fastrp::FastRPBuilder;

// A graph with 3 nodes: Node 0 -> {1, 2}, Node 1 -> {0}, Node 2 -> {0}
let row_ptrs = vec![0, 2, 3, 4];
let col_indices = vec![1, 2, 0, 0];
let values = vec![1.0, 1.0, 1.0, 1.0];
let num_nodes = 3;

// Generate embeddings using raw CSR inputs
let embeddings = FastRPBuilder::new(128)
    .with_weights(vec![0.0, 0.5, 1.0])
    .with_seed(42)
    .from_csr(row_ptrs, col_indices, values, num_nodes)
    .expect("Failed to compute embeddings");

assert_eq!(embeddings.nrows(), 3);
assert_eq!(embeddings.ncols(), 128);
```

---

## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
