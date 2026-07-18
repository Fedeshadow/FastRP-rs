# FastRP-rs

A high-performance safe implementation of the **Fast Random Projection (FastRP)** algorithm.

This project implements FastRP in Rust for maximum speed, memory efficiency, and safety. It also provides zero-copy Python bindings via PyO3.

Original Paper: [Fast and Accurate Network Embeddings via Very Sparse Random Projection](https://arxiv.org/abs/1908.11512)

> **Note**: This is an independent implementation and is not the official implementation of the paper's authors.

---

## Project Structure

This repository is organized as a monorepo containing:

- **[fastrp](fastrp/) (Rust Crate)**: Core implementation of the algorithm and sparse matrix formats.
- **[fastrp-py](fastrp-py/) (Python Package)**: Python bindings for `fastrp` exposing a scikit-learn compatible estimator API.

---

## 🦀 Rust Crate (`fastrp`)

`fastrp` is published on [crates.io](https://crates.io/crates/fastrp).

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
fastrp = "0.1.0"
```

### Quick Example (Adjacency List)

```rust
use fastrp::FastRPBuilder;

let adj_list = vec![
    vec![(1, 1.0), (2, 1.0)], // Node 0 connects to Node 1 and Node 2
    vec![(0, 1.0)],           // Node 1 connects to Node 0
    vec![(0, 1.0)],           // Node 2 connects to Node 0
];

let embeddings = FastRPBuilder::new(128)
    .with_weights(vec![0.0, 0.5, 1.0])
    .with_seed(42)
    .fit_adj_list(adj_list)
    .expect("Failed to compute embeddings");
```

For other input formats (e.g., dense matrices and raw CSR data), see the [Rust crate documentation](https://docs.rs/fastrp) or the [fastrp README](fastrp/README.md).

---

## 🐍 Python Package (`fastrp`)

Python bindings are built on top of the Rust core with zero-copy array operations using PyO3 and rust-numpy.

### Installation

```bash
pip install fastrp
```

### Quick Example (Scikit-Learn API)

```python
from fastrp import FastRP
import numpy as np

# A simple adjacency matrix
adj_matrix = np.array([
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0]
])

# Initialize and fit
transformer = FastRP(dim=128, weights=[0.0, 0.5, 1.0], seed=42)
embeddings = transformer.fit_transform(adj_matrix)
```

For more Python usage examples, check out the [python folder](fastrp-py/) and the [examples folder](examples/).

---

## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.