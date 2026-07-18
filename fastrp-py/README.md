# FastRP-py

A high-performance, memory-efficient implementation of the Fast Random Projection (FastRP) algorithm in Rust, with safe and ergonomic Python bindings.

> **Note**: This is an independent implementation and is not the official implementation of the paper's authors.

## Features

- **Blazing Fast**: Core routines are implemented in Rust to maximize performance.
- **Scikit-Learn API**: Provides a `FastRP` estimator class that perfectly mimics the scikit-learn API (`fit`, `fit_transform`).
- **Multiple Input Formats**: Directly ingest dense arrays, adjacency lists, or SciPy sparse matrices.
- **Zero-Copy CSR Evaluation**: If using SciPy CSR matrices, the graph structure is passed to Rust with zero-copy overhead.

## Installation

You can install `fastrp` directly from PyPI:

```bash
pip install fastrp
```

To enable **optional** support for SciPy sparse matrices (highly recommended for large graphs), install with the `sparse` extra:

```bash
pip install fastrp[sparse]
```

## Quick Start

The easiest way to use `fastrp` is via its scikit-learn compatible estimator:

```python
import numpy as np
from fastrp import FastRP

# 1. Create a dense adjacency matrix (e.g., 4 nodes)
adj_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [0, 1, 0, 0]
])

# 2. Initialize the estimator
# dim: the target embedding dimension
# weights: the weights for the random projection iterations
model = FastRP(dim=128, weights=[0.0, 1.0, 1.0], seed=42)

# 3. Fit and transform
embeddings = model.fit_transform(adj_matrix)

print(f"Embeddings shape: {embeddings.shape}") # (4, 128)
```

## Advanced Usage

### Large Graphs with SciPy Sparse Matrices

For large networks, a dense matrix will consume too much memory. `fastrp` natively supports zero-copy evaluations of `scipy.sparse.csr_matrix` inputs. Make sure you installed the `sparse` extra dependency.

```python
import scipy.sparse as sp
from fastrp import FastRP

# Create a large random sparse matrix
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
sparse_matrix = sp.csr_matrix((data, (row, col)), shape=(3, 3))

# FastRP will detect the CSR format and process it efficiently
model = FastRP(dim=256, weights=[0.0, 1.0, 1.0])
embeddings = model.fit_transform(sparse_matrix)
```

### Direct API

Alternatively, you can skip the scikit-learn wrapper and use the functional API directly:

```python
import fastrp
import scipy.sparse as sp

# For a CSR matrix:
csr = sp.csr_matrix(...)
embeddings = fastrp.fit_csr(
    csr.indptr, 
    csr.indices, 
    csr.data, 
    num_nodes=csr.shape[0], 
    dim=128
)

# For an adjacency list:
adj_list = [[(1, 1.0), (2, 1.0)], [(0, 1.0)], [(0, 1.0)]]
embeddings = fastrp.fit_adj_list(adj_list, dim=128)
```

## References

[1] Chen, Haochen, Syed Fahad Sultan, Yingtao Tian, Muhao Chen, and Steven Skiena. "Fast and Accurate Network Embeddings via Very Sparse Random Projection." In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management*, pp. 399-408. 2019. https://doi.org/10.1145/3357384.3357879
