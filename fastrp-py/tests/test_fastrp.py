import numpy as np
import pytest
import fastrp

# Try to import scipy for testing the sparse matrix support
try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# An asymmetric graph (A -> B, B -> C)
# 0 -> 1, 1 -> 2
ASYMMETRIC_ADJ = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
], dtype=np.float64)

ASYMMETRIC_ADJ_LIST = [
    [(1, 1.0)],
    [(2, 1.0)],
    []
]

# CSR representation of the asymmetric graph:
# row_ptrs: node 0 starts at 0, node 1 starts at 1, node 2 starts at 2, end at 2
# col_indices: 0 -> 1, 1 -> 2
# values: 1.0, 1.0
ASYMMETRIC_CSR_ROW_PTRS = [0, 1, 2, 2]
ASYMMETRIC_CSR_COL_INDICES = [1, 2]
ASYMMETRIC_CSR_VALUES = [1.0, 1.0]

def test_dense_directed_undirected():
    # Test directed
    emb_dir = fastrp.fit_dense(ASYMMETRIC_ADJ, dim=8, weights=[0.0, 1.0, 1.0], seed=42, undirected=False)
    assert emb_dir.shape == (3, 8)
    assert not np.allclose(emb_dir, 0.0)

    # Test undirected
    emb_undir = fastrp.fit_dense(ASYMMETRIC_ADJ, dim=8, weights=[0.0, 1.0, 1.0], seed=42, undirected=True)
    assert emb_undir.shape == (3, 8)
    assert not np.allclose(emb_undir, 0.0)

    # They must produce different results because the adjacency structure changes
    assert not np.allclose(emb_dir, emb_undir)

def test_csr_directed_undirected():
    # Test directed
    emb_dir = fastrp.fit_csr(
        ASYMMETRIC_CSR_ROW_PTRS,
        ASYMMETRIC_CSR_COL_INDICES,
        ASYMMETRIC_CSR_VALUES,
        num_nodes=3,
        dim=8,
        weights=[0.0, 1.0, 1.0],
        seed=42,
        undirected=False
    )
    assert emb_dir.shape == (3, 8)
    assert not np.allclose(emb_dir, 0.0)

    # Test undirected
    emb_undir = fastrp.fit_csr(
        ASYMMETRIC_CSR_ROW_PTRS,
        ASYMMETRIC_CSR_COL_INDICES,
        ASYMMETRIC_CSR_VALUES,
        num_nodes=3,
        dim=8,
        weights=[0.0, 1.0, 1.0],
        seed=42,
        undirected=True
    )
    assert emb_undir.shape == (3, 8)
    assert not np.allclose(emb_undir, 0.0)

    # They must produce different results
    assert not np.allclose(emb_dir, emb_undir)

def test_adj_list_directed_undirected():
    # Test directed
    emb_dir = fastrp.fit_adj_list(ASYMMETRIC_ADJ_LIST, dim=8, weights=[0.0, 1.0, 1.0], seed=42, undirected=False)
    assert emb_dir.shape == (3, 8)
    assert not np.allclose(emb_dir, 0.0)

    # Test undirected
    emb_undir = fastrp.fit_adj_list(ASYMMETRIC_ADJ_LIST, dim=8, weights=[0.0, 1.0, 1.0], seed=42, undirected=True)
    assert emb_undir.shape == (3, 8)
    assert not np.allclose(emb_undir, 0.0)

    # They must produce different results
    assert not np.allclose(emb_dir, emb_undir)

def test_seed_consistency():
    # Passing same seed should result in identical embeddings
    emb1 = fastrp.fit_dense(ASYMMETRIC_ADJ, dim=8, weights=[0.0, 1.0, 1.0], seed=123, undirected=True)
    emb2 = fastrp.fit_dense(ASYMMETRIC_ADJ, dim=8, weights=[0.0, 1.0, 1.0], seed=123, undirected=True)
    assert np.allclose(emb1, emb2)

    # Passing different seed should yield different embeddings
    emb3 = fastrp.fit_dense(ASYMMETRIC_ADJ, dim=8, weights=[0.0, 1.0, 1.0], seed=456, undirected=True)
    assert not np.allclose(emb1, emb3)

def test_estimator_dense():
    # Test FastRP class interface for dense matrix
    model = fastrp.FastRP(dim=8, seed=42, undirected=False)
    emb_dir = model.fit_transform(ASYMMETRIC_ADJ)
    assert emb_dir.shape == (3, 8)
    
    model_undir = fastrp.FastRP(dim=8, seed=42, undirected=True)
    emb_undir = model_undir.fit_transform(ASYMMETRIC_ADJ)
    assert emb_undir.shape == (3, 8)
    assert not np.allclose(emb_dir, emb_undir)

def test_estimator_adj_list():
    # Test FastRP class interface for adjacency list
    model = fastrp.FastRP(dim=8, seed=42, undirected=False)
    emb_dir = model.fit_transform(ASYMMETRIC_ADJ_LIST)
    assert emb_dir.shape == (3, 8)

    model_undir = fastrp.FastRP(dim=8, seed=42, undirected=True)
    emb_undir = model_undir.fit_transform(ASYMMETRIC_ADJ_LIST)
    assert emb_undir.shape == (3, 8)
    assert not np.allclose(emb_dir, emb_undir)

def test_estimator_scipy_csr():
    if not HAS_SCIPY:
        pytest.skip("scipy is not installed, skipping sparse matrix estimator tests")

    # Create scipy sparse CSR matrix
    csr_matrix = sp.csr_matrix(ASYMMETRIC_ADJ)
    
    model = fastrp.FastRP(dim=8, seed=42, undirected=False)
    emb_dir = model.fit_transform(csr_matrix)
    assert emb_dir.shape == (3, 8)

    model_undir = fastrp.FastRP(dim=8, seed=42, undirected=True)
    emb_undir = model_undir.fit_transform(csr_matrix)
    assert emb_undir.shape == (3, 8)
    assert not np.allclose(emb_dir, emb_undir)

def test_equivalence_across_formats():
    # Dense, CSR, and AdjList should yield identical results when running with the same settings and seed
    emb_dense = fastrp.fit_dense(ASYMMETRIC_ADJ, dim=8, weights=[0.0, 1.0, 1.0], seed=42, undirected=False)
    
    emb_csr = fastrp.fit_csr(
        ASYMMETRIC_CSR_ROW_PTRS,
        ASYMMETRIC_CSR_COL_INDICES,
        ASYMMETRIC_CSR_VALUES,
        num_nodes=3,
        dim=8,
        weights=[0.0, 1.0, 1.0],
        seed=42,
        undirected=False
    )
    
    emb_adj = fastrp.fit_adj_list(ASYMMETRIC_ADJ_LIST, dim=8, weights=[0.0, 1.0, 1.0], seed=42, undirected=False)
    
    assert np.allclose(emb_dense, emb_csr, atol=1e-6)
    assert np.allclose(emb_dense, emb_adj, atol=1e-6)
