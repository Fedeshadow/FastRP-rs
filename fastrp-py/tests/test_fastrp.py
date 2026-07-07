import numpy as np
import fastrp

def test_dense():
    adj = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    emb = fastrp.fit_dense(adj, dim=8, weights=[0.0, 1.0, 1.0], seed=42)
    assert emb.shape == (3, 8)
    assert not np.allclose(emb, 0)

def test_csr():
    row_ptrs = [0, 2, 3, 4]
    col_indices = [1, 2, 0, 0]
    values = [1.0, 1.0, 1.0, 1.0]
    emb = fastrp.fit_csr(row_ptrs, col_indices, values, 3, dim=8, weights=[0.0, 1.0, 1.0], seed=42)
    assert emb.shape == (3, 8)
    assert not np.allclose(emb, 0)

def test_adj_list():
    adj_list = [
        [(1, 1.0), (2, 1.0)],
        [(0, 1.0)],
        [(0, 1.0)]
    ]
    emb = fastrp.fit_adj_list(adj_list, dim=8, weights=[0.0, 1.0, 1.0], seed=42)
    assert emb.shape == (3, 8)
    assert not np.allclose(emb, 0)

def test_estimator():
    adj = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    model = fastrp.FastRP(dim=8, seed=42)
    emb = model.fit_transform(adj)
    assert emb.shape == (3, 8)
