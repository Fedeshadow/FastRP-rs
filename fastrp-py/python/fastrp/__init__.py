import numpy as np
from ._rust_core import fit_dense as _fit_dense
from ._rust_core import fit_csr as _fit_csr
from ._rust_core import fit_adj_list as _fit_adj_list

from logging import getLogger

logger = getLogger(__name__)


# Resolve scipy once at import time — avoid per-call try/except overhead
try:
    import scipy.sparse as sp
    _HAS_SCIPY = True
except ImportError:
    sp = None
    _HAS_SCIPY = False

__all__ = ["fit_dense", "fit_csr", "fit_adj_list", "FastRP"]

def fit_dense(adj_matrix, dim, weights=[0.0, 1.0, 1.0], seed=None):
    """
    Compute FastRP embeddings from a dense adjacency matrix.
    
    Parameters:
    -----------
    adj_matrix : array-like of shape (n_nodes, n_nodes)
        The dense adjacency matrix.
    dim : int
        The embedding dimension.
    weights : list of float, optional (default=[0.0, 1.0, 1.0])
        Iteration weights.
    seed : int, optional (default=None)
        Random seed.
        
    Returns:
    --------
    embeddings : numpy.ndarray of shape (n_nodes, dim)
        The computed node embeddings.
    """
    adj_matrix = np.asarray(adj_matrix, dtype=np.float64)
    return _fit_dense(adj_matrix, dim, weights, seed)

def fit_csr(row_ptrs, col_indices, values, num_nodes, dim, weights=[0.0, 1.0, 1.0], seed=None):
    """
    Compute FastRP embeddings from Compressed Sparse Row (CSR) matrix components.
    
    Parameters:
    -----------
    row_ptrs : array-like
        The CSR row pointers (indptr).
    col_indices : array-like
        The CSR column indices.
    values : array-like
        The CSR non-zero values.
    num_nodes : int
        The number of nodes in the graph.
    dim : int
        The embedding dimension.
    weights : list of float, optional (default=[0.0, 1.0, 1.0])
        Iteration weights.
    seed : int, optional (default=None)
        Random seed.
        
    Returns:
    --------
    embeddings : numpy.ndarray of shape (num_nodes, dim)
        The computed node embeddings.
    """
    # Pass numpy arrays directly — Rust borrows the buffer zero-copy
    row_ptrs_arr = np.asarray(row_ptrs, dtype=np.int64)
    col_indices_arr = np.asarray(col_indices, dtype=np.int64)
    values_arr = np.asarray(values, dtype=np.float64)
    return _fit_csr(row_ptrs_arr, col_indices_arr, values_arr, int(num_nodes), dim, weights, seed)

def fit_adj_list(adj_list, dim, weights=[0.0, 1.0, 1.0], seed=None):
    """
    Compute FastRP embeddings from an adjacency list.
    
    Parameters:
    -----------
    adj_list : list of list of (int, float)
        The adjacency list where adj_list[i] is a list of tuples (neighbor_id, edge_weight).
    dim : int
        The embedding dimension.
    weights : list of float, optional (default=[0.0, 1.0, 1.0])
        Iteration weights.
    seed : int, optional (default=None)
        Random seed.
        
    Returns:
    --------
    embeddings : numpy.ndarray of shape (n_nodes, dim)
        The computed node embeddings.
    """
    # Build flat COO arrays from the nested adjacency list, then let Rust
    # reconstruct the adj list from contiguous buffers — avoids PyO3 walking
    # a nested Python structure with per-element extraction.
    sources = []
    targets = []
    edge_weights = []
    for src, neighbors in enumerate(adj_list):
        for tgt, w in neighbors:
            sources.append(src)
            targets.append(tgt)
            edge_weights.append(w)
    sources_arr = np.array(sources, dtype=np.int64)
    targets_arr = np.array(targets, dtype=np.int64)
    edge_weights_arr = np.array(edge_weights, dtype=np.float64)
    num_nodes = len(adj_list)
    return _fit_adj_list(sources_arr, targets_arr, edge_weights_arr, num_nodes, dim, weights, seed)


class FastRP:
    """
    Scikit-learn compatible estimator wrapper for the Fast Random Projection (FastRP) algorithm.
    """
    def __init__(self, dim, weights=[0.0, 1.0, 1.0], seed=None):
        self.dim = dim
        self.weights = weights
        self.seed = seed
        self.embeddings_ = None
        
    def fit(self, X, y=None):
        """
        Fit the model using X.
        
        Parameters:
        -----------
        X : input graph in one of the following formats:
            - 2D array-like (dense adjacency matrix)
            - scipy.sparse matrix (any format — non-CSR is auto-converted)
            - list of lists (adjacency list)
        """
        if _HAS_SCIPY and sp.issparse(X):
            if X.format != 'csr':
                X = X.tocsr()
            logger.debug("Fitting FastRP from CSR matrix...")
            self.embeddings_ = fit_csr(
                X.indptr,
                X.indices,
                X.data,
                X.shape[0],
                dim=self.dim,
                weights=self.weights,
                seed=self.seed
            )
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], list):
            if len(X[0]) > 0 and isinstance(X[0][0], (tuple, list)):
                logger.debug("Fitting FastRP from adjacency list...")
                self.embeddings_ = fit_adj_list(X, dim=self.dim, weights=self.weights, seed=self.seed)
            else:
                self.embeddings_ = fit_dense(X, dim=self.dim, weights=self.weights, seed=self.seed)
        else:
            logger.debug("Fitting FastRP from dense matrix...")
            self.embeddings_ = fit_dense(X, dim=self.dim, weights=self.weights, seed=self.seed)
        return self
        
    def fit_transform(self, X, y=None):
        """
        Fit the model using X and return the embeddings.
        """
        self.fit(X, y)
        return self.embeddings_
