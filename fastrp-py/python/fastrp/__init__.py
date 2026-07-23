"""
FastRP-rs: A high-performance safe implementation of the Fast Random Projection algorithm.

> **Note**: This is an independent implementation and is not the official implementation of the paper's authors.

References
----------
.. [1] Chen, Haochen, Syed Fahad Sultan, Yingtao Tian, Muhao Chen, and
   Steven Skiena. "Fast and Accurate Network Embeddings via Very Sparse
   Random Projection." In Proceedings of the 28th ACM International
   Conference on Information and Knowledge Management, pp. 399-408. 2019.
   https://doi.org/10.1145/3357384.3357879
"""

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

def fit_dense(adj_matrix, dim, weights=[0.0, 1.0, 1.0], seed=None, undirected=True, node_features=None, feature_weight=1.0):
    """
    Compute FastRP embeddings from a dense adjacency matrix.
    
    Parameters
    ----------
    adj_matrix : array-like of shape (n_nodes, n_nodes)
        The dense adjacency matrix.
    dim : int
        The embedding dimension.
    weights : list of float, optional (default=[0.0, 1.0, 1.0])
        Iteration weights.
    seed : int, optional (default=None)
        Random seed.
    undirected : bool, optional (default=True)
        Whether to convert the matrix to undirected (M + M^T) before processing.
    node_features : array-like of shape (n_nodes, feature_dim), optional (default=None)
        Optional node feature matrix to influence H0 initialization.
    feature_weight : float, optional (default=1.0)
        Scaling weight for the feature projection matrix in H0.
        
    Returns
    -------
    embeddings : numpy.ndarray of shape (n_nodes, dim)
        The computed node embeddings.

    References
    ----------
    .. [1] Chen, Haochen, Syed Fahad Sultan, Yingtao Tian, Muhao Chen, and
       Steven Skiena. "Fast and Accurate Network Embeddings via Very Sparse
       Random Projection." In Proceedings of the 28th ACM International
       Conference on Information and Knowledge Management, pp. 399-408. 2019.
       https://doi.org/10.1145/3357384.3357879
    """
    adj_matrix = np.asarray(adj_matrix, dtype=np.float64)
    nf_arr = np.asarray(node_features, dtype=np.float64) if node_features is not None else None
    return _fit_dense(adj_matrix, dim, weights, seed, undirected, nf_arr, float(feature_weight))

def fit_csr(row_ptrs, col_indices, values, num_nodes, dim, weights=[0.0, 1.0, 1.0], seed=None, undirected=True, node_features=None, feature_weight=1.0):
    """
    Compute FastRP embeddings from Compressed Sparse Row (CSR) matrix components.
    
    Parameters
    ----------
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
    undirected : bool, optional (default=True)
        Whether to convert the matrix to undirected (M + M^T) before processing.
    node_features : array-like of shape (num_nodes, feature_dim), optional (default=None)
        Optional node feature matrix to influence H0 initialization.
    feature_weight : float, optional (default=1.0)
        Scaling weight for the feature projection matrix in H0.
        
    Returns
    -------
    embeddings : numpy.ndarray of shape (num_nodes, dim)
        The computed node embeddings.

    References
    ----------
    .. [1] Chen, Haochen, Syed Fahad Sultan, Yingtao Tian, Muhao Chen, and
       Steven Skiena. "Fast and Accurate Network Embeddings via Very Sparse
       Random Projection." In Proceedings of the 28th ACM International
       Conference on Information and Knowledge Management, pp. 399-408. 2019.
       https://doi.org/10.1145/3357384.3357879
    """
    # Pass numpy arrays directly — Rust borrows the buffer zero-copy
    row_ptrs_arr = np.asarray(row_ptrs, dtype=np.int64)
    col_indices_arr = np.asarray(col_indices, dtype=np.int64)
    values_arr = np.asarray(values, dtype=np.float64)
    nf_arr = np.asarray(node_features, dtype=np.float64) if node_features is not None else None
    return _fit_csr(row_ptrs_arr, col_indices_arr, values_arr, int(num_nodes), dim, weights, seed, undirected, nf_arr, float(feature_weight))

def fit_adj_list(adj_list, dim, weights=[0.0, 1.0, 1.0], seed=None, undirected=True, node_features=None, feature_weight=1.0):
    """
    Compute FastRP embeddings from an adjacency list.
    
    Parameters
    ----------
    adj_list : list of list of (int, float)
        The adjacency list where adj_list[i] is a list of tuples (neighbor_id, edge_weight).
    dim : int
        The embedding dimension.
    weights : list of float, optional (default=[0.0, 1.0, 1.0])
        Iteration weights.
    seed : int, optional (default=None)
        Random seed.
    undirected : bool, optional (default=True)
        Whether to convert the matrix to undirected (M + M^T) before processing.
    node_features : array-like of shape (n_nodes, feature_dim), optional (default=None)
        Optional node feature matrix to influence H0 initialization.
    feature_weight : float, optional (default=1.0)
        Scaling weight for the feature projection matrix in H0.
        
    Returns
    -------
    embeddings : numpy.ndarray of shape (n_nodes, dim)
        The computed node embeddings.

    References
    ----------
    .. [1] Chen, Haochen, Syed Fahad Sultan, Yingtao Tian, Muhao Chen, and
       Steven Skiena. "Fast and Accurate Network Embeddings via Very Sparse
       Random Projection." In Proceedings of the 28th ACM International
       Conference on Information and Knowledge Management, pp. 399-408. 2019.
       https://doi.org/10.1145/3357384.3357879
    """
    nf_arr = np.asarray(node_features, dtype=np.float64) if node_features is not None else None
    return _fit_adj_list(adj_list, dim, weights, seed, undirected, nf_arr, float(feature_weight))



class FastRP:
    """
    Scikit-learn compatible estimator wrapper for the Fast Random Projection (FastRP) algorithm.

    References
    ----------
    .. [1] Chen, Haochen, Syed Fahad Sultan, Yingtao Tian, Muhao Chen, and
       Steven Skiena. "Fast and Accurate Network Embeddings via Very Sparse
       Random Projection." In Proceedings of the 28th ACM International
       Conference on Information and Knowledge Management, pp. 399-408. 2019.
       https://doi.org/10.1145/3357384.3357879
    """
    def __init__(self, dim, weights=[0.0, 1.0, 1.0], seed=None, undirected=True, node_features=None, feature_weight=1.0):
        self.dim = dim
        self.weights = weights
        self.seed = seed
        self.undirected = undirected
        self.node_features = node_features
        self.feature_weight = feature_weight
        self.embeddings_ = None
        
    def fit(self, X, y=None, node_features=None, feature_weight=None):
        """
        Fit the model using X.
        
        Parameters
        ----------
        X : input graph in one of the following formats:
            - 2D array-like (dense adjacency matrix)
            - scipy.sparse matrix (any format — non-CSR is auto-converted)
            - list of lists (adjacency list)
        node_features : array-like, optional
            Node feature matrix. Overrides self.node_features if specified.
        feature_weight : float, optional
            Feature scaling weight. Overrides self.feature_weight if specified.
        """
        nf = node_features if node_features is not None else self.node_features
        fw = feature_weight if feature_weight is not None else self.feature_weight

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
                seed=self.seed,
                undirected=self.undirected,
                node_features=nf,
                feature_weight=fw,
            )
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], list):
            if len(X[0]) > 0 and isinstance(X[0][0], (tuple, list)):
                logger.debug("Fitting FastRP from adjacency list...")
                self.embeddings_ = fit_adj_list(
                    X,
                    dim=self.dim,
                    weights=self.weights,
                    seed=self.seed,
                    undirected=self.undirected,
                    node_features=nf,
                    feature_weight=fw,
                )
            else:
                self.embeddings_ = fit_dense(
                    X,
                    dim=self.dim,
                    weights=self.weights,
                    seed=self.seed,
                    undirected=self.undirected,
                    node_features=nf,
                    feature_weight=fw,
                )
        else:
            logger.debug("Fitting FastRP from dense matrix...")
            self.embeddings_ = fit_dense(
                X,
                dim=self.dim,
                weights=self.weights,
                seed=self.seed,
                undirected=self.undirected,
                node_features=nf,
                feature_weight=fw,
            )
        return self
        
    def fit_transform(self, X, y=None, node_features=None, feature_weight=None):
        """
        Fit the model using X and return the embeddings.
        """
        self.fit(X, y, node_features=node_features, feature_weight=feature_weight)
        return self.embeddings_

