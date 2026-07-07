import numpy as np
from ._rust_core import fit_dense as _fit_dense
from ._rust_core import fit_csr as _fit_csr
from ._rust_core import fit_adj_list as _fit_adj_list

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
    row_ptrs_seq = np.asarray(row_ptrs, dtype=np.uintp).tolist()
    col_indices_seq = np.asarray(col_indices, dtype=np.uintp).tolist()
    values_seq = np.asarray(values, dtype=np.float64).tolist()
    return _fit_csr(row_ptrs_seq, col_indices_seq, values_seq, int(num_nodes), dim, weights, seed)

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
    processed_adj_list = []
    for neighbors in adj_list:
        processed_neighbors = []
        for target, weight in neighbors:
            processed_neighbors.append((int(target), float(weight)))
        processed_adj_list.append(processed_neighbors)
    return _fit_adj_list(processed_adj_list, dim, weights, seed)


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
            - scipy.sparse.csr_matrix
            - list of lists (adjacency list)
        """
        try:
            import scipy.sparse as sp
            is_csr = sp.issparse(X) and getattr(X, 'format', None) == 'csr'
        except ImportError:
            is_csr = False
            
        if is_csr:
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
                self.embeddings_ = fit_adj_list(X, dim=self.dim, weights=self.weights, seed=self.seed)
            else:
                self.embeddings_ = fit_dense(X, dim=self.dim, weights=self.weights, seed=self.seed)
        else:
            self.embeddings_ = fit_dense(X, dim=self.dim, weights=self.weights, seed=self.seed)
        return self
        
    def fit_transform(self, X, y=None):
        """
        Fit the model using X and return the embeddings.
        """
        self.fit(X, y)
        return self.embeddings_
