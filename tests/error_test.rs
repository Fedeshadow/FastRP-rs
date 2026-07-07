use fastrp::{FastRPBuilder, FastRPError};

#[test]
fn test_capacity_overflow() {
    let num_nodes = 2;
    // this dim will cause num_nodes.checked_mul(dim) to overflow
    let dim = usize::MAX;

    let adj_list = vec![vec![], vec![]];

    let builder = FastRPBuilder::new(dim);
    let result_csr = builder.fit_adj_list(adj_list.clone());

    assert!(result_csr.is_err());
    match result_csr.unwrap_err() {
        FastRPError::ShapeMismatch(msg) => {
            assert_eq!(msg, "Capacity overflow");
        }
        _ => panic!("Expected ShapeMismatch"),
    }

    let adj_matrix = ndarray::Array2::<f64>::zeros((num_nodes, num_nodes));
    let result_dense = builder.fit_dense(&adj_matrix);
    assert!(result_dense.is_err());
    match result_dense.unwrap_err() {
        FastRPError::ShapeMismatch(msg) => {
            assert_eq!(msg, "Capacity overflow");
        }
        _ => panic!("Expected ShapeMismatch"),
    }
}

#[test]
fn test_oom_error() {
    let num_nodes = 1;
    // this dim won't overflow checked_mul, but is too large to allocate in memory
    // We request enough elements so that it's just under isize::MAX bytes (to avoid CapacityOverflow panic)
    // but way more memory than any machine has (e.g., 1 Exabyte).
    let dim = usize::MAX / 16 - 1;

    let adj_list = vec![vec![]];

    let builder = FastRPBuilder::new(dim);

    let result_csr = builder.fit_adj_list(adj_list.clone());
    assert!(result_csr.is_err());
    match result_csr.unwrap_err() {
        FastRPError::Oom(_) => {} // Expected
        err => panic!("Expected Oom, got {:?}", err),
    }

    let adj_matrix = ndarray::Array2::<f64>::zeros((num_nodes, num_nodes));
    let result_dense = builder.fit_dense(&adj_matrix);
    assert!(result_dense.is_err());
    match result_dense.unwrap_err() {
        FastRPError::Oom(_) => {} // Expected
        err => panic!("Expected Oom, got {:?}", err),
    }
}

#[test]
fn test_empty_graph() {
    let dim = 10;
    let adj_list: Vec<Vec<(usize, f64)>> = vec![];

    let builder = FastRPBuilder::new(dim);

    // CSR
    let result_csr = builder.fit_adj_list(adj_list.clone());
    assert!(result_csr.is_ok());
    assert_eq!(result_csr.unwrap().shape(), &[0, dim]);

    // Dense
    let adj_matrix = ndarray::Array2::<f64>::zeros((0, 0));
    let result_dense = builder.fit_dense(&adj_matrix);
    assert!(result_dense.is_ok());
    assert_eq!(result_dense.unwrap().shape(), &[0, dim]);
}
