use fastrp::CsrMatrix;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[test]
fn test_undirected_transformations_happy_path() {
    // Construct a directed 3x3 matrix:
    // Row 0: 1: 1.5, 2: 2.0
    // Row 1: 0: 3.0
    // Row 2: (empty)
    //
    // Matrix M:
    // [0.0, 1.5, 2.0]
    // [3.0, 0.0, 0.0]
    // [0.0, 0.0, 0.0]
    //
    // Transpose M^T:
    // [0.0, 3.0, 0.0]
    // [1.5, 0.0, 0.0]
    // [2.0, 0.0, 0.0]
    //
    // M + M^T:
    // [0.0, 4.5, 2.0]
    // [4.5, 0.0, 0.0]
    // [2.0, 0.0, 0.0]
    //
    // Expected row_ptrs: [0, 2, 3, 4]
    // Expected col_indices: [1, 2, 0, 0]
    // Expected values: [4.5, 2.0, 4.5, 2.0]

    let row_ptrs = vec![0, 2, 3, 3];
    let col_indices = vec![1, 2, 0];
    let values = vec![1.5, 2.0, 3.0];
    let num_nodes = 3;

    let m = CsrMatrix::build_from_csr(row_ptrs, col_indices, values, num_nodes)
        .expect("Failed to build original CsrMatrix");

    let m_undirected = m.make_undirected().expect("Failed make_undirected");
    let expected_row_ptrs = vec![0, 2, 3, 4];
    let expected_col_indices = vec![1, 2, 0, 0];
    let expected_values = vec![4.5, 2.0, 4.5, 2.0];

    assert_eq!(m_undirected.num_nodes(), num_nodes);
    assert_eq!(m_undirected.row_ptrs(), expected_row_ptrs);
    assert_eq!(m_undirected.col_indices(), expected_col_indices);
    assert_eq!(m_undirected.values(), expected_values);

    // Verify the original is unchanged
    assert_eq!(m.num_nodes(), num_nodes);
    assert_eq!(m.row_ptrs(), &[0, 2, 3, 3]);
    assert_eq!(m.col_indices(), &[1, 2, 0]);
    assert_eq!(m.values(), &[1.5, 2.0, 3.0]);
}

