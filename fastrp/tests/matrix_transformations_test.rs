use fastrp::CsrMatrix;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[test]
fn test_manual_5x5_transpose() {
    // Construct a manual 5x5 CSR Matrix:
    // Row 0: connects to 1 (value 1.5) and 2 (value 2.5) -> index 0..2
    // Row 1: connects to 0 (value 3.5) and 3 (value 4.5) -> index 2..4
    // Row 2: connects to 4 (value 5.5) -> index 4..5
    // Row 3: connects to 1 (value 6.5) and 2 (value 7.5) -> index 5..7
    // Row 4: connects to 0 (value 8.5) and 3 (value 9.5) -> index 7..9
    let row_ptrs = vec![0, 2, 4, 5, 7, 9];
    let col_indices = vec![1, 2, 0, 3, 4, 1, 2, 0, 3];
    let values = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
    let num_nodes = 5;

    let original = CsrMatrix::build_from_csr(row_ptrs, col_indices, values, num_nodes)
        .expect("Failed to build original CsrMatrix");

    let transposed = original.transpose().expect("Failed to transpose matrix");

    // Expected transpose:
    // Row 0 (original col 0): connects to 1 (value 3.5), 4 (value 8.5)
    // Row 1 (original col 1): connects to 0 (value 1.5), 3 (value 6.5)
    // Row 2 (original col 2): connects to 0 (value 2.5), 3 (value 7.5)
    // Row 3 (original col 3): connects to 1 (value 4.5), 4 (value 9.5)
    // Row 4 (original col 4): connects to 2 (value 5.5)
    let expected_row_ptrs = vec![0, 2, 4, 6, 8, 9];
    let expected_col_indices = vec![1, 4, 0, 3, 0, 3, 1, 4, 2];
    let expected_values = vec![3.5, 8.5, 1.5, 6.5, 2.5, 7.5, 4.5, 9.5, 5.5];

    assert_eq!(transposed.num_nodes(), num_nodes);
    assert_eq!(transposed.row_ptrs(), expected_row_ptrs);
    assert_eq!(transposed.col_indices(), expected_col_indices);
    assert_eq!(transposed.values(), expected_values);

    // Verify double transpose brings us back to the original matrix
    let double_transposed = transposed.transpose().expect("Failed to double transpose");
    assert_eq!(double_transposed.num_nodes(), original.num_nodes());
    assert_eq!(double_transposed.row_ptrs(), original.row_ptrs());
    assert_eq!(double_transposed.col_indices(), original.col_indices());
    assert_eq!(double_transposed.values(), original.values());
}

#[test]
fn test_random_100k_nodes_transpose() {
    let num_nodes = 100_000;
    let mut rng = StdRng::seed_from_u64(12345);
    let mut adj_list = Vec::with_capacity(num_nodes);

    for _ in 0..num_nodes {
        let degree = rng.gen_range(0..20);
        let mut neighbors = Vec::with_capacity(degree);
        let mut seen = std::collections::HashSet::with_capacity(degree);
        for _ in 0..degree {
            let mut target = rng.gen_range(0..num_nodes);
            while seen.contains(&target) {
                target = rng.gen_range(0..num_nodes);
            }
            seen.insert(target);
            let weight = rng.gen_range(-10.0..10.0);
            neighbors.push((target, weight));
        }
        // Sort neighbors by target ID so original col_indices are sorted per row.
        // This ensures identity checking with double transpose succeeds exactly.
        neighbors.sort_by_key(|&(target, _)| target);
        adj_list.push(neighbors);
    }

    let original = CsrMatrix::build_from_adj_list(adj_list)
        .expect("Failed to build random CsrMatrix");

    let transposed = original.transpose().expect("Failed to transpose random matrix");

    // Verify structural correctness
    assert_eq!(transposed.num_nodes(), original.num_nodes());
    assert_eq!(transposed.row_ptrs().len(), num_nodes + 1);
    assert_eq!(transposed.row_ptrs()[0], 0);
    assert_eq!(transposed.row_ptrs()[num_nodes], transposed.col_indices().len());
    assert_eq!(transposed.col_indices().len(), transposed.values().len());

    // Verify double transpose matches original exactly
    let double_transposed = transposed.transpose().expect("Failed to double transpose random matrix");
    assert_eq!(double_transposed.num_nodes(), original.num_nodes());
    assert_eq!(double_transposed.row_ptrs(), original.row_ptrs());
    assert_eq!(double_transposed.col_indices(), original.col_indices());
    assert_eq!(double_transposed.values(), original.values());

    // Perform element-wise mapping verification
    let orig_row_ptrs = original.row_ptrs();
    let orig_cols = original.col_indices();
    let orig_vals = original.values();

    let trans_row_ptrs = transposed.row_ptrs();
    let trans_cols = transposed.col_indices();
    let trans_vals = transposed.values();

    for i in 0..num_nodes {
        let start = orig_row_ptrs[i];
        let end = orig_row_ptrs[i + 1];
        for idx in start..end {
            let j = orig_cols[idx];
            let val = orig_vals[idx];

            let trans_start = trans_row_ptrs[j];
            let trans_end = trans_row_ptrs[j + 1];

            // Linear search for source row `i` in the transposed row `j`
            let mut found = false;
            for t_idx in trans_start..trans_end {
                if trans_cols[t_idx] == i {
                    assert!(
                        (trans_vals[t_idx] - val).abs() < 1e-9,
                        "Value mismatch at original edge ({}, {}) with weight {}, transpose weight {}",
                        i, j, val, trans_vals[t_idx]
                    );
                    found = true;
                    break;
                }
            }
            assert!(found, "Edge ({}, {}) with weight {} was not found in transpose as ({}, {})", i, j, val, j, i);
        }
    }
}

#[test]
fn test_manual_5x5_add() {
    // Construct manual 5x5 CSR Matrices A and B:
    // Matrix A:
    // Row 0: 0: 1.0, 2: 2.0
    // Row 1: 1: 3.0
    // Row 2: 0: 4.0, 1: 5.0, 2: 6.0
    // Row 3: (empty)
    // Row 4: 4: 7.0
    let row_ptrs_a = vec![0, 2, 3, 6, 6, 7];
    let col_indices_a = vec![0, 2, 1, 0, 1, 2, 4];
    let values_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let num_nodes = 5;

    let a = CsrMatrix::build_from_csr(row_ptrs_a, col_indices_a, values_a, num_nodes)
        .expect("Failed to build CsrMatrix A");

    // Matrix B:
    // Row 0: 0: -1.0, 1: 1.5, 2: 2.5
    // Row 1: 1: -3.0
    // Row 2: 1: -5.0
    // Row 3: 3: 8.0
    // Row 4: 4: -7.0
    let row_ptrs_b = vec![0, 3, 4, 5, 6, 7];
    let col_indices_b = vec![0, 1, 2, 1, 1, 3, 4];
    let values_b = vec![-1.0, 1.5, 2.5, -3.0, -5.0, 8.0, -7.0];

    let b = CsrMatrix::build_from_csr(row_ptrs_b, col_indices_b, values_b, num_nodes)
        .expect("Failed to build CsrMatrix B");

    // Expected sum (A + B):
    // Row 0: 1: 1.5, 2: 4.5
    // Row 1: (empty - elements at index (1,1) cancelled: 3.0 + (-3.0) = 0.0)
    // Row 2: 0: 4.0, 2: 6.0 (element at index (2,1) cancelled: 5.0 + (-5.0) = 0.0)
    // Row 3: 3: 8.0
    // Row 4: (empty - elements at index (4,4) cancelled: 7.0 + (-7.0) = 0.0)
    let expected_row_ptrs = vec![0, 2, 2, 4, 5, 5];
    let expected_col_indices = vec![1, 2, 0, 2, 3];
    let expected_values = vec![1.5, 4.5, 4.0, 6.0, 8.0];

    let sum = a.add(&b).expect("Failed to add matrices");

    assert_eq!(sum.num_nodes(), num_nodes);
    assert_eq!(sum.row_ptrs(), expected_row_ptrs);
    assert_eq!(sum.col_indices(), expected_col_indices);
    assert_eq!(sum.values(), expected_values);
}

#[test]
fn test_add_dimension_mismatch() {
    let row_ptrs_a = vec![0, 1];
    let col_indices_a = vec![0];
    let values_a = vec![1.0];
    let a = CsrMatrix::build_from_csr(row_ptrs_a, col_indices_a, values_a, 1)
        .expect("Failed to build CsrMatrix A");

    let row_ptrs_b = vec![0, 1, 2];
    let col_indices_b = vec![0, 1];
    let values_b = vec![1.0, 2.0];
    let b = CsrMatrix::build_from_csr(row_ptrs_b, col_indices_b, values_b, 2)
        .expect("Failed to build CsrMatrix B");

    let result = a.add(&b);
    assert!(result.is_err());
    match result.unwrap_err() {
        fastrp::FastRPError::ShapeMismatch(msg) => {
            assert!(msg.contains("Dimension mismatch"));
        }
        _ => panic!("Expected ShapeMismatch"),
    }
}

#[test]
fn test_random_add_properties() {
    let num_nodes = 500;
    let mut rng = StdRng::seed_from_u64(54321);

    // Build random CsrMatrix A
    let mut adj_list_a = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        let degree = rng.gen_range(0..10);
        let mut neighbors = Vec::with_capacity(degree);
        let mut seen = std::collections::HashSet::with_capacity(degree);
        for _ in 0..degree {
            let mut target = rng.gen_range(0..num_nodes);
            while seen.contains(&target) {
                target = rng.gen_range(0..num_nodes);
            }
            seen.insert(target);
            let weight = rng.gen_range(-5.0..5.0);
            neighbors.push((target, weight));
        }
        neighbors.sort_by_key(|&(target, _)| target);
        adj_list_a.push(neighbors);
    }
    let a = CsrMatrix::build_from_adj_list(adj_list_a).expect("Failed to build random CsrMatrix A");

    // Build random CsrMatrix B
    let mut adj_list_b = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        let degree = rng.gen_range(0..10);
        let mut neighbors = Vec::with_capacity(degree);
        let mut seen = std::collections::HashSet::with_capacity(degree);
        for _ in 0..degree {
            let mut target = rng.gen_range(0..num_nodes);
            while seen.contains(&target) {
                target = rng.gen_range(0..num_nodes);
            }
            seen.insert(target);
            let weight = rng.gen_range(-5.0..5.0);
            neighbors.push((target, weight));
        }
        neighbors.sort_by_key(|&(target, _)| target);
        adj_list_b.push(neighbors);
    }
    let b = CsrMatrix::build_from_adj_list(adj_list_b).expect("Failed to build random CsrMatrix B");

    // 1. Commutativity: A + B == B + A
    let sum_ab = a.add(&b).expect("Failed to compute A + B");
    let sum_ba = b.add(&a).expect("Failed to compute B + A");

    assert_eq!(sum_ab.num_nodes(), sum_ba.num_nodes());
    assert_eq!(sum_ab.row_ptrs(), sum_ba.row_ptrs());
    assert_eq!(sum_ab.col_indices(), sum_ba.col_indices());
    // Use epsilon comparison for floating values
    for (v1, v2) in sum_ab.values().iter().zip(sum_ba.values().iter()) {
        assert!((v1 - v2).abs() < 1e-9, "Commutativity value mismatch: {} != {}", v1, v2);
    }

    // 2. Identity element: A + Z == A
    let zero_adj = vec![vec![]; num_nodes];
    let z = CsrMatrix::build_from_adj_list(zero_adj).expect("Failed to build zero matrix");
    let sum_az = a.add(&z).expect("Failed to compute A + Z");

    assert_eq!(sum_az.num_nodes(), a.num_nodes());
    assert_eq!(sum_az.row_ptrs(), a.row_ptrs());
    assert_eq!(sum_az.col_indices(), a.col_indices());
    assert_eq!(sum_az.values(), a.values());

    // 3. Self addition: A + A == 2 * A
    let sum_aa = a.add(&a).expect("Failed to compute A + A");
    assert_eq!(sum_aa.num_nodes(), a.num_nodes());
    assert_eq!(sum_aa.row_ptrs(), a.row_ptrs());
    assert_eq!(sum_aa.col_indices(), a.col_indices());
    for (v1, v2) in sum_aa.values().iter().zip(a.values().iter()) {
        assert!((v1 - 2.0 * v2).abs() < 1e-9, "Self addition value mismatch: {} != 2 * {}", v1, v2);
    }
}

