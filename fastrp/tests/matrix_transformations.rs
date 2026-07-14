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
fn test_manual_multiply() {
    // Construct manual matrices A and B:
    // A:
    // Row 0: connects to 1 (value 1.0) and 2 (value 2.0)
    // Row 1: connects to 0 (value 3.0)
    // Row 2: connects to 1 (value 4.0)
    let a_row_ptrs = vec![0, 2, 3, 4];
    let a_col_indices = vec![1, 2, 0, 1];
    let a_values = vec![1.0, 2.0, 3.0, 4.0];
    let a = CsrMatrix::build_from_csr(a_row_ptrs, a_col_indices, a_values, 3).unwrap();

    // B:
    // Row 0: connects to 2 (value 5.0)
    // Row 1: connects to 1 (value 6.0)
    // Row 2: connects to 0 (value 7.0) and 2 (value 8.0)
    let b_row_ptrs = vec![0, 1, 2, 4];
    let b_col_indices = vec![2, 1, 0, 2];
    let b_values = vec![5.0, 6.0, 7.0, 8.0];
    let b = CsrMatrix::build_from_csr(b_row_ptrs, b_col_indices, b_values, 3).unwrap();

    let c = a.multiply(&b).expect("Failed to multiply A and B");

    // Expected C = A * B:
    // Row 0: connects to 0 (value 14.0), 1 (value 6.0), and 2 (value 16.0)
    // Row 1: connects to 2 (value 15.0)
    // Row 2: connects to 1 (value 24.0)
    let expected_row_ptrs = vec![0, 3, 4, 5];
    let expected_col_indices = vec![0, 1, 2, 2, 1];
    let expected_values = vec![14.0, 6.0, 16.0, 15.0, 24.0];

    assert_eq!(c.num_nodes(), 3);
    assert_eq!(c.row_ptrs(), expected_row_ptrs);
    assert_eq!(c.col_indices(), expected_col_indices);
    assert_eq!(c.values(), expected_values);
}

#[test]
fn test_random_multiply() {
    let num_nodes = 1000;
    let mut rng = StdRng::seed_from_u64(54321);

    let gen_random_matrix = |rng: &mut StdRng| {
        let mut adj_list = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let degree = rng.gen_range(0..15);
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
            adj_list.push(neighbors);
        }
        CsrMatrix::build_from_adj_list(adj_list).unwrap()
    };

    let a = gen_random_matrix(&mut rng);
    let b = gen_random_matrix(&mut rng);

    let result = a.multiply(&b).expect("Failed to multiply random matrices");

    // Reference implementation
    let mut expected_col_indices = Vec::new();
    let mut expected_values = Vec::new();
    let mut expected_row_ptrs = vec![0];

    let mut accum = vec![0.0; num_nodes];
    let mut marker = vec![usize::MAX; num_nodes];
    let mut touched = Vec::new();

    let a_row_ptrs = a.row_ptrs();
    let a_cols = a.col_indices();
    let a_vals = a.values();

    let b_row_ptrs = b.row_ptrs();
    let b_cols = b.col_indices();
    let b_vals = b.values();

    for i in 0..num_nodes {
        touched.clear();
        let a_start = a_row_ptrs[i];
        let a_end = a_row_ptrs[i + 1];
        for a_idx in a_start..a_end {
            let k = a_cols[a_idx];
            let a_val = a_vals[a_idx];

            let b_start = b_row_ptrs[k];
            let b_end = b_row_ptrs[k + 1];
            for b_idx in b_start..b_end {
                let j = b_cols[b_idx];
                let b_val = b_vals[b_idx];
                let contrib = a_val * b_val;

                if marker[j] == i {
                    accum[j] += contrib;
                } else {
                    marker[j] = i;
                    accum[j] = contrib;
                    touched.push(j);
                }
            }
        }

        touched.sort_unstable();
        for &j in &touched {
            expected_col_indices.push(j);
            expected_values.push(accum[j]);
        }
        expected_row_ptrs.push(expected_col_indices.len());
    }

    assert_eq!(result.num_nodes(), num_nodes);
    assert_eq!(result.row_ptrs(), expected_row_ptrs);
    assert_eq!(result.col_indices(), expected_col_indices);
    // Compare floating values with delta
    assert_eq!(result.values().len(), expected_values.len());
    for (i, (&v1, &v2)) in result.values().iter().zip(expected_values.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-9,
            "Multiplier result mismatch at value index {}: {} vs expected {}",
            i, v1, v2
        );
    }
}

#[test]
fn test_multiply_shape_mismatch() {
    let a = CsrMatrix::build_from_csr(vec![0, 0, 0, 0], vec![], vec![], 3).unwrap();
    let b = CsrMatrix::build_from_csr(vec![0, 0, 0, 0, 0], vec![], vec![], 4).unwrap();
    let result = a.multiply(&b);
    assert!(result.is_err());
    match result.unwrap_err() {
        fastrp::FastRPError::ShapeMismatch(msg) => {
            assert!(msg.contains("cannot multiply: self has 3 nodes but other has 4"));
        }
        err => panic!("Expected ShapeMismatch, got {:?}", err),
    }
}

