use fastrp::FastRPBuilder;
use ndarray::array;

#[test]
fn test_node_features_integration() {
    let num_nodes = 4;
    let dim = 16;
    let seed = 42;

    let adj_list = vec![
        vec![(1, 1.0), (2, 1.0)],
        vec![(0, 1.0), (3, 1.0)],
        vec![(0, 1.0)],
        vec![(1, 1.0)],
    ];

    let weights = vec![0.0, 1.0, 1.0];

    // Compute embeddings without node features
    let emb_base = FastRPBuilder::new(dim)
        .with_weights(weights.clone())
        .with_seed(seed)
        .fit_adj_list(adj_list.clone())
        .expect("Failed to compute base embeddings");

    // Node feature matrix (4 nodes, 3 features each)
    let features = array![
        [1.0, 0.0, 0.5],
        [0.0, 2.0, 1.5],
        [0.5, 0.5, 0.0],
        [1.0, 1.0, 1.0],
    ];

    // Compute embeddings with node features
    let emb_features = FastRPBuilder::new(dim)
        .with_weights(weights.clone())
        .with_seed(seed)
        .with_node_features(features.clone())
        .with_feature_weight(1.0)
        .fit_adj_list(adj_list.clone())
        .expect("Failed to compute feature-influenced embeddings");

    assert_eq!(emb_features.shape(), &[num_nodes, dim]);

    // The embeddings with features should differ from base embeddings
    let mut diff = 0.0f32;
    for i in 0..num_nodes {
        for j in 0..dim {
            diff += (emb_base[[i, j]] - emb_features[[i, j]]).abs();
        }
    }
    assert!(diff > 1e-4, "Features should alter the resulting embeddings");

    // Reproducibility test: identical features + seed must yield identical embeddings
    let emb_features_repeat = FastRPBuilder::new(dim)
        .with_weights(weights)
        .with_seed(seed)
        .with_node_features(features)
        .with_feature_weight(1.0)
        .fit_adj_list(adj_list)
        .expect("Failed to compute feature-influenced embeddings repeat");

    for i in 0..num_nodes {
        for j in 0..dim {
            assert_eq!(
                emb_features[[i, j]],
                emb_features_repeat[[i, j]],
                "Reproducibility failure at [{}, {}]",
                i,
                j
            );
        }
    }
}

#[test]
fn test_node_features_shape_mismatch() {
    let dim = 8;
    let adj_list = vec![
        vec![(1, 1.0)],
        vec![(0, 1.0)],
    ];

    // Node feature matrix with 3 rows (graph only has 2 nodes)
    let invalid_features = array![
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ];

    let result = FastRPBuilder::new(dim)
        .with_node_features(invalid_features)
        .fit_adj_list(adj_list);

    assert!(result.is_err(), "Expected error due to node count mismatch");
    if let Err(e) = result {
        let msg = e.to_string();
        assert!(
            msg.contains("Node features row count 3 does not match graph node count 2"),
            "Unexpected error message: {}",
            msg
        );
    }
}
