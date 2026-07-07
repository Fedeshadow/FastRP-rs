use fastrp::FastRPBuilder;

#[test]
fn test_fastrp_pipeline() {
    let num_nodes = 4;
    let dim = 16;
    
    // Adjacency list for a simple 4-node undirected graph.
    // The structure is `Vec<Vec<(TargetNode, Weight)>>`, where the outer index is the SourceNode.
    let adj_list = vec![
        vec![(1, 1.0), (2, 1.0)], // Node 0 is connected to 1 and 2
        vec![(0, 1.0), (3, 1.0)], // Node 1 is connected to 0 and 3
        vec![(0, 1.0)],           // Node 2 is connected to 0
        vec![(1, 1.0)],           // Node 3 is connected to 1
    ];
    
    // The iterations weights e.g. [alpha1, alpha2, alpha3].
    // Here we use weights mimicking the FastRP baseline:
    // 0.0 weight on the 0-hop (random vector), 1.0 on the 1-hop, 1.0 on the 2-hop.
    let weights = vec![0.0, 1.0, 1.0];
    
    let builder = FastRPBuilder::new(dim).with_weights(weights).with_seed(42);
    
    // 1. Run CSR memory-efficient path using the adjacency list helper
    let embeddings_csr = builder.fit_adj_list(adj_list.clone()).unwrap();
    
    // Verify the returned embeddings have the expected shape: N x d
    assert_eq!(embeddings_csr.shape(), &[num_nodes, dim]);
    // The sum should be non-zero since we aggregated features
    assert!(embeddings_csr.sum() != 0.0);
    
    // 2. Run Dense parallel path (useful for testing or small bounds)
    // Construct the normalized adjacency matrix first
    let mut adj_matrix = ndarray::Array2::<f64>::zeros((num_nodes, num_nodes));
    for (i, neighbors) in adj_list.iter().enumerate() {
        for &(target, w) in neighbors {
            adj_matrix[[i, target]] = w;
        }
    }
    
    let embeddings_dense = builder.fit_dense(&adj_matrix).unwrap();
    
    assert_eq!(embeddings_dense.shape(), &[num_nodes, dim]);
    assert!(embeddings_dense.sum() != 0.0);
    
    // As both execution paths use computationally identical matrices initialized randomly, seeding guarantees
    // the output aligns 1-to-1 cleanly across matrix dimensions allowing deterministic evaluation equivalence.
    for i in 0..num_nodes {
        for j in 0..dim {
            let csr_val = embeddings_csr[[i, j]];
            let dense_val = embeddings_dense[[i, j]];
            assert!(
                (csr_val - dense_val).abs() < 1e-9, 
                "CSR and Dense embeddings differ at index [{}, {}]! {} vs {}", 
                i, j, csr_val, dense_val
            );
        }
    }
}
