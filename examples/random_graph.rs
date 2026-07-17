use fastrp::FastRPBuilder;
use rand::Rng;

fn main() {
    // Define parameters
    let num_nodes = 5;
    let dim = 4;
    let seed = 42;
    let edge_probability = 0.4;

    println!("Generating a random graph with {} nodes...", num_nodes);
    let mut rng = rand::thread_rng();

    // Create a random adjacency list: Vec<Vec<(TargetNode, Weight)>>
    // For each node, we randomly add connections to other nodes.
    let mut adj_list = vec![vec![]; num_nodes];
    for src in 0..num_nodes {
        for dest in 0..num_nodes {
            if src != dest && rng.gen_bool(edge_probability) {
                let weight = rng.gen_range(0.5..2.0);
                adj_list[src].push((dest, weight));
            }
        }
    }

    // Print the generated random adjacency list
    println!("\nGenerated Random Adjacency List:");
    for (node, neighbors) in adj_list.iter().enumerate() {
        print!("  Node {}: ", node);
        if neighbors.is_empty() {
            println!("No outgoing edges");
        } else {
            let formatted = neighbors
                .iter()
                .map(|(n, w)| format!("(-> {}, weight: {:.2})", n, w))
                .collect::<Vec<_>>()
                .join(", ");
            println!("{}", formatted);
        }
    }

    // Initialize FastRPBuilder
    let builder = FastRPBuilder::new(dim)
        .with_weights(vec![0.0, 1.0, 1.0]) // 0.0 weight on 0-hop, 1.0 on 1-hop, 1.0 on 2-hop
        .with_seed(seed)
        .with_undirected(true); // Compute embeddings for undirected version (M + M^T) by default

    // Compute embeddings using the adjacency list method
    println!("\n--- Computing embeddings using Adjacency List ---");
    match builder.fit_adj_list(adj_list.clone()) {
        Ok(embeddings) => {
            println!("CSR Embeddings Shape: {:?}", embeddings.shape());
            println!("Embeddings:\n{}", embeddings);
        }
        Err(e) => {
            eprintln!("Error computing CSR embeddings: {:?}", e);
        }
    }

    // Compute embeddings using the Dense method
    println!("\n--- Computing embeddings using Dense method ---");
    let mut adj_matrix = ndarray::Array2::<f64>::zeros((num_nodes, num_nodes));
    for (i, neighbors) in adj_list.iter().enumerate() {
        let degree = neighbors.len() as f64;
        let norm = if degree > 0.0 { 1.0 / degree } else { 0.0 };
        for &(target, _w) in neighbors {
            adj_matrix[[i, target]] = norm;
        }
    }
    match builder.fit_dense(&adj_matrix) {
        Ok(embeddings) => {
            println!("Dense Embeddings Shape: {:?}", embeddings.shape());
            println!("Embeddings:\n{}", embeddings);
        }
        Err(e) => {
            eprintln!("Error computing Dense embeddings: {:?}", e);
        }
    }
}
