use fastrp::FastRPBuilder;
use rand::Rng;
use std::time::Instant;

fn main() {
    let num_nodes = 10_000_000;
    let edges_per_node = 10;
    let dim = 128;
    let seed = 42;

    println!(
        "Generating a random graph with {} nodes and {} edges per node...",
        num_nodes, edges_per_node
    );

    let start_gen = Instant::now();
    let mut rng = rand::thread_rng();

    let mut row_ptrs = Vec::with_capacity(num_nodes + 1);
    let mut col_indices = Vec::with_capacity(num_nodes * edges_per_node);
    let mut values = Vec::with_capacity(num_nodes * edges_per_node);

    row_ptrs.push(0);

    for _ in 0..num_nodes {
        for _ in 0..edges_per_node {
            let dest = rng.gen_range(0..num_nodes);
            col_indices.push(dest);
            values.push(1.0 / (edges_per_node as f64));
        }
        row_ptrs.push(col_indices.len());
    }

    let gen_duration = start_gen.elapsed();
    println!("Graph generation took: {:?}", gen_duration);

    println!("Initializing FastRPBuilder...");
    let builder = FastRPBuilder::new(dim)
        .with_weights(vec![0.0, 1.0, 1.0])
        .with_seed(seed);

    println!("Computing embeddings using from_csr method...");
    let start_fit = Instant::now();

    match builder.from_csr(row_ptrs, col_indices, values, num_nodes) {
        Ok(embeddings) => {
            let fit_duration = start_fit.elapsed();
            println!("FastRP computation took: {:?}", fit_duration);
            println!("Embeddings Shape: {:?}", embeddings.shape());
            // Just print a slice of the first row as a sample
            let sample_slice: Vec<f32> = embeddings
                .row(0)
                .to_slice()
                .unwrap()
                .iter()
                .take(5)
                .copied()
                .collect();
            println!(
                "Sample Embedding (Node 0, first 5 dims): {:?} ...",
                sample_slice
            );
        }
        Err(e) => {
            eprintln!("Error computing CSR embeddings:\n{}", e);
        }
    }
}
