use fastrp::{FastRPBuilder, CsrMatrix};
use rand::Rng;
use std::time::Instant;

fn main() {
    let num_nodes = 1_000_000;
    let edges_per_node = 10;
    let dim = 128;
    let seed = 42;

    println!(
        "Generating a random graph with {} nodes and {} edges per node (10M edges total)...",
        num_nodes, edges_per_node
    );

    let start_gen = Instant::now();
    let mut rng = rand::thread_rng();

    // Prepare CSR data
    let mut row_ptrs = Vec::with_capacity(num_nodes + 1);
    let mut col_indices = Vec::with_capacity(num_nodes * edges_per_node);
    let mut values = Vec::with_capacity(num_nodes * edges_per_node);
    
    // Prepare AdjList data
    let mut adj_list: Vec<Vec<(usize, f64)>> = vec![Vec::with_capacity(edges_per_node); num_nodes];

    row_ptrs.push(0);

    for u in 0..num_nodes {
        for _ in 0..edges_per_node {
            let v = rng.gen_range(0..num_nodes);
            let weight = 1.0 / (edges_per_node as f64);
            
            // CSR
            col_indices.push(v);
            values.push(weight);
            
            // AdjList
            adj_list[u].push((v, weight));
        }
        row_ptrs.push(col_indices.len());
    }

    let csr = CsrMatrix::build_from_csr(row_ptrs, col_indices, values, num_nodes).unwrap();

    let gen_duration = start_gen.elapsed();
    println!("Graph generation took: {:?}", gen_duration);

    // Test 1: CSR Undirected = True
    println!("\n--- CSR (Undirected = True) ---");
    let builder_csr_undir = FastRPBuilder::new(dim)
        .with_weights(vec![0.0, 1.0, 1.0])
        .with_seed(seed)
        .with_undirected(true);

    let start_fit = Instant::now();
    let _ = builder_csr_undir.fit_csr(&csr).unwrap();
    println!("Time: {:?}", start_fit.elapsed());

    // Test 2: CSR Undirected = False
    println!("\n--- CSR (Undirected = False) ---");
    let builder_csr_dir = FastRPBuilder::new(dim)
        .with_weights(vec![0.0, 1.0, 1.0])
        .with_seed(seed)
        .with_undirected(false);

    let start_fit = Instant::now();
    let _ = builder_csr_dir.fit_csr(&csr).unwrap();
    println!("Time: {:?}", start_fit.elapsed());

    // Test 3: AdjList Undirected = True
    println!("\n--- AdjList (Undirected = True) ---");
    let builder_adj_undir = FastRPBuilder::new(dim)
        .with_weights(vec![0.0, 1.0, 1.0])
        .with_seed(seed)
        .with_undirected(true);

    let start_fit = Instant::now();
    let _ = builder_adj_undir.fit_adj_list(adj_list.clone()).unwrap();
    println!("Time: {:?}", start_fit.elapsed());

    // Test 4: AdjList Undirected = False
    println!("\n--- AdjList (Undirected = False) ---");
    let builder_adj_dir = FastRPBuilder::new(dim)
        .with_weights(vec![0.0, 1.0, 1.0])
        .with_seed(seed)
        .with_undirected(false);

    let start_fit = Instant::now();
    let _ = builder_adj_dir.fit_adj_list(adj_list).unwrap();
    println!("Time: {:?}", start_fit.elapsed());
}
