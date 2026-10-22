//! Voyage AI embeddings.
//!
//! Run with: VOYAGE_API_KEY=... cargo run --example embedding_voyage

use langchain_rust::embedding::{voyage::VoyageEmbedder, Embedder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let embedder = VoyageEmbedder::new()
        .with_model("voyage-3.5-lite")
        .with_output_dimension(512);

    // Documents and queries are embedded with Voyage's `document` / `query` input types.
    let docs = embedder
        .embed_documents(&[
            "langchain-rust is a port of langchain to Rust.".to_string(),
            "The capital of France is Paris.".to_string(),
        ])
        .await?;
    let query = embedder.embed_query("What is langchain-rust?").await?;

    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    for (i, doc) in docs.iter().enumerate() {
        println!("doc {i}: similarity {:.3}", dot(doc, &query));
    }
    Ok(())
}
