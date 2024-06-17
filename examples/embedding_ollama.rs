#[cfg(feature = "ollama")]
use langchain_rust::embedding::{
    embedder_trait::Embedder, ollama::ollama_embedder::OllamaEmbedder,
};

#[cfg(feature = "ollama")]
#[tokio::main]
async fn main() {
    let ollama = OllamaEmbedder::default().with_model("nomic-embed-text");

    let response = ollama.embed_query("Why is the sky blue?").await.unwrap();

    println!("{:?}", response);
}

#[cfg(not(feature = "ollama"))]
fn main() {
    println!("This example requires the 'ollama' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example embedding_ollama --features=ollama");
}
