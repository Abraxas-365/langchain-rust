// To run this example execute: cargo run --example vector_store_qdrant --features qdrant

#[cfg(feature = "qdrant")]
use langchain_rust::{
    embedding::openai::openai_embedder::OpenAiEmbedder,
    schemas::Document,
    vectorstore::qdrant::{QdrantClient, StoreBuilder},
    vectorstore::VectorStore,
};
#[cfg(feature = "qdrant")]
use std::io::Write;

#[cfg(feature = "qdrant")]
#[tokio::main]
async fn main() {
    // Initialize Embedder

    use langchain_rust::vectorstore::VecStoreOptions;

    // Requires OpenAI API key to be set in the environment variable OPENAI_API_KEY
    let embedder = OpenAiEmbedder::default();

    // Initialize the qdrant_client::QdrantClient
    // Ensure Qdrant is running at localhost, with gRPC port at 6334
    // docker run -p 6334:6334 qdrant/qdrant
    let client = QdrantClient::from_url("http://localhost:6334")
        .build()
        .unwrap();

    let store = StoreBuilder::new()
        .embedder(embedder)
        .client(client)
        .collection_name("langchain-rs")
        .build()
        .await
        .unwrap();

    // Add documents to the database
    let doc1 = Document::new(
        "langchain-rust is a port of the langchain python library to rust and was written in 2024.",
    );
    let doc2 = Document::new(
        "langchaingo is a port of the langchain python library to go language and was written in 2023."
    );
    let doc3 = Document::new(
        "Capital of United States of America (USA) is Washington D.C. and the capital of France is Paris."
    );
    let doc4 = Document::new("Capital of France is Paris.");

    store
        .add_documents(&vec![doc1, doc2, doc3, doc4], &VecStoreOptions::default())
        .await
        .unwrap();

    // Ask for user input
    print!("Query> ");
    std::io::stdout().flush().unwrap();
    let mut query = String::new();
    std::io::stdin().read_line(&mut query).unwrap();

    let results = store
        .similarity_search(&query, 2, &VecStoreOptions::default())
        .await
        .unwrap();

    if results.is_empty() {
        println!("No results found.");
        return;
    } else {
        results.iter().for_each(|r| {
            println!("Document: {}", r.page_content);
        });
    }
}

#[cfg(not(feature = "qdrant"))]
fn main() {
    println!("This example requires the 'qdrant' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example vector_store_qdrant --features qdrant");
}
