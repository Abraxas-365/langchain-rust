// To run this example execute: cargo run --example vector_stores --features postgres

#[cfg(feature = "postgres")]
use langchain_rust::{
    add_documents,
    embedding::openai::openai_embedder::OpenAiEmbedder,
    schemas::Document,
    similarity_search,
    vectorstore::{pgvector::StoreBuilder, VectorStore},
};
#[cfg(feature = "postgres")]
use std::io::Write;
#[cfg(feature = "postgres")]
use tokio::io::{self, AsyncBufReadExt, BufReader};

#[cfg(feature = "postgres")]
#[tokio::main]
async fn main() {
    // Initialize Embedder
    let embedder = OpenAiEmbedder::default();

    // Initialize the Postgres Vector Store
    let store = StoreBuilder::new()
        .embedder(embedder)
        .pre_delete_collection(true)
        .connection_url("postgresql://postgres:postgres@localhost:5432/postgres")
        .vector_dimensions(1536)
        .build()
        .await
        .unwrap();

    // Get input with words list
    let mut input = String::new();
    print!("Please enter a list separated by commas: ");
    std::io::stdout().flush().unwrap();
    let mut reader = BufReader::new(io::stdin());
    reader.read_line(&mut input).await.unwrap();
    let input = input.trim_end();
    let list: Vec<&str> = input.split(',').collect();

    // Transform it to a list of documents
    let documents: Vec<Document> = list
        .iter()
        .map(|text| Document::new(text.trim().to_string()))
        .collect();

    // Add documents to the database
    let _ = add_documents!(store, &documents).await.map_err(|e| {
        println!("Error adding documents: {:?}", e);
    });

    // Get the input to search
    let mut search_input = String::new();
    print!("Please enter the text you want to search: ");
    std::io::stdout().flush().unwrap();

    reader.read_line(&mut search_input).await.unwrap();
    let search_input = search_input.trim_end();

    // Perform a similarity search in the database
    let data = similarity_search!(store, search_input, 10)
        .await
        .map_err(|e| {
            println!("Error searching documents: {:?}", e);
        })
        .unwrap();

    data.iter().for_each(|d| println!("{:?}", d.page_content));
}

#[cfg(not(feature = "postgres"))]
fn main() {
    println!("This example requires the 'postgres' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example vector_stores --features postgres");
}
