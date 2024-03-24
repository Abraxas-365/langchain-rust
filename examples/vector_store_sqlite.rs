// To run this example execute: cargo run --example vector_store_sqlite --features sqlite
// Make sure vector0 and vss0 libraries are installed in the system or the path of the executable.
// Download the libraries from https://github.com/asg017/sqlite-vss
// For static compliation of sqlite-vss extension refer to the following link:
// https://github.com/launchbadge/sqlx/issues/3147.

#[cfg(feature = "sqlite")]
use langchain_rust::{
    embedding::openai::openai_embedder::OpenAiEmbedder,
    schemas::Document,
    vectorstore::{sqlite_vss::StoreBuilder, VecStoreOptions, VectorStore},
};
#[cfg(feature = "sqlite")]
use std::io::Write;

#[cfg(feature = "sqlite")]
#[tokio::main]
async fn main() {
    // Initialize Embedder
    let embedder = OpenAiEmbedder::default();

    let database_url = std::env::var("DATABASE_URL").unwrap_or("sqlite::memory:".to_string());

    // Initialize the Sqlite Vector Store
    let store = StoreBuilder::new()
        .embedder(embedder)
        .connection_url(database_url)
        .table("documents")
        .vector_dimensions(1536)
        .build()
        .await
        .unwrap();

    // Intialize the tables in the database. This is required to be done only once.
    store.initialize().await.unwrap();

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

#[cfg(not(feature = "sqlite"))]
fn main() {
    println!("This example requires the 'sqlite' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example vector_store_sqlite --features sqlite");
}
