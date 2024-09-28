// To run this example execute: `cargo run` in the folder.

use langchain_rust::{
    embedding::openai::openai_embedder::OpenAiEmbedder,
    schemas::Document,
    vectorstore::{surrealdb::StoreBuilder, VecStoreOptions, VectorStore},
};
use std::io::Write;

#[tokio::main]
async fn main() {
    // Initialize Embedder
    let embedder = OpenAiEmbedder::default();

    let database_url = std::env::var("DATABASE_URL").unwrap_or("memory".to_string());

    let surrealdb_config = surrealdb::opt::Config::new()
        .set_strict(true)
        .capabilities(surrealdb::opt::capabilities::Capabilities::all());
    //  Uncomment the following lines to enable authentication
    //  .user(surrealdb::opt::auth::Root {
    //      username: "root".into(),
    //      password: "root".into(),
    //  });

    let db = surrealdb::engine::any::connect((database_url, surrealdb_config))
        .await
        .unwrap();
    db.query("DEFINE NAMESPACE test;")
        .await
        .unwrap()
        .check()
        .unwrap();
    db.query("USE NAMESPACE test; DEFINE DATABASE test;")
        .await
        .unwrap()
        .check()
        .unwrap();

    db.use_ns("test").await.unwrap();
    db.use_db("test").await.unwrap();

    // Initialize the Sqlite Vector Store
    let store = StoreBuilder::new()
        .embedder(embedder)
        .db(db)
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
        .similarity_search(
            &query,
            2,
            &VecStoreOptions::default().with_score_threshold(0.6),
        )
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
