// Start MongoDB Atlas locally (or point MONGODB_URI at an Atlas cluster):
//   docker run -p 27017:27017 mongodb/mongodb-atlas-local:8
// Then: OPENAI_API_KEY=... cargo run --example vector_store_mongodb --features mongodb

#[cfg(feature = "mongodb")]
#[tokio::main]
async fn main() {
    use langchain_rust::{
        embedding::openai::openai_embedder::OpenAiEmbedder,
        schemas::Document,
        vectorstore::{mongodb::StoreBuilder, VecStoreOptions, VectorStore},
    };
    use serde_json::json;

    let uri = std::env::var("MONGODB_URI")
        .unwrap_or_else(|_| "mongodb://localhost:27017/?directConnection=true".to_string());
    let store = StoreBuilder::new()
        .uri(&uri)
        .database("langchain_rust")
        .collection("docs")
        .embedder(OpenAiEmbedder::default())
        // Paths you want to filter on have to be in the vector index.
        .filter_paths(["metadata.topic", "metadata.year"])
        .build()
        .await
        .unwrap();

    let docs = [
        (
            "langchain-rust is a port of the langchain python library to rust.",
            "rust",
            2024,
        ),
        (
            "langchaingo is a port of the langchain python library to go.",
            "go",
            2023,
        ),
        ("The capital of France is Paris.", "geography", 1999),
    ]
    .map(|(text, topic, year)| {
        Document::new(text).with_metadata(
            [
                ("topic".to_string(), json!(topic)),
                ("year".to_string(), json!(year)),
            ]
            .into(),
        )
    });
    // The first write creates the index and waits until it's queryable.
    store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();
    // Atlas indexes new documents asynchronously.
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    for result in store
        .similarity_search(
            "Which library is written in Rust?",
            2,
            &VecStoreOptions::default(),
        )
        .await
        .unwrap()
    {
        println!("{:.3}  {}", result.score, result.page_content);
    }

    // `filters` is an MQL expression over indexed filter paths.
    let recent = store
        .similarity_search(
            "langchain ports",
            5,
            &VecStoreOptions::default().with_filters(json!({ "metadata.year": { "$gte": 2024 } })),
        )
        .await
        .unwrap();
    println!(
        "\nyear >= 2024: {:?}",
        recent.iter().map(|d| &d.page_content).collect::<Vec<_>>()
    );
}

#[cfg(not(feature = "mongodb"))]
fn main() {
    println!("Run with: cargo run --example vector_store_mongodb --features mongodb");
}
