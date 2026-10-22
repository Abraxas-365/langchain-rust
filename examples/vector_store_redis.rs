// Start Redis 8 (the query engine is built in) or Redis Stack:
//   docker run -p 6379:6379 redis:8
// Then: OPENAI_API_KEY=... cargo run --example vector_store_redis --features redis

#[cfg(feature = "redis")]
#[tokio::main]
async fn main() {
    use langchain_rust::{
        embedding::openai::openai_embedder::OpenAiEmbedder,
        schemas::Document,
        vectorstore::{
            redis::{MetadataField, StoreBuilder},
            VecStoreOptions, VectorStore,
        },
    };
    use serde_json::json;

    let store = StoreBuilder::new()
        .url("redis://127.0.0.1:6379/")
        .index_name("langchain-rust")
        .embedder(OpenAiEmbedder::default())
        // Metadata you want to filter on has to be in the index.
        .metadata_schema([MetadataField::tag("topic"), MetadataField::numeric("year")])
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
    store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();

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

    // An object filters on equality (numbers as ranges, strings as TAGs)...
    let go = store
        .similarity_search(
            "langchain ports",
            5,
            &VecStoreOptions::default().with_filters(json!({ "topic": "go" })),
        )
        .await
        .unwrap();
    println!(
        "\ntopic=go: {:?}",
        go.iter().map(|d| &d.page_content).collect::<Vec<_>>()
    );

    // ...and a string is any Redis query expression.
    let recent = store
        .similarity_search(
            "langchain ports",
            5,
            &VecStoreOptions::default().with_filters(json!("@year:[2024 +inf]")),
        )
        .await
        .unwrap();
    println!(
        "year >= 2024: {:?}",
        recent.iter().map(|d| &d.page_content).collect::<Vec<_>>()
    );
}

#[cfg(not(feature = "redis"))]
fn main() {
    println!("Run with: cargo run --example vector_store_redis --features redis");
}
