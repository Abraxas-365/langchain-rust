// MariaDB vector store (MariaDB 11.7+, native VECTOR type).
//
//   docker run -d --name mariadb -p 3306:3306 -e MARIADB_ROOT_PASSWORD=pw -e MARIADB_DATABASE=lcr mariadb:11.8
// Then: OPENAI_API_KEY=... cargo run --example vector_store_mariadb --features mariadb
// Or with local Ollama embeddings (nomic-embed-text has 768 dimensions):
//   OPENAI_API_BASE=http://localhost:11434/v1 OPENAI_API_KEY=ollama EMBEDDING_MODEL=nomic-embed-text \
//     EMBEDDING_DIMENSIONS=768 cargo run --example vector_store_mariadb --features mariadb

#[cfg(feature = "mariadb")]
#[tokio::main]
async fn main() {
    use langchain_rust::{
        embedding::openai::openai_embedder::OpenAiEmbedder,
        llm::openai::OpenAIConfig,
        schemas::Document,
        vectorstore::{
            mariadb::{MariaDBOptions, StoreBuilder},
            VectorStore,
        },
    };
    use serde_json::json;

    let mut config = OpenAIConfig::default();
    if let Ok(base) = std::env::var("OPENAI_API_BASE") {
        config = config.with_api_base(base);
    }
    let mut embedder = OpenAiEmbedder::new(config);
    if let Ok(model) = std::env::var("EMBEDDING_MODEL") {
        embedder = embedder.with_model(model);
    }
    let dimensions = std::env::var("EMBEDDING_DIMENSIONS")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(1536);

    let store = StoreBuilder::new()
        .embedder(embedder)
        .connection_url(
            std::env::var("MARIADB_URL").unwrap_or("mysql://root:pw@localhost:3306/lcr".into()),
        )
        .vector_dimensions(dimensions)
        .collection_name("langchain_rust_example")
        .embedding_table_name(format!("lc_example_embedding_{dimensions}"))
        .collection_table_name(format!("lc_example_collection_{dimensions}"))
        // Start empty on every run.
        .pre_delete_collection(true)
        .build()
        .await
        .unwrap();

    let docs = [
        (
            "langchain-rust is a port of the langchain python library to rust.",
            "rust",
        ),
        (
            "langchaingo is a port of the langchain python library to go.",
            "go",
        ),
        ("The capital of France is Paris.", "geography"),
    ]
    .map(|(text, topic)| {
        Document::new(text).with_metadata([("topic".to_string(), json!(topic))].into())
    });
    store
        .add_documents(&docs, &MariaDBOptions::default())
        .await
        .unwrap();

    for result in store
        .similarity_search(
            "Which library is written in Rust?",
            2,
            &MariaDBOptions::default(),
        )
        .await
        .unwrap()
    {
        println!("{:.3}  {}", result.score, result.page_content);
    }

    // `filters` is a JSON object matched against metadata (JSON_CONTAINS).
    let filtered = store
        .similarity_search(
            "langchain ports",
            5,
            &MariaDBOptions::default().with_filters(json!({ "topic": "go" })),
        )
        .await
        .unwrap();
    println!(
        "\nfiltered to topic = go: {:?}",
        filtered.iter().map(|d| &d.page_content).collect::<Vec<_>>()
    );
}

#[cfg(not(feature = "mariadb"))]
fn main() {
    println!("Run with: cargo run --example vector_store_mariadb --features mariadb");
}
