// Milvus vector store (REST API v2, Milvus 2.4+ or Zilliz Cloud).
//
// Start Milvus standalone: https://milvus.io/docs/install_standalone-docker.md
//   curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
//   bash standalone_embed.sh start
// Then: OPENAI_API_KEY=... cargo run --example vector_store_milvus --features milvus
// Or with local Ollama embeddings:
//   OPENAI_API_BASE=http://localhost:11434/v1 OPENAI_API_KEY=ollama EMBEDDING_MODEL=nomic-embed-text \
//     cargo run --example vector_store_milvus --features milvus

#[cfg(feature = "milvus")]
#[tokio::main]
async fn main() {
    use langchain_rust::{
        embedding::openai::openai_embedder::OpenAiEmbedder,
        llm::openai::OpenAIConfig,
        schemas::Document,
        vectorstore::{milvus::StoreBuilder, VecStoreOptions, VectorStore},
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

    // The collection is created on the first write (dimension taken from the embedder).
    let store = StoreBuilder::new()
        .url(&std::env::var("MILVUS_URL").unwrap_or("http://localhost:19530".into()))
        // .token("root:Milvus") with auth on, or a Zilliz Cloud API key
        .collection("langchain_rust_example")
        .embedder(embedder)
        .build()
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
    // Namespaces keep tenants apart (the namespace field is the partition key).
    let tenant = VecStoreOptions::default().with_name_space("tenant-1");
    store.add_documents(&docs, &tenant).await.unwrap();

    for result in store
        .similarity_search("Which library is written in Rust?", 2, &tenant)
        .await
        .unwrap()
    {
        println!("{:.3}  {}", result.score, result.page_content);
    }

    // `filters` is a Milvus boolean expression; templates keep values out of the text.
    let filtered = store
        .similarity_search(
            "langchain ports",
            5,
            &VecStoreOptions::default()
                .with_name_space("tenant-1")
                .with_filters(json!({
                    "expr": "metadata[\"year\"] >= {since}",
                    "params": { "since": 2024 },
                })),
        )
        .await
        .unwrap();
    println!(
        "\nfiltered to year >= 2024: {:?}",
        filtered.iter().map(|d| &d.page_content).collect::<Vec<_>>()
    );
}

#[cfg(not(feature = "milvus"))]
fn main() {
    println!("Run with: cargo run --example vector_store_milvus --features milvus");
}
