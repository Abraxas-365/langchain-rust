// Start Chroma (1.x):
//   docker run -p 8000:8000 chromadb/chroma
//   # or: uvx --from chromadb chroma run --port 8000
// Then: OPENAI_API_KEY=... cargo run --example vector_store_chroma --features chroma

#[cfg(feature = "chroma")]
#[tokio::main]
async fn main() {
    use langchain_rust::{
        embedding::openai::openai_embedder::OpenAiEmbedder,
        schemas::Document,
        vectorstore::{chroma::StoreBuilder, VecStoreOptions, VectorStore},
    };
    use serde_json::json;

    let store = StoreBuilder::new()
        .url("http://localhost:8000")
        .collection_name("langchain-rust")
        .embedder(OpenAiEmbedder::default())
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

    // `filters` is a Chroma `where` filter.
    let filtered = store
        .similarity_search(
            "langchain ports",
            5,
            &VecStoreOptions::default().with_filters(json!({ "topic": "go" })),
        )
        .await
        .unwrap();
    println!(
        "\nfiltered to topic=go: {:?}",
        filtered.iter().map(|d| &d.page_content).collect::<Vec<_>>()
    );
}

#[cfg(not(feature = "chroma"))]
fn main() {
    println!("Run with: cargo run --example vector_store_chroma --features chroma");
}
