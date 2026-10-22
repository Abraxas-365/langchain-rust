// Weaviate vector store with bring-your-own embeddings (no vectorizer module needed).
//
//   docker run -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
//     -e DEFAULT_VECTORIZER_MODULE=none -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
//     -e CLUSTER_HOSTNAME=node1 cr.weaviate.io/semitechnologies/weaviate:1.32.4
//   OPENAI_API_KEY=... cargo run --example vector_store_weaviate --features weaviate

#[cfg(feature = "weaviate")]
#[tokio::main]
async fn main() {
    use langchain_rust::{
        embedding::openai::openai_embedder::OpenAiEmbedder,
        schemas::Document,
        vectorstore::{weaviate::StoreBuilder, VecStoreOptions, VectorStore},
    };
    use serde_json::json;

    // The collection is created on the first write if it doesn't exist.
    let store = StoreBuilder::new()
        .url("http://localhost:8080")
        .class_name("LangchainRustExample")
        .embedder(OpenAiEmbedder::default())
        .build()
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
    // Namespaces keep tenants apart inside one collection.
    let tenant = VecStoreOptions::default().with_name_space("tenant-1");
    store.add_documents(&docs, &tenant).await.unwrap();

    for result in store
        .similarity_search("Which library is written in Rust?", 2, &tenant)
        .await
        .unwrap()
    {
        println!("{:.3}  {}", result.score, result.page_content);
    }

    // `filters` is a Weaviate where filter in its JSON form.
    let filtered = store
        .similarity_search(
            "langchain ports",
            5,
            &VecStoreOptions::default()
                .with_name_space("tenant-1")
                .with_filters(json!({ "path": ["topic"], "operator": "Equal", "valueText": "go" })),
        )
        .await
        .unwrap();
    println!(
        "\nfiltered to topic=go: {:?}",
        filtered.iter().map(|d| &d.page_content).collect::<Vec<_>>()
    );
}

#[cfg(not(feature = "weaviate"))]
fn main() {
    println!("Run with: cargo run --example vector_store_weaviate --features weaviate");
}
