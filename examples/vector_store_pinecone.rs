// Uses an existing Pinecone index with the embedder's dimension (1536 for OpenAI's
// text-embedding-3-small). Copy the index host from the console.
//
//   PINECONE_API_KEY=... PINECONE_HOST=my-index-abc123.svc.aped-1234.pinecone.io \
//   OPENAI_API_KEY=... cargo run --example vector_store_pinecone --features pinecone

#[cfg(feature = "pinecone")]
#[tokio::main]
async fn main() {
    use langchain_rust::{
        embedding::openai::openai_embedder::OpenAiEmbedder,
        schemas::Document,
        vectorstore::{pinecone::StoreBuilder, VecStoreOptions, VectorStore},
    };
    use serde_json::json;

    let store = StoreBuilder::new()
        .host(&std::env::var("PINECONE_HOST").expect("PINECONE_HOST"))
        .namespace("langchain-rust")
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
    store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();

    // Pinecone is eventually consistent: freshly upserted vectors can take a moment to show up.
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

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

    // `filters` is a Pinecone metadata filter.
    let filtered = store
        .similarity_search(
            "langchain ports",
            5,
            &VecStoreOptions::default().with_filters(json!({ "topic": { "$eq": "go" } })),
        )
        .await
        .unwrap();
    println!(
        "\nfiltered to topic=go: {:?}",
        filtered.iter().map(|d| &d.page_content).collect::<Vec<_>>()
    );
}

#[cfg(not(feature = "pinecone"))]
fn main() {
    println!("Run with: cargo run --example vector_store_pinecone --features pinecone");
}
