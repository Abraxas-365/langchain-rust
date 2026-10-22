use mockito::{Matcher, Server, ServerGuard};
use serde_json::json;

use super::*;
use crate::{
    schemas::Document,
    test_utils::FakeEmbedder,
    vectorstore::{VecStoreOptions, VectorStore},
};

fn embedder() -> FakeEmbedder {
    FakeEmbedder::new(
        2,
        &[
            ("alpha", vec![1.0, 0.0]),
            ("beta", vec![0.0, 1.0]),
            ("query", vec![0.5, 0.5]),
        ],
    )
}

/// Real embedders (OpenAI...) reject empty input, so the store must not call them for it.
struct PanickingEmbedder;

#[async_trait::async_trait]
impl crate::embedding::Embedder for PanickingEmbedder {
    async fn embed_documents(
        &self,
        _: &[String],
    ) -> Result<Vec<Vec<f64>>, crate::embedding::EmbedderError> {
        panic!("embed_documents called")
    }

    async fn embed_query(&self, _: &str) -> Result<Vec<f64>, crate::embedding::EmbedderError> {
        panic!("embed_query called")
    }
}

fn store(server: &ServerGuard, builder: StoreBuilder) -> Store {
    builder
        .host(&server.url())
        .api_key("pc-key")
        .embedder(embedder())
        .build()
        .unwrap()
}

fn matches_body() -> String {
    json!({
        "namespace": "",
        "matches": [
            { "id": "1", "score": 0.75, "metadata": { "text": "alpha", "source": "a.txt" } },
            { "id": "2", "score": 0.4, "metadata": { "text": "beta" } },
        ],
    })
    .to_string()
}

#[test]
fn build_requires_host_embedder_and_key_and_normalizes_the_host() {
    let err = StoreBuilder::new()
        .embedder(embedder())
        .api_key("k")
        .build()
        .err()
        .unwrap();
    assert!(err.to_string().contains("host"), "{err}");
    let err = StoreBuilder::new()
        .host("h")
        .api_key("k")
        .build()
        .err()
        .unwrap();
    assert!(err.to_string().contains("embedder"), "{err}");

    let store = StoreBuilder::new()
        .host("my-index.svc.pinecone.io/")
        .api_key("k")
        .embedder(embedder())
        .build()
        .unwrap();
    assert_eq!(store.host, "https://my-index.svc.pinecone.io");
}

#[tokio::test]
async fn add_documents_upserts_text_metadata_ids_and_headers() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new().namespace("docs"));
    let upsert = server
        .mock("POST", "/vectors/upsert")
        .match_header("Api-Key", "pc-key")
        .match_header("X-Pinecone-Api-Version", DEFAULT_PINECONE_API_VERSION)
        .match_body(Matcher::PartialJson(json!({
            "namespace": "docs",
            "vectors": [
                {
                    "values": [1.0, 0.0],
                    "metadata": { "text": "alpha", "source": "a.txt", "extra": "{\"k\":1}" },
                },
                { "values": [0.0, 1.0], "metadata": { "text": "beta" } },
            ],
        })))
        .with_body(json!({ "upsertedCount": 2 }).to_string())
        .create_async()
        .await;

    let docs = vec![
        Document::new("alpha").with_metadata(
            [
                ("source".to_string(), json!("a.txt")),
                ("extra".to_string(), json!({ "k": 1 })),
                ("dropped".to_string(), json!(null)),
            ]
            .into(),
        ),
        Document::new("beta"),
    ];
    let ids = store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();
    upsert.assert_async().await;
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
}

#[tokio::test]
async fn add_documents_sends_the_ids_it_returns_and_batches_by_100() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    let sent = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Vec<String>>::new()));
    let sent_in_mock = sent.clone();
    let upsert = server
        .mock("POST", "/vectors/upsert")
        .with_body_from_request(move |request| {
            let body: serde_json::Value = serde_json::from_slice(request.body().unwrap()).unwrap();
            let ids = body["vectors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v["id"].as_str().unwrap().to_string())
                .collect();
            sent_in_mock.lock().unwrap().push(ids);
            b"{\"upsertedCount\":1}".to_vec()
        })
        .expect(3)
        .create_async()
        .await;

    let docs: Vec<Document> = (0..250)
        .map(|i| Document::new(format!("doc {i}")))
        .collect();
    let ids = store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();
    upsert.assert_async().await;

    let batches = sent.lock().unwrap().clone();
    assert_eq!(
        batches.iter().map(Vec::len).collect::<Vec<_>>(),
        [100, 100, 50]
    );
    assert_eq!(batches.concat(), ids);
}

#[tokio::test]
async fn add_documents_skips_empty_input_and_reports_errors() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    let upsert = server
        .mock("POST", "/vectors/upsert")
        .with_status(400)
        .with_body(json!({ "code": 3, "message": "Vector dimension 2 does not match the dimension of the index 1536" }).to_string())
        .expect(1)
        .create_async()
        .await;

    assert!(
        store
            .add_documents(
                &[],
                &VecStoreOptions::default().with_embedder(PanickingEmbedder)
            )
            .await
            .unwrap()
            .is_empty(),
        "empty input must not reach the embedder or Pinecone"
    );
    let err = store
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap_err();
    upsert.assert_async().await;
    assert_eq!(
        err.to_string(),
        "Pinecone error (400 Bad Request): Vector dimension 2 does not match the dimension of the index 1536"
    );
}

#[tokio::test]
async fn similarity_search_queries_and_maps_matches() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    let query = server
        .mock("POST", "/query")
        .match_header("Api-Key", "pc-key")
        .match_body(Matcher::Json(json!({
            "vector": [0.5, 0.5],
            "topK": 2,
            "namespace": "",
            "includeMetadata": true,
            "includeValues": false,
        })))
        .with_body(matches_body())
        .create_async()
        .await;

    let docs = store
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap();
    query.assert_async().await;

    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].page_content, "alpha");
    assert_eq!(docs[0].score, 0.75);
    assert_eq!(
        docs[0].metadata.len(),
        1,
        "text key is not repeated in metadata"
    );
    assert_eq!(docs[0].metadata["source"], json!("a.txt"));
    assert_eq!(docs[1].page_content, "beta");
    assert!(docs[1].metadata.is_empty());
}

#[tokio::test]
async fn similarity_search_passes_filter_and_namespace_and_applies_threshold() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new().namespace("default-ns"));
    let query = server
        .mock("POST", "/query")
        .match_body(Matcher::PartialJson(json!({
            "namespace": "tenant-a",
            "filter": { "source": { "$eq": "a.txt" } },
        })))
        .with_body(matches_body())
        .create_async()
        .await;

    let docs = store
        .similarity_search(
            "query",
            2,
            &VecStoreOptions::default()
                .with_name_space("tenant-a")
                .with_filters(json!({ "source": { "$eq": "a.txt" } }))
                .with_score_threshold(0.75),
        )
        .await
        .unwrap();
    query.assert_async().await;
    assert_eq!(
        docs.len(),
        1,
        "a score equal to the threshold is kept, 0.4 is dropped"
    );
    assert_eq!(docs[0].page_content, "alpha");
}

#[tokio::test]
async fn custom_text_key_and_per_call_embedder() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new().text_key("content"));
    let upsert = server
        .mock("POST", "/vectors/upsert")
        .match_body(Matcher::PartialJson(json!({
            "vectors": [{ "values": [7.0, 7.0], "metadata": { "content": "alpha" } }],
        })))
        .with_body("{}")
        .create_async()
        .await;
    let query = server
        .mock("POST", "/query")
        .match_body(Matcher::PartialJson(json!({ "vector": [7.0, 7.0] })))
        .with_body(
            json!({ "matches": [{ "id": "1", "score": 1.0, "metadata": { "content": "alpha", "text": "kept" } }] })
                .to_string(),
        )
        .create_async()
        .await;

    let other = || FakeEmbedder::new(2, &[("alpha", vec![7.0, 7.0])]);
    store
        .add_documents(
            &[Document::new("alpha")],
            &VecStoreOptions::default().with_embedder(other()),
        )
        .await
        .unwrap();
    let docs = store
        .similarity_search(
            "alpha",
            1,
            &VecStoreOptions::default().with_embedder(other()),
        )
        .await
        .unwrap();
    upsert.assert_async().await;
    query.assert_async().await;
    assert_eq!(docs[0].page_content, "alpha");
    assert_eq!(docs[0].metadata["text"], json!("kept"));
}

#[tokio::test]
async fn similarity_search_handles_missing_metadata_and_errors() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    let ok = server
        .mock("POST", "/query")
        .with_body(json!({ "matches": [{ "id": "1", "score": 0.5 }] }).to_string())
        .create_async()
        .await;
    let docs = store
        .similarity_search("query", 1, &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].page_content, "");
    ok.remove_async().await;

    server
        .mock("POST", "/query")
        .with_status(401)
        .with_body("Invalid API Key")
        .create_async()
        .await;
    let err = store
        .similarity_search("query", 1, &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Pinecone error (401 Unauthorized): Invalid API Key"
    );
}

/// Against a real index with dimension 2 and the cosine metric, e.g. Pinecone Local:
///
/// ```sh
/// docker run -d -p 5081:5081 -e PORT=5081 -e INDEX_TYPE=serverless -e VECTOR_TYPE=dense \
///   -e DIMENSION=2 -e METRIC=cosine ghcr.io/pinecone-io/pinecone-index:latest
/// PINECONE_TEST_HOST=http://localhost:5081 cargo test --features pinecone -- --ignored pinecone_live
/// ```
#[tokio::test]
#[ignore = "needs PINECONE_TEST_HOST"]
async fn pinecone_live_round_trip() {
    let host = std::env::var("PINECONE_TEST_HOST").expect("PINECONE_TEST_HOST");
    let namespace = format!("lcr-test-{}", uuid::Uuid::new_v4());
    let store = StoreBuilder::new()
        .host(&host)
        .api_key(&std::env::var("PINECONE_API_KEY").unwrap_or_else(|_| "local".into()))
        .namespace(&namespace)
        .embedder(embedder())
        .build()
        .unwrap();

    let docs = vec![
        Document::new("alpha").with_metadata(
            [
                ("source".to_string(), json!("a.txt")),
                ("tags".to_string(), json!(["x", "y"])),
                ("nested".to_string(), json!({ "k": 1 })),
            ]
            .into(),
        ),
        Document::new("beta").with_metadata([("source".to_string(), json!("b.txt"))].into()),
    ];
    let ids = store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(ids.len(), 2);

    // Pinecone is eventually consistent.
    let mut found = vec![];
    for _ in 0..20 {
        found = store
            .similarity_search("alpha", 2, &VecStoreOptions::default())
            .await
            .unwrap();
        if found.len() == 2 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    assert_eq!(found.len(), 2);
    assert_eq!(found[0].page_content, "alpha");
    assert!((found[0].score - 1.0).abs() < 1e-4, "{}", found[0].score);
    assert_eq!(found[0].metadata["tags"], json!(["x", "y"]));
    assert_eq!(found[0].metadata["nested"], json!("{\"k\":1}"));
    assert!(found[1].score < found[0].score);

    let filtered = store
        .similarity_search(
            "alpha",
            2,
            &VecStoreOptions::default().with_filters(json!({ "source": { "$eq": "b.txt" } })),
        )
        .await
        .unwrap();
    assert_eq!(
        filtered
            .iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>(),
        ["beta"]
    );

    let other_namespace = store
        .similarity_search(
            "alpha",
            2,
            &VecStoreOptions::default().with_name_space(format!("{namespace}-other")),
        )
        .await
        .unwrap();
    assert!(other_namespace.is_empty());
}
