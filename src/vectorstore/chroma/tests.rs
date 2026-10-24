use std::sync::Arc;

use mockito::{Matcher, Server, ServerGuard};
use serde_json::json;

use super::*;
use crate::{
    schemas::Document,
    test_utils::FakeEmbedder,
    vectorstore::{VecStoreOptions, VectorStore},
};

const COLLECTIONS: &str = "/api/v2/tenants/default_tenant/databases/default_database/collections";
const COLLECTION_ID: &str = "c0ffee00-0000-0000-0000-000000000001";

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

async fn store_on(server: &mut ServerGuard, builder: StoreBuilder) -> Store {
    let create = server
        .mock("POST", COLLECTIONS)
        .with_status(200)
        .with_body(json!({ "id": COLLECTION_ID, "name": "docs" }).to_string())
        .create_async()
        .await;
    let store = builder
        .url(&server.url())
        .collection_name("docs")
        .embedder(embedder())
        .build()
        .await
        .unwrap();
    create.assert_async().await;
    store
}

fn collection_path(op: &str) -> String {
    format!("{COLLECTIONS}/{COLLECTION_ID}/{op}")
}

fn query_body() -> String {
    json!({
        "ids": [["1", "2"]],
        "documents": [["alpha", "beta"]],
        "metadatas": [[{ "source": "a.txt" }, null]],
        "distances": [[0.25, 0.75]],
        "include": ["documents", "metadatas", "distances"],
    })
    .to_string()
}

#[tokio::test]
async fn build_gets_or_creates_the_collection_with_the_distance_and_token() {
    let mut server = Server::new_async().await;
    let create = server
        .mock("POST", "/api/v2/tenants/acme/databases/prod/collections")
        .match_header("x-chroma-token", "secret")
        .match_body(Matcher::Json(json!({
            "name": "docs",
            "get_or_create": true,
            "configuration": { "hnsw": { "space": "l2" } },
            "metadata": { "team": "search" },
        })))
        .with_body(json!({ "id": COLLECTION_ID }).to_string())
        .create_async()
        .await;

    StoreBuilder::new()
        .url(&format!("{}/", server.url()))
        .tenant("acme")
        .database("prod")
        .api_key("secret")
        .distance(ChromaDistance::L2)
        .collection_metadata(json!({ "team": "search" }))
        .collection_name("docs")
        .embedder(embedder())
        .build()
        .await
        .unwrap();
    create.assert_async().await;
}

#[tokio::test]
async fn build_requires_name_and_embedder_and_surfaces_server_errors() {
    let err = StoreBuilder::new()
        .embedder(embedder())
        .build()
        .await
        .err()
        .unwrap();
    assert!(err.to_string().contains("collection_name"), "{err}");
    let err = StoreBuilder::new()
        .collection_name("docs")
        .build()
        .await
        .err()
        .unwrap();
    assert!(err.to_string().contains("embedder"), "{err}");

    let mut server = Server::new_async().await;
    server
        .mock("POST", COLLECTIONS)
        .with_status(401)
        .with_body(json!({ "error": "AuthError", "message": "bad token" }).to_string())
        .create_async()
        .await;
    let err = StoreBuilder::new()
        .url(&server.url())
        .collection_name("docs")
        .embedder(embedder())
        .build()
        .await
        .err()
        .unwrap();
    assert!(err.to_string().contains("401"), "{err}");
    assert!(err.to_string().contains("bad token"), "{err}");
}

#[tokio::test]
async fn add_documents_upserts_embeddings_texts_and_metadata() {
    let mut server = Server::new_async().await;
    let store = store_on(&mut server, StoreBuilder::new()).await;
    let upsert = server
        .mock("POST", collection_path("upsert").as_str())
        .match_body(Matcher::PartialJson(json!({
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            "documents": ["alpha", "beta"],
            "metadatas": [{ "source": "a.txt", "extra": "{\"k\":1}" }, null],
        })))
        .with_body("{}")
        .create_async()
        .await;

    let docs = vec![
        Document::new("alpha").with_metadata(
            [
                ("source".to_string(), json!("a.txt")),
                ("extra".to_string(), json!({ "k": 1 })),
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
    assert!(ids.iter().all(|id| uuid::Uuid::parse_str(id).is_ok()));
}

#[tokio::test]
async fn add_documents_sends_the_ids_it_returns() {
    let mut server = Server::new_async().await;
    let store = store_on(&mut server, StoreBuilder::new()).await;
    let seen = Arc::new(std::sync::Mutex::new(String::new()));
    let seen_in_mock = seen.clone();
    server
        .mock("POST", collection_path("upsert").as_str())
        .with_body_from_request(move |request| {
            *seen_in_mock.lock().unwrap() = String::from_utf8_lossy(request.body().unwrap()).into();
            b"{}".to_vec()
        })
        .create_async()
        .await;
    let ids = store
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap();
    let sent: serde_json::Value = serde_json::from_str(&seen.lock().unwrap()).unwrap();
    assert_eq!(sent["ids"], json!(ids));
}

#[tokio::test]
async fn add_documents_replaces_caller_ids_by_deleting_then_upserting() {
    let mut server = Server::new_async().await;
    let store = store_on(&mut server, StoreBuilder::new()).await;
    // (endpoint, body) in request order.
    let seen = Arc::new(std::sync::Mutex::new(
        Vec::<(String, serde_json::Value)>::new(),
    ));
    for endpoint in ["upsert", "delete"] {
        let seen_in_mock = seen.clone();
        server
            .mock("POST", collection_path(endpoint).as_str())
            .with_body_from_request(move |request| {
                seen_in_mock.lock().unwrap().push((
                    endpoint.to_string(),
                    serde_json::from_slice(request.body().unwrap()).unwrap(),
                ));
                b"{}".to_vec()
            })
            .create_async()
            .await;
    }
    let docs = [
        Document::new("alpha").with_id("doc-1"),
        Document::new("beta"),
    ];
    let ids = store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(ids[0], "doc-1");
    assert!(uuid::Uuid::parse_str(&ids[1]).is_ok(), "{ids:?}");
    let again = store
        .add_documents(
            &[Document::new("beta").with_id("doc-1")],
            &VecStoreOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(again, ["doc-1"]);
    // Generated ids are new, so plain writes skip the delete.
    store
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap();

    let seen = seen.lock().unwrap();
    let endpoints: Vec<&str> = seen.iter().map(|(e, _)| e.as_str()).collect();
    assert_eq!(
        endpoints,
        ["delete", "upsert", "delete", "upsert", "upsert"]
    );
    // Only caller ids are deleted (upsert merges metadata, so they must go first).
    assert_eq!(seen[0].1, json!({ "ids": ["doc-1"] }));
    assert_eq!(seen[1].1["ids"], json!(ids));
    assert_eq!(seen[2].1, json!({ "ids": ["doc-1"] }));
    assert_eq!(seen[3].1["ids"], json!(["doc-1"]));
    assert_eq!(seen[3].1["documents"], json!(["beta"]));
}

#[tokio::test]
async fn a_failed_delete_stops_the_write() {
    let mut server = Server::new_async().await;
    let store = store_on(&mut server, StoreBuilder::new()).await;
    server
        .mock("POST", collection_path("delete").as_str())
        .with_status(500)
        .with_body(json!({ "error": "Internal", "message": "boom" }).to_string())
        .create_async()
        .await;
    let upsert = server
        .mock("POST", collection_path("upsert").as_str())
        .expect(0)
        .create_async()
        .await;
    let err = store
        .add_documents(
            &[Document::new("alpha").with_id("doc-1")],
            &VecStoreOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("boom"), "{err}");
    upsert.assert_async().await;
}

#[tokio::test]
async fn add_documents_rejects_an_empty_id_before_embedding() {
    let mut server = Server::new_async().await;
    let store = store_on(&mut server, StoreBuilder::new()).await;
    let upsert = server
        .mock("POST", collection_path("upsert").as_str())
        .expect(0)
        .create_async()
        .await;
    let opts = VecStoreOptions::default().with_embedder(crate::test_utils::PanickingEmbedder);
    let err = store
        .add_documents(
            &[Document::new("alpha"), Document::new("beta").with_id("")],
            &opts,
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("must not be empty"), "{err}");
    upsert.assert_async().await;
}

#[tokio::test]
async fn add_documents_with_a_namespace_tags_metadata_and_skips_empty_input() {
    let mut server = Server::new_async().await;
    let store = store_on(&mut server, StoreBuilder::new().namespace_key("tenant_id")).await;
    let upsert = server
        .mock("POST", collection_path("upsert").as_str())
        .match_body(Matcher::PartialJson(json!({
            "metadatas": [{ "tenant_id": "acme" }],
        })))
        .with_body("{}")
        .expect(1)
        .create_async()
        .await;

    let opts = VecStoreOptions::default().with_name_space("acme");
    store
        .add_documents(&[Document::new("alpha")], &opts)
        .await
        .unwrap();
    assert!(store.add_documents(&[], &opts).await.unwrap().is_empty());
    upsert.assert_async().await;
}

#[tokio::test]
async fn add_documents_reports_chroma_errors() {
    let mut server = Server::new_async().await;
    let store = store_on(&mut server, StoreBuilder::new()).await;
    server
        .mock("POST", collection_path("upsert").as_str())
        .with_status(400)
        .with_body(
            json!({ "error": "InvalidArgumentError", "message": "dimension mismatch" }).to_string(),
        )
        .create_async()
        .await;
    let err = store
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("dimension mismatch"), "{err}");
}

#[tokio::test]
async fn similarity_search_returns_documents_with_scores() {
    let mut server = Server::new_async().await;
    let store = store_on(&mut server, StoreBuilder::new()).await;
    let query = server
        .mock("POST", collection_path("query").as_str())
        .match_body(Matcher::Json(json!({
            "query_embeddings": [[0.5, 0.5]],
            "n_results": 2,
            "include": ["documents", "metadatas", "distances"],
        })))
        .with_body(query_body())
        .create_async()
        .await;

    let docs = store
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap();
    query.assert_async().await;

    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].page_content, "alpha");
    assert_eq!(docs[0].metadata["source"], json!("a.txt"));
    assert_eq!(docs[0].score, 0.75);
    assert_eq!(docs[0].id.as_deref(), Some("1"));
    assert_eq!(docs[1].page_content, "beta");
    assert!(docs[1].metadata.is_empty());
    assert_eq!(docs[1].score, 0.25);
    assert_eq!(docs[1].id.as_deref(), Some("2"));
}

#[tokio::test]
async fn similarity_search_scores_l2_and_applies_the_threshold() {
    let mut server = Server::new_async().await;
    let store = store_on(
        &mut server,
        StoreBuilder::new().distance(ChromaDistance::L2),
    )
    .await;
    server
        .mock("POST", collection_path("query").as_str())
        .with_body(query_body())
        .create_async()
        .await;

    // L2 distances 0.25 and 0.75 → scores 0.8 and ~0.571.
    let docs = store
        .similarity_search(
            "query",
            2,
            &VecStoreOptions::default().with_score_threshold(0.7),
        )
        .await
        .unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].page_content, "alpha");
    assert!((docs[0].score - 0.8).abs() < 1e-9);
}

#[tokio::test]
async fn similarity_search_combines_namespace_and_filters() {
    let mut server = Server::new_async().await;
    let store = store_on(&mut server, StoreBuilder::new()).await;

    let only_filter = server
        .mock("POST", collection_path("query").as_str())
        .match_body(Matcher::PartialJson(
            json!({ "where": { "source": "a.txt" } }),
        ))
        .with_body(query_body())
        .create_async()
        .await;
    store
        .similarity_search(
            "query",
            2,
            &VecStoreOptions::default().with_filters(json!({ "source": "a.txt" })),
        )
        .await
        .unwrap();
    only_filter.assert_async().await;

    let both = server
        .mock("POST", collection_path("query").as_str())
        .match_body(Matcher::PartialJson(json!({
            "where": { "$and": [{ "namespace": "acme" }, { "source": "a.txt" }] }
        })))
        .with_body(query_body())
        .create_async()
        .await;
    store
        .similarity_search(
            "query",
            2,
            &VecStoreOptions::default()
                .with_name_space("acme")
                .with_filters(json!({ "source": "a.txt" })),
        )
        .await
        .unwrap();
    both.assert_async().await;
}

#[tokio::test]
async fn similarity_search_handles_empty_results_and_errors() {
    let mut server = Server::new_async().await;
    let store = store_on(&mut server, StoreBuilder::new()).await;
    let empty = server
        .mock("POST", collection_path("query").as_str())
        .with_body(json!({ "ids": [[]], "documents": [[]], "include": [] }).to_string())
        .create_async()
        .await;
    let docs = store
        .similarity_search("query", 3, &VecStoreOptions::default())
        .await
        .unwrap();
    assert!(docs.is_empty());
    empty.remove_async().await;

    server
        .mock("POST", collection_path("query").as_str())
        .with_status(404)
        .with_body(json!({ "error": "NotFoundError", "message": "Collection gone" }).to_string())
        .create_async()
        .await;
    let err = store
        .similarity_search("query", 3, &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Chroma error (404 Not Found): Collection gone"
    );
}

#[tokio::test]
async fn options_embedder_overrides_the_store_embedder() {
    let mut server = Server::new_async().await;
    let store = store_on(&mut server, StoreBuilder::new()).await;
    let query = server
        .mock("POST", collection_path("query").as_str())
        .match_body(Matcher::PartialJson(
            json!({ "query_embeddings": [[9.0, 9.0]] }),
        ))
        .with_body(query_body())
        .create_async()
        .await;
    let other = FakeEmbedder::new(2, &[("query", vec![9.0, 9.0])]);
    store
        .similarity_search("query", 2, &VecStoreOptions::default().with_embedder(other))
        .await
        .unwrap();
    query.assert_async().await;
}

/// Against a real Chroma 1.x server:
/// `uvx --from chromadb chroma run --port 8000` (or `docker run -p 8000:8000 chromadb/chroma`), then
/// `CHROMA_TEST_URL=http://localhost:8000 cargo test --features chroma -- --ignored chroma`.
#[tokio::test]
#[ignore]
async fn live_chroma_round_trip() {
    let url = std::env::var("CHROMA_TEST_URL").unwrap_or("http://localhost:8000".into());
    let embedder = FakeEmbedder::new(
        3,
        &[
            ("rust crab", vec![1.0, 0.0, 0.0]),
            ("go gopher", vec![0.0, 1.0, 0.0]),
            ("python snake", vec![0.0, 0.0, 1.0]),
            ("crabs", vec![0.9, 0.1, 0.0]),
        ],
    );
    let collection = format!("langchain-rust-test-{}", uuid::Uuid::new_v4());
    let store = StoreBuilder::new()
        .url(&url)
        .collection_name(&collection)
        .embedder(embedder)
        .build()
        .await
        .unwrap();

    let docs: Vec<Document> = ["rust crab", "go gopher", "python snake"]
        .into_iter()
        .map(|text| {
            Document::new(text).with_metadata(
                [
                    ("lang".to_string(), json!(text.split(' ').next().unwrap())),
                    ("nested".to_string(), json!({ "ok": true })),
                ]
                .into(),
            )
        })
        .collect();
    store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();
    store
        .add_documents(
            &[Document::new("rust crab")],
            &VecStoreOptions::default().with_name_space("other"),
        )
        .await
        .unwrap();

    let found = store
        .similarity_search("crabs", 2, &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(found.len(), 2);
    assert_eq!(found[0].page_content, "rust crab");
    assert!(found[0].score > 0.9, "{}", found[0].score);
    assert!(found[0].score >= found[1].score);

    let filtered = store
        .similarity_search(
            "crabs",
            5,
            &VecStoreOptions::default().with_filters(json!({ "lang": "go" })),
        )
        .await
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].page_content, "go gopher");
    assert_eq!(filtered[0].metadata["nested"], json!("{\"ok\":true}"));

    let namespaced = store
        .similarity_search(
            "crabs",
            5,
            &VecStoreOptions::default().with_name_space("other"),
        )
        .await
        .unwrap();
    assert_eq!(namespaced.len(), 1);
    assert_eq!(namespaced[0].metadata["namespace"], json!("other"));

    let thresholded = store
        .similarity_search(
            "crabs",
            5,
            &VecStoreOptions::default().with_score_threshold(0.5),
        )
        .await
        .unwrap();
    assert_eq!(thresholded.len(), 2, "both rust crab copies");

    // Caller ids: writing one again replaces the record, and searches return it.
    let ids_ns = VecStoreOptions::default().with_name_space("ids");
    store
        .add_documents(
            &[Document::new("go gopher")
                .with_id("doc-1")
                .with_metadata([("old".to_string(), json!(true))].into())],
            &ids_ns,
        )
        .await
        .unwrap();
    let ids = store
        .add_documents(&[Document::new("rust crab").with_id("doc-1")], &ids_ns)
        .await
        .unwrap();
    assert_eq!(ids, ["doc-1"]);
    let found = store.similarity_search("crabs", 5, &ids_ns).await.unwrap();
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].id.as_deref(), Some("doc-1"));
    assert_eq!(found[0].page_content, "rust crab");
    assert!(!found[0].metadata.contains_key("old"), "{found:?}");

    reqwest::Client::new()
        .delete(format!(
            "{url}/api/v2/tenants/default_tenant/databases/default_database/collections/{collection}"
        ))
        .send()
        .await
        .unwrap();
}
