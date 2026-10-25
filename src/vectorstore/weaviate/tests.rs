use mockito::{Matcher, Server, ServerGuard};
use serde_json::{json, Value};

use super::{store::BATCH_SIZE, *};
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
            ("query", vec![0.6, 0.8]),
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
        .url(&server.url())
        .class_name("docs")
        .embedder(embedder())
        .build()
        .unwrap()
}

fn schema(properties: &[(&str, &str)]) -> String {
    json!({
        "class": "Docs",
        "properties": properties
            .iter()
            .map(|(name, ty)| json!({ "name": name, "dataType": [ty] }))
            .collect::<Vec<_>>(),
    })
    .to_string()
}

fn batch_ok(request: &mockito::Request) -> Vec<u8> {
    let body: Value = serde_json::from_slice(request.body().unwrap()).unwrap();
    let results: Vec<Value> = body["objects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| json!({ "id": o["id"], "result": { "status": "SUCCESS" } }))
        .collect();
    serde_json::to_vec(&results).unwrap()
}

fn graphql_query(request: &mockito::Request) -> String {
    let body: Value = serde_json::from_slice(request.body().unwrap()).unwrap();
    body["query"].as_str().unwrap().to_string()
}

#[test]
fn build_validates_and_capitalizes_the_class_name() {
    let err = StoreBuilder::new()
        .embedder(embedder())
        .build()
        .err()
        .unwrap();
    assert!(err.to_string().contains("class_name"), "{err}");
    let err = StoreBuilder::new()
        .class_name("Docs")
        .build()
        .err()
        .unwrap();
    assert!(err.to_string().contains("embedder"), "{err}");
    for bad in [
        StoreBuilder::new().class_name("my-docs"),
        StoreBuilder::new()
            .class_name("Docs")
            .text_key("page content"),
        StoreBuilder::new()
            .class_name("Docs")
            .namespace_key("name.space"),
    ] {
        assert!(bad.embedder(embedder()).build().is_err());
    }

    let store = StoreBuilder::new()
        .url("http://weaviate:8080/")
        .class_name("docs")
        .embedder(embedder())
        .build()
        .unwrap();
    assert_eq!(store.class_name, "Docs");
    assert_eq!(store.url, "http://weaviate:8080");
}

#[tokio::test]
async fn first_write_creates_the_collection_then_batches_objects() {
    let mut server = Server::new_async().await;
    let store = store(
        &server,
        StoreBuilder::new().namespace("tenant-a").api_key("wv-key"),
    );
    let missing = server
        .mock("GET", "/v1/schema/Docs")
        .match_header("authorization", "Bearer wv-key")
        .with_status(404)
        .expect(1)
        .create_async()
        .await;
    let create = server
        .mock("POST", "/v1/schema")
        .match_header("authorization", "Bearer wv-key")
        .match_body(Matcher::Json(json!({
            "class": "Docs",
            "vectorizer": "none",
            "invertedIndexConfig": { "stopwords": { "preset": "none" } },
            "properties": [
                { "name": "text", "dataType": ["text"] },
                { "name": "nameSpace", "dataType": ["text"], "tokenization": "field" },
            ],
        })))
        .with_body("{}")
        .expect(1)
        .create_async()
        .await;
    let batch = server
        .mock("POST", "/v1/batch/objects")
        .match_header("authorization", "Bearer wv-key")
        .match_body(Matcher::PartialJson(json!({
            "objects": [
                {
                    "class": "Docs",
                    "vector": [1.0, 0.0],
                    "properties": {
                        "text": "alpha",
                        "nameSpace": "tenant-a",
                        "source": "a.txt",
                        "extra": "{\"k\":1}",
                    },
                },
                {
                    "class": "Docs",
                    "vector": [0.0, 1.0],
                    "properties": { "text": "beta", "nameSpace": "tenant-a" },
                },
            ],
        })))
        .with_body_from_request(batch_ok)
        .expect(2)
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
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);
    // The schema is only checked once per store.
    store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();

    missing.assert_async().await;
    create.assert_async().await;
    batch.assert_async().await;
}

#[tokio::test]
async fn existing_collection_gets_the_namespace_property_added() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    server
        .mock("GET", "/v1/schema/Docs")
        .with_body(schema(&[("text", "text")]))
        .create_async()
        .await;
    let add_property = server
        .mock("POST", "/v1/schema/Docs/properties")
        .match_body(Matcher::Json(json!({
            "name": "nameSpace", "dataType": ["text"], "tokenization": "field",
        })))
        // Another writer added it first: still fine.
        .with_status(422)
        .with_body(r#"{"error":[{"message":"property nameSpace already exists"}]}"#)
        .create_async()
        .await;
    let create = server
        .mock("POST", "/v1/schema")
        .expect(0)
        .create_async()
        .await;
    let batch = server
        .mock("POST", "/v1/batch/objects")
        .with_body_from_request(batch_ok)
        .create_async()
        .await;

    store
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap();
    add_property.assert_async().await;
    create.assert_async().await;
    batch.assert_async().await;
}

#[tokio::test]
async fn schema_errors_other_than_already_exists_fail_the_write() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    server
        .mock("GET", "/v1/schema/Docs")
        .with_status(404)
        .create_async()
        .await;
    server
        .mock("POST", "/v1/schema")
        .with_status(422)
        .with_body(r#"{"error":[{"message":"vectorizer: no module with name \"none\""}]}"#)
        .create_async()
        .await;
    let batch = server
        .mock("POST", "/v1/batch/objects")
        .expect(0)
        .create_async()
        .await;
    let err = store
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap_err();
    batch.assert_async().await;
    assert_eq!(
        err.to_string(),
        r#"Weaviate error (422 Unprocessable Entity): vectorizer: no module with name "none""#
    );
}

#[tokio::test]
async fn ids_match_what_was_sent_and_batches_hold_100_objects() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    server
        .mock("GET", "/v1/schema/Docs")
        .with_body(schema(&[("text", "text"), ("nameSpace", "text")]))
        .create_async()
        .await;
    let sent = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Vec<String>>::new()));
    let sent_in_mock = sent.clone();
    let batch = server
        .mock("POST", "/v1/batch/objects")
        .with_body_from_request(move |request| {
            let body: Value = serde_json::from_slice(request.body().unwrap()).unwrap();
            sent_in_mock.lock().unwrap().push(
                body["objects"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|o| o["id"].as_str().unwrap().to_string())
                    .collect(),
            );
            batch_ok(request)
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
    batch.assert_async().await;
    let batches = sent.lock().unwrap().clone();
    assert_eq!(
        batches.iter().map(Vec::len).collect::<Vec<_>>(),
        [100, 100, 50]
    );
    assert_eq!(batches.concat(), ids);
}

#[tokio::test]
async fn caller_ids_must_be_uuids_and_are_sent_normalized() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    server
        .mock("GET", "/v1/schema/Docs")
        .with_body(schema(&[("text", "text"), ("nameSpace", "text")]))
        .create_async()
        .await;
    let batch = server
        .mock("POST", "/v1/batch/objects")
        .match_body(Matcher::PartialJson(json!({
            "objects": [{ "id": "6f1c2a8e-0b1d-4c3e-9f00-0123456789ab" }],
        })))
        .with_body_from_request(batch_ok)
        .expect(1)
        .create_async()
        .await;
    // Uppercase / simple forms are accepted and sent the way Weaviate returns them.
    let ids = store
        .add_documents(
            &[Document::new("alpha").with_id("6F1C2A8E0B1D4C3E9F000123456789AB")],
            &VecStoreOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(ids, ["6f1c2a8e-0b1d-4c3e-9f00-0123456789ab"]);
    batch.assert_async().await;

    // Non-UUID and empty ids fail before embedding or any request.
    let opts = VecStoreOptions::default().with_embedder(PanickingEmbedder);
    for bad in ["doc-1", ""] {
        let err = store
            .add_documents(&[Document::new("alpha").with_id(bad)], &opts)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("UUID") || err.to_string().contains("must not be empty"),
            "{bad:?}: {err}"
        );
    }
}

#[tokio::test]
async fn per_object_batch_failures_and_http_errors_are_reported() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    server
        .mock("GET", "/v1/schema/Docs")
        .with_body(schema(&[("text", "text"), ("nameSpace", "text")]))
        .create_async()
        .await;
    let partial = server
        .mock("POST", "/v1/batch/objects")
        .with_body(
            json!([
                { "result": { "status": "SUCCESS" } },
                { "result": { "status": "FAILED", "errors": { "error": [{ "message": "2 vs 3: vector lengths don't match" }] } } },
            ])
            .to_string(),
        )
        .create_async()
        .await;
    assert!(store
        .add_documents(
            &[],
            &VecStoreOptions::default().with_embedder(PanickingEmbedder)
        )
        .await
        .unwrap()
        .is_empty());
    let err = store
        .add_documents(
            &[Document::new("alpha"), Document::new("beta")],
            &VecStoreOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Weaviate rejected 1 object(s): 2 vs 3: vector lengths don't match"
    );
    partial.remove_async().await;

    server
        .mock("POST", "/v1/batch/objects")
        .with_status(401)
        .with_body(r#"{"code":401,"message":"anonymous access not enabled"}"#)
        .create_async()
        .await;
    let err = store
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        r#"Weaviate error (401 Unauthorized): {"code":401,"message":"anonymous access not enabled"}"#
    );
}

#[tokio::test]
async fn similarity_search_builds_the_graphql_query_and_maps_hits() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    server
        .mock("GET", "/v1/schema/Docs")
        .with_body(schema(&[
            ("text", "text"),
            ("nameSpace", "text"),
            ("source", "text"),
            ("tags", "text[]"),
            ("author", "cref"),
            ("geo", "geoCoordinates"),
        ]))
        .create_async()
        .await;
    let query = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let query_in_mock = query.clone();
    let graphql = server
        .mock("POST", "/v1/graphql")
        .with_body_from_request(move |request| {
            *query_in_mock.lock().unwrap() = graphql_query(request);
            json!({ "data": { "Get": { "Docs": [
                { "text": "alpha", "source": "a.txt", "tags": ["x"], "nameSpace": null,
                  "_additional": { "id": "1", "certainty": 0.9, "distance": 0.2 } },
                { "text": "beta", "source": null, "tags": null, "nameSpace": null,
                  "_additional": { "id": "2", "certainty": 0.5, "distance": 1.0 } },
            ] } } })
            .to_string()
            .into_bytes()
        })
        .create_async()
        .await;

    let docs = store
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap();
    graphql.assert_async().await;

    assert_eq!(
        *query.lock().unwrap(),
        "{ Get { Docs(nearVector: {vector: [0.6, 0.8]}, limit: 2) \
         { text nameSpace source tags _additional { id certainty distance } } } }"
    );
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].page_content, "alpha");
    assert_eq!(docs[0].score, 0.9);
    assert_eq!(docs[0].id.as_deref(), Some("1"));
    assert_eq!(docs[1].id.as_deref(), Some("2"));
    assert_eq!(
        docs[0].metadata,
        std::collections::HashMap::from([
            ("source".to_string(), json!("a.txt")),
            ("tags".to_string(), json!(["x"])),
        ])
    );
    assert_eq!(docs[1].page_content, "beta");
    assert!(docs[1].metadata.is_empty(), "nulls are not metadata");
}

#[tokio::test]
async fn similarity_search_combines_namespace_and_filter_and_applies_threshold() {
    let mut server = Server::new_async().await;
    let store = store(
        &server,
        StoreBuilder::new()
            .namespace("default-ns")
            .metadata_fields(["source"]),
    );
    let schema_lookup = server
        .mock("GET", "/v1/schema/Docs")
        .expect(0)
        .create_async()
        .await;
    let query = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let query_in_mock = query.clone();
    server
        .mock("POST", "/v1/graphql")
        .with_body_from_request(move |request| {
            *query_in_mock.lock().unwrap() = graphql_query(request);
            json!({ "data": { "Get": { "Docs": [
                { "text": "alpha", "_additional": { "certainty": 0.75 } },
                { "text": "beta", "_additional": { "certainty": 0.4 } },
            ] } } })
            .to_string()
            .into_bytes()
        })
        .create_async()
        .await;

    let docs = store
        .similarity_search(
            "query",
            5,
            &VecStoreOptions::default()
                .with_name_space("tenant-a")
                .with_filters(
                    json!({ "path": ["source"], "operator": "Equal", "valueText": "a.txt" }),
                )
                .with_score_threshold(0.75),
        )
        .await
        .unwrap();
    schema_lookup.assert_async().await;
    assert_eq!(
        *query.lock().unwrap(),
        r#"{ Get { Docs(nearVector: {vector: [0.6, 0.8]}, limit: 5, where: {operands: [{operator: Equal, path: ["nameSpace"], valueText: "tenant-a"}, {operator: Equal, path: ["source"], valueText: "a.txt"}], operator: And}) { text source _additional { id certainty distance } } } }"#
    );
    assert_eq!(
        docs.iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>(),
        ["alpha"],
        "a score equal to the threshold is kept"
    );

    // Builder namespace alone, as a plain where filter.
    store
        .similarity_search("query", 1, &VecStoreOptions::default())
        .await
        .unwrap();
    assert!(query.lock().unwrap().contains(
        r#"limit: 1, where: {operator: Equal, path: ["nameSpace"], valueText: "default-ns"}) {"#
    ));
}

#[tokio::test]
async fn per_call_embedder_is_used_for_writes_and_searches() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new().metadata_fields(["text"]));
    server
        .mock("GET", "/v1/schema/Docs")
        .with_body(schema(&[("text", "text"), ("nameSpace", "text")]))
        .create_async()
        .await;
    let batch = server
        .mock("POST", "/v1/batch/objects")
        .match_body(Matcher::PartialJson(
            json!({ "objects": [{ "vector": [7.0, 7.0] }] }),
        ))
        .with_body_from_request(batch_ok)
        .create_async()
        .await;
    let graphql = server
        .mock("POST", "/v1/graphql")
        .match_body(Matcher::Regex(
            r"nearVector: \{vector: \[7\.0, 7\.0\]\}".into(),
        ))
        .with_body(r#"{"data":{"Get":{"Docs":[]}}}"#)
        .create_async()
        .await;

    let other = || {
        VecStoreOptions::default().with_embedder(FakeEmbedder::new(2, &[("alpha", vec![7.0, 7.0])]))
    };
    store
        .add_documents(&[Document::new("alpha")], &other())
        .await
        .unwrap();
    store.similarity_search("alpha", 1, &other()).await.unwrap();
    batch.assert_async().await;
    graphql.assert_async().await;
}

#[tokio::test]
async fn search_scores_without_certainty_use_negative_distance() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new().metadata_fields(["text"]));
    server
        .mock("POST", "/v1/graphql")
        .with_body(
            json!({ "data": { "Get": { "Docs": [
                { "text": "near", "_additional": { "certainty": null, "distance": 0.5 } },
                { "text": "far", "_additional": { "certainty": null, "distance": 3.0 } },
            ] } } })
            .to_string(),
        )
        .create_async()
        .await;
    let docs = store
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(
        docs.iter().map(|d| d.score).collect::<Vec<_>>(),
        [-0.5, -3.0]
    );
}

#[tokio::test]
async fn search_on_a_missing_collection_is_empty_and_errors_are_reported() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    let missing = server
        .mock("GET", "/v1/schema/Docs")
        .with_status(404)
        .create_async()
        .await;
    let graphql = server
        .mock("POST", "/v1/graphql")
        .expect(0)
        .create_async()
        .await;
    assert!(store
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap()
        .is_empty());
    graphql.assert_async().await;
    missing.remove_async().await;
    graphql.remove_async().await;

    server
        .mock("GET", "/v1/schema/Docs")
        .with_body(schema(&[("text", "text")]))
        .create_async()
        .await;
    server
        .mock("POST", "/v1/graphql")
        .with_body(r#"{"data":{"Get":{"Docs":null}},"errors":[{"message":"vector lengths don't match: 2 vs 3"}]}"#)
        .create_async()
        .await;
    let err = store
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Weaviate GraphQL error: vector lengths don't match: 2 vs 3"
    );

    let err = store
        .similarity_search(
            "query",
            2,
            &VecStoreOptions::default().with_filters(json!({ "operator": "Equal) { x }" })),
        )
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("invalid Weaviate filter operator"),
        "{err}"
    );
}

const ID_A: &str = "6a1c3f4e-0000-4000-8000-00000000000a";
const ID_B: &str = "6a1c3f4e-0000-4000-8000-00000000000b";

#[tokio::test]
async fn get_by_ids_queries_by_id_within_the_namespace_in_request_order() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new().namespace("tenant"));
    server
        .mock("GET", "/v1/schema/Docs")
        .with_body(schema(&[("text", "text"), ("source", "text")]))
        .create_async()
        .await;
    let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let sink = captured.clone();
    let graphql = server
        .mock("POST", "/v1/graphql")
        .match_request(move |request| {
            *sink.lock().unwrap() = graphql_query(request);
            true
        })
        .with_body(
            json!({ "data": { "Get": { "Docs": [
                { "text": "alpha", "source": "a.txt", "_additional": { "id": ID_A } },
                { "text": "beta", "source": null, "_additional": { "id": ID_B } },
            ] } } })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;
    let docs = store
        .get_by_ids(
            &[ID_B.to_uppercase(), "not-a-uuid".into(), ID_A.into()],
            &VecStoreOptions::default(),
        )
        .await
        .unwrap();
    graphql.assert_async().await;
    let ids: Vec<_> = docs.iter().map(|d| d.id.as_deref().unwrap()).collect();
    assert_eq!(ids, [ID_B, ID_A]);
    assert_eq!(docs[0].page_content, "beta");
    assert!(docs[0].metadata.is_empty());
    assert_eq!(docs[1].metadata["source"], json!("a.txt"));
    assert_eq!(docs[1].score, 0.0);

    let query = captured.lock().unwrap().clone();
    assert!(query.starts_with("{ Get { Docs(where: "), "{query}");
    assert!(query.contains("limit: 2"), "{query}");
    assert!(query.contains("text source _additional { id }"), "{query}");
    let by_id = |id: &str| format!(r#"{{operator: Equal, path: ["id"], valueText: "{id}"}}"#);
    assert!(query.contains(&by_id(ID_A)), "{query}");
    assert!(query.contains(&by_id(ID_B)), "{query}");
    assert!(!query.contains("not-a-uuid"), "{query}");
    assert!(
        query.contains(r#"{operator: Equal, path: ["nameSpace"], valueText: "tenant"}"#),
        "{query}"
    );
    assert!(query.contains("operator: And"), "{query}");
    assert!(query.contains("operator: Or"), "{query}");
}

#[tokio::test]
async fn get_by_ids_batches_and_skips_missing_collections_and_bad_ids() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    let graphql = server
        .mock("POST", "/v1/graphql")
        .expect(0)
        .create_async()
        .await;
    // No UUIDs: no request at all, not even the schema.
    let schema_mock = server
        .mock("GET", "/v1/schema/Docs")
        .with_status(404)
        .expect(1)
        .create_async()
        .await;
    let opt = VecStoreOptions::default();
    assert!(store
        .get_by_ids(&["x".into()], &opt)
        .await
        .unwrap()
        .is_empty());
    assert!(store
        .get_by_ids(&[ID_A.into()], &opt)
        .await
        .unwrap()
        .is_empty());
    schema_mock.assert_async().await;
    graphql.assert_async().await;
    schema_mock.remove_async().await;
    graphql.remove_async().await;

    server
        .mock("GET", "/v1/schema/Docs")
        .with_body(schema(&[("text", "text")]))
        .create_async()
        .await;
    let batches = server
        .mock("POST", "/v1/graphql")
        .with_body(r#"{"data":{"Get":{"Docs":[]}}}"#)
        .expect(2)
        .create_async()
        .await;
    let many: Vec<String> = (0..BATCH_SIZE + 1)
        .map(|_| uuid::Uuid::new_v4().to_string())
        .collect();
    assert!(store.get_by_ids(&many, &opt).await.unwrap().is_empty());
    batches.assert_async().await;
    batches.remove_async().await;

    server
        .mock("POST", "/v1/graphql")
        .with_body(r#"{"errors":[{"message":"boom"}]}"#)
        .create_async()
        .await;
    let err = store.get_by_ids(&many, &opt).await.unwrap_err();
    assert_eq!(err.to_string(), "Weaviate GraphQL error: boom");
}

#[tokio::test]
async fn delete_batch_deletes_by_id_within_the_scope() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    server
        .mock("GET", "/v1/schema/Docs")
        .with_body(schema(&[("text", "text")]))
        .create_async()
        .await;
    let delete = server
        .mock("DELETE", "/v1/batch/objects")
        .match_body(Matcher::Json(json!({
            "match": {
                "class": "Docs",
                "where": {
                    "operator": "And",
                    "operands": [
                        { "operator": "Or", "operands": [
                            { "path": ["id"], "operator": "Equal", "valueText": ID_A },
                        ] },
                        { "path": ["nameSpace"], "operator": "Equal", "valueText": "t" },
                    ],
                },
            },
            "output": "minimal",
        })))
        .with_body(r#"{"results":{"matches":1,"successful":1,"failed":0}}"#)
        .expect(1)
        .create_async()
        .await;
    store
        .delete(
            &[ID_A.to_uppercase(), "nope".into()],
            &VecStoreOptions::default().with_name_space("t"),
        )
        .await
        .unwrap();
    delete.assert_async().await;
    delete.remove_async().await;

    let batches = server
        .mock("DELETE", "/v1/batch/objects")
        .with_body(r#"{"results":{"failed":0}}"#)
        .expect(2)
        .create_async()
        .await;
    let many: Vec<String> = (0..BATCH_SIZE + 1)
        .map(|_| uuid::Uuid::new_v4().to_string())
        .collect();
    let opt = VecStoreOptions::default();
    store.delete(&many, &opt).await.unwrap();
    store.delete(&["nope".into()], &opt).await.unwrap();
    batches.assert_async().await;
    batches.remove_async().await;

    server
        .mock("DELETE", "/v1/batch/objects")
        .with_body(r#"{"results":{"failed":2}}"#)
        .create_async()
        .await;
    let err = store.delete(&[ID_A.into()], &opt).await.unwrap_err();
    assert!(err.to_string().contains("failed to delete 2"), "{err}");
}

#[tokio::test]
async fn delete_on_a_missing_collection_sends_nothing() {
    let mut server = Server::new_async().await;
    let store = store(&server, StoreBuilder::new());
    server
        .mock("GET", "/v1/schema/Docs")
        .with_status(404)
        .create_async()
        .await;
    let delete = server
        .mock("DELETE", "/v1/batch/objects")
        .expect(0)
        .create_async()
        .await;
    store
        .delete(&[ID_A.into()], &VecStoreOptions::default())
        .await
        .unwrap();
    delete.assert_async().await;
}

/// Against a real Weaviate without a vectorizer module:
///
/// ```sh
/// docker run -d -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
///   -e DEFAULT_VECTORIZER_MODULE=none -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
///   -e CLUSTER_HOSTNAME=node1 cr.weaviate.io/semitechnologies/weaviate:1.32.4
/// WEAVIATE_TEST_URL=http://localhost:8080 cargo test --features weaviate -- --ignored weaviate_live
/// ```
#[tokio::test]
#[ignore = "needs WEAVIATE_TEST_URL"]
async fn weaviate_live_round_trip() {
    let url = std::env::var("WEAVIATE_TEST_URL").expect("WEAVIATE_TEST_URL");
    let class = format!("LcrTest{}", uuid::Uuid::new_v4().simple());
    let new_store = || {
        StoreBuilder::new()
            .url(&url)
            .class_name(&class)
            .embedder(embedder())
            .build()
            .unwrap()
    };
    let store = new_store();

    assert!(store
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap()
        .is_empty());

    let tenant = |ns: &str| VecStoreOptions::default().with_name_space(ns);
    store
        .add_documents(
            &[
                Document::new("alpha").with_metadata(
                    [
                        ("source".to_string(), json!("a.txt")),
                        ("tags".to_string(), json!(["x", "y"])),
                        ("nested".to_string(), json!({ "k": 1 })),
                    ]
                    .into(),
                ),
                Document::new("beta")
                    .with_metadata([("source".to_string(), json!("b.txt"))].into()),
            ],
            &tenant("tenant"),
        )
        .await
        .unwrap();
    // A prefix-sharing namespace must not leak into "tenant".
    new_store()
        .add_documents(&[Document::new("alpha")], &tenant("tenant-b"))
        .await
        .unwrap();

    let found = store
        .similarity_search("alpha", 5, &tenant("tenant"))
        .await
        .unwrap();
    assert_eq!(
        found
            .iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "beta"]
    );
    assert!((found[0].score - 1.0).abs() < 1e-4, "{}", found[0].score);
    assert!((found[1].score - 0.5).abs() < 1e-4, "{}", found[1].score);
    assert_eq!(found[0].metadata["source"], json!("a.txt"));
    assert_eq!(found[0].metadata["tags"], json!(["x", "y"]));
    assert_eq!(found[0].metadata["nested"], json!("{\"k\":1}"));
    assert_eq!(found[0].metadata["nameSpace"], json!("tenant"));

    let filtered = store
        .similarity_search(
            "alpha",
            5,
            &tenant("tenant").with_filters(
                json!({ "path": ["source"], "operator": "Equal", "valueText": "b.txt" }),
            ),
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

    let all = store
        .similarity_search("alpha", 5, &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(all.len(), 3);

    // Stopwords would be rejected as filter values with Weaviate's default preset.
    new_store()
        .add_documents(&[Document::new("beta")], &tenant("a"))
        .await
        .unwrap();
    let short = store
        .similarity_search("alpha", 5, &tenant("a"))
        .await
        .unwrap();
    assert_eq!(
        short
            .iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>(),
        ["beta"]
    );

    // Caller ids: writing one again replaces the object, and searches return it.
    let id = uuid::Uuid::new_v4().to_string();
    new_store()
        .add_documents(
            &[Document::new("alpha")
                .with_id(&id)
                .with_metadata([("source".to_string(), json!("old.txt"))].into())],
            &tenant("ids"),
        )
        .await
        .unwrap();
    let ids = new_store()
        .add_documents(&[Document::new("beta").with_id(&id)], &tenant("ids"))
        .await
        .unwrap();
    assert_eq!(ids, [id.as_str()]);
    let found = store
        .similarity_search("beta", 5, &tenant("ids"))
        .await
        .unwrap();
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].id.as_deref(), Some(id.as_str()));
    assert_eq!(found[0].page_content, "beta");
    assert_ne!(found[0].metadata.get("source"), Some(&json!("old.txt")));

    // get_by_ids / delete honour the namespace; unknown and non-UUID ids are ignored.
    let other = uuid::Uuid::new_v4().to_string();
    let ask = vec![
        other.clone(),
        "not-a-uuid".to_string(),
        id.to_uppercase(),
        ids[0].clone(),
    ];
    let got = store.get_by_ids(&ask, &tenant("ids")).await.unwrap();
    assert_eq!(got.len(), 1, "{got:?}");
    assert_eq!(got[0].id.as_deref(), Some(id.as_str()));
    assert_eq!(got[0].page_content, "beta");
    assert_eq!(got[0].score, 0.0);
    assert!(store
        .get_by_ids(&ask, &tenant("tenant"))
        .await
        .unwrap()
        .is_empty());
    store.delete(&ask, &tenant("tenant")).await.unwrap();
    assert_eq!(
        store.get_by_ids(&ask, &tenant("ids")).await.unwrap().len(),
        1
    );
    store.delete(&ask, &tenant("ids")).await.unwrap();
    assert!(store
        .get_by_ids(&ask, &tenant("ids"))
        .await
        .unwrap()
        .is_empty());
    assert_eq!(
        store
            .similarity_search("alpha", 5, &VecStoreOptions::default())
            .await
            .unwrap()
            .len(),
        4
    );

    reqwest::Client::new()
        .delete(format!(
            "{}/v1/schema/{}",
            url.trim_end_matches('/'),
            store.class_name
        ))
        .send()
        .await
        .unwrap();
}
