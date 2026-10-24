use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use mockito::{Matcher, Server, ServerGuard};
use serde_json::{json, Value};

use super::{
    store::{milvus_result, parse_schema, score, Schema, NAMESPACE_PARAM},
    *,
};
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

/// Returns one vector whatever the input size.
struct ShortEmbedder;

#[async_trait::async_trait]
impl crate::embedding::Embedder for ShortEmbedder {
    async fn embed_documents(
        &self,
        _: &[String],
    ) -> Result<Vec<Vec<f64>>, crate::embedding::EmbedderError> {
        Ok(vec![vec![1.0, 0.0]])
    }

    async fn embed_query(&self, _: &str) -> Result<Vec<f64>, crate::embedding::EmbedderError> {
        Ok(vec![1.0, 0.0])
    }
}

fn builder(server: &ServerGuard) -> StoreBuilder {
    StoreBuilder::new()
        .url(&format!("{}/", server.url()))
        .collection("docs")
        .embedder(embedder())
}

/// A `collections/describe` answer for the store's default schema.
fn describe(load: &str, metric: &str, auto_id: bool) -> Value {
    json!({
        "code": 0,
        "data": {
            "collectionName": "docs",
            "autoId": auto_id,
            "load": load,
            "fields": [
                { "name": "pk", "type": if auto_id { "Int64" } else { "VarChar" },
                  "primaryKey": true, "autoId": auto_id },
                { "name": "text", "type": "VarChar" },
                { "name": "vector", "type": "FloatVector" },
                { "name": "metadata", "type": "JSON" },
                { "name": "namespace", "type": "VarChar", "partitionKey": true },
                { "name": "year", "type": "Int64" },
            ],
            "indexes": [{ "fieldName": "vector", "metricType": metric }],
        },
    })
}

fn not_found() -> String {
    json!({ "code": 100, "message": "can't find collection[database=default][collection=docs]" })
        .to_string()
}

fn ok(data: Value) -> String {
    json!({ "code": 0, "data": data }).to_string()
}

fn body_of(request: &mockito::Request) -> Value {
    serde_json::from_slice(request.body().unwrap()).unwrap()
}

#[test]
fn build_validates() {
    let err = |b: StoreBuilder| b.build().err().unwrap().to_string();
    assert!(err(StoreBuilder::new().embedder(embedder())).contains("collection"));
    assert!(err(StoreBuilder::new().collection("docs")).contains("embedder"));
    let base = || StoreBuilder::new().collection("docs").embedder(embedder());
    for (b, expected) in [
        (base().collection("my-docs"), "invalid Milvus collection"),
        (base().text_field("1text"), "invalid Milvus text_field"),
        (base().vector_field(""), "invalid Milvus vector_field"),
        (
            base().metadata_field("a.b"),
            "invalid Milvus metadata_field",
        ),
        (
            base().namespace_field("a b"),
            "invalid Milvus namespace_field",
        ),
        (base().primary_field("$id"), "invalid Milvus primary_field"),
        (
            base().text_field("metadata"),
            "metadata_field and text_field",
        ),
        (
            base().namespace_field("pk"),
            "namespace_field and primary_field",
        ),
        (base().metric("HAMMING"), "unsupported Milvus metric"),
        (
            base().consistency_level("strong"),
            "unknown Milvus consistency",
        ),
        (base().max_text_length(0), "max_text_length"),
        (base().max_text_length(65_536), "max_text_length"),
    ] {
        let e = err(b);
        assert!(e.contains(expected), "{expected}: {e}");
    }
    assert!(
        base().metric("ip").build().is_ok(),
        "metrics are case-insensitive"
    );
    assert!(base().max_text_length(65_535).build().is_ok());
    assert!(base().collection(&"a".repeat(255)).build().is_ok());
    assert!(base().collection(&"a".repeat(256)).build().is_err());
}

#[test]
fn milvus_answers_and_scores() {
    assert!(milvus_result(json!({ "code": 0, "data": {} })).is_ok());
    let err = milvus_result(json!({ "code": 1100, "message": "bad filter" })).unwrap_err();
    assert_eq!((err.code, err.message.as_str()), (1100, "bad filter"));
    assert_eq!(err.to_string(), "Milvus error 1100: bad filter");
    let err = milvus_result(json!({ "unexpected": true })).unwrap_err();
    assert_eq!(err.code, -1);
    assert!(err.message.contains("unexpected"), "{err}");

    assert_eq!(score("COSINE", 0.8), 0.8);
    assert_eq!(score("IP", 3.0), 3.0);
    assert_eq!(score("L2", 0.4), -0.4);
}

#[test]
fn reads_the_schema() {
    let schema = |d: &Value| parse_schema(&d["data"], "text", "vector", "metadata", "namespace");
    assert_eq!(
        schema(&describe("LoadStateLoaded", "L2", false)).unwrap(),
        Schema {
            primary: "pk".into(),
            auto_id: false,
            has_namespace: true,
            has_metadata: true,
            output_fields: vec!["text".into(), "metadata".into(), "year".into()],
            metric: "L2".into(),
        }
    );
    let auto = schema(&describe("LoadStateLoaded", "ip", true)).unwrap();
    assert!(auto.auto_id);
    assert_eq!(auto.metric, "IP");
    // autoId may be reported on the collection or on the primary field alone.
    for (collection_flag, field_flag) in [(true, false), (false, true)] {
        let mut d = describe("LoadStateLoaded", "COSINE", true);
        d["data"]["autoId"] = json!(collection_flag);
        d["data"]["fields"][0]["autoId"] = json!(field_flag);
        assert!(
            schema(&d).unwrap().auto_id,
            "{collection_flag} {field_flag}"
        );
    }

    // A minimal collection: no namespace, no metadata, no index info.
    let minimal = json!({ "data": { "fields": [
        { "name": "id", "type": "VarChar", "primaryKey": true },
        { "name": "text", "type": "VarChar" },
        { "name": "vector", "type": "FloatVector" },
        { "name": "metadata", "type": "VarChar" },
    ]}});
    let minimal = schema(&minimal).unwrap();
    assert_eq!(minimal.primary, "id");
    assert!(!minimal.has_namespace && !minimal.has_metadata);
    assert_eq!(minimal.metric, "COSINE");

    let fields = |fields: Value| json!({ "data": { "fields": fields } });
    for (bad, expected) in [
        (json!({ "data": {} }), "no fields"),
        (
            fields(json!([{ "name": "text", "type": "VarChar" }])),
            "no primary key",
        ),
        (
            fields(json!([{ "name": "id", "type": "Int64", "primaryKey": true }])),
            "VarChar or an autoId key",
        ),
        (
            fields(json!([
                { "name": "id", "type": "VarChar", "primaryKey": true },
                { "name": "text", "type": "JSON" },
            ])),
            "no VarChar field \"text\"",
        ),
        (
            fields(json!([
                { "name": "id", "type": "VarChar", "primaryKey": true },
                { "name": "text", "type": "VarChar" },
                { "name": "vector", "type": "BinaryVector" },
            ])),
            "no FloatVector field \"vector\"",
        ),
    ] {
        let e = schema(&bad).unwrap_err().to_string();
        assert!(e.contains(expected), "{expected}: {e}");
    }
}

#[tokio::test]
async fn filters_combine_namespace_and_expressions() {
    let server = Server::new_async().await;
    let store = builder(&server).build().unwrap();
    let schema = parse_schema(
        &describe("LoadStateLoaded", "COSINE", false)["data"],
        "text",
        "vector",
        "metadata",
        "namespace",
    )
    .unwrap();
    let filter = |opt: MilvusOptions| store.filter(&opt, &schema);

    assert_eq!(
        filter(MilvusOptions::default()).unwrap(),
        (None, Default::default())
    );
    let (expr, params) =
        filter(MilvusOptions::default().with_filters(json!("metadata[\"source\"] == \"a\"")))
            .unwrap();
    assert_eq!(expr.as_deref(), Some("metadata[\"source\"] == \"a\""));
    assert!(params.is_empty());

    // The namespace goes through a template parameter, never into the expression text.
    let (expr, params) = filter(
        MilvusOptions::default()
            .with_name_space("t\" or 1==1")
            .with_filters(
                json!({ "expr": "year >= {min} or year < 0", "params": { "min": 2020 } }),
            ),
    )
    .unwrap();
    assert_eq!(
        expr.as_deref(),
        Some("namespace == {__lc_namespace} and (year >= {min} or year < 0)")
    );
    assert_eq!(
        Value::Object(params),
        json!({ "min": 2020, NAMESPACE_PARAM: "t\" or 1==1" })
    );

    let (_, params) = filter(
        MilvusOptions::default().with_filters(json!({ "expr": "year > 1", "params": null })),
    )
    .unwrap();
    assert!(params.is_empty());

    for (opt, expected) in [
        (
            MilvusOptions::default().with_filters(json!(3)),
            "expression string",
        ),
        (
            MilvusOptions::default().with_filters(json!({ "params": {} })),
            "`expr` string",
        ),
        (
            MilvusOptions::default().with_filters(json!({ "expr": "x", "params": [1] })),
            "must be an object",
        ),
        (
            MilvusOptions::default().with_filters(json!({ "expr": "x", "param": {} })),
            "unknown Milvus filter key \"param\"",
        ),
        (
            MilvusOptions::default()
                .with_name_space("a")
                .with_filters(json!({ "expr": "x", "params": { NAMESPACE_PARAM: 1 } })),
            "reserved",
        ),
    ] {
        let e = filter(opt).unwrap_err().to_string();
        assert!(e.contains(expected), "{expected}: {e}");
    }

    // The builder namespace is the default; options override it.
    let store = builder(&server).namespace("team").build().unwrap();
    let (_, params) = store.filter(&MilvusOptions::default(), &schema).unwrap();
    assert_eq!(params[NAMESPACE_PARAM], "team");
    let (_, params) = store
        .filter(&MilvusOptions::default().with_name_space("other"), &schema)
        .unwrap();
    assert_eq!(params[NAMESPACE_PARAM], "other");

    // Collections without a namespace field can't be scoped.
    let no_ns = Schema {
        has_namespace: false,
        ..schema
    };
    let e = store.filter(&MilvusOptions::default(), &no_ns).unwrap_err();
    assert!(e.to_string().contains("no \"namespace\" field"), "{e}");
}

#[tokio::test]
async fn create_body_uses_the_configured_names_and_dimension() {
    let server = Server::new_async().await;
    let store = builder(&server)
        .primary_field("id")
        .text_field("content")
        .vector_field("embedding")
        .metadata_field("meta")
        .namespace_field("tenant")
        .metric("L2")
        .build()
        .unwrap();
    assert_eq!(
        store.create_body(3),
        json!({
            "schema": {
                "autoId": false,
                "enableDynamicField": false,
                "fields": [
                    { "fieldName": "id", "dataType": "VarChar", "isPrimary": true,
                      "elementTypeParams": { "max_length": 64 } },
                    { "fieldName": "content", "dataType": "VarChar",
                      "elementTypeParams": { "max_length": 65535 } },
                    { "fieldName": "embedding", "dataType": "FloatVector",
                      "elementTypeParams": { "dim": 3 } },
                    { "fieldName": "meta", "dataType": "JSON" },
                    { "fieldName": "tenant", "dataType": "VarChar", "isPartitionKey": true,
                      "elementTypeParams": { "max_length": 256 } },
                ],
            },
            "indexParams": [{ "fieldName": "embedding", "indexName": "embedding",
                              "metricType": "L2", "indexType": "AUTOINDEX" }],
            "params": { "consistencyLevel": "Strong" },
        })
    );
}

#[tokio::test]
async fn rows_and_documents() {
    let server = Server::new_async().await;
    let store = builder(&server).build().unwrap();
    let schema = parse_schema(
        &describe("LoadStateLoaded", "L2", false)["data"],
        "text",
        "vector",
        "metadata",
        "namespace",
    )
    .unwrap();
    let doc = Document::new("hi").with_metadata([("source".to_string(), json!("a.txt"))].into());
    assert_eq!(
        store
            .row(&doc, &[0.5, 1.0], Some("id1"), Some("ns"), &schema)
            .unwrap(),
        json!({ "pk": "id1", "text": "hi", "vector": [0.5, 1.0],
                "metadata": { "source": "a.txt" }, "namespace": "ns" })
    );
    // autoId: no id; no namespace: "" (the partition key can't be null).
    assert_eq!(
        store.row(&doc, &[0.5], None, None, &schema).unwrap(),
        json!({ "text": "hi", "vector": [0.5], "metadata": { "source": "a.txt" }, "namespace": "" })
    );

    let bare = Schema {
        has_metadata: false,
        has_namespace: false,
        ..schema.clone()
    };
    assert_eq!(
        store
            .row(&Document::new("x"), &[1.0], Some("i"), None, &bare)
            .unwrap(),
        json!({ "pk": "i", "text": "x", "vector": [1.0] })
    );
    let e = store.row(&doc, &[1.0], None, None, &bare).unwrap_err();
    assert!(e.to_string().contains("no JSON field \"metadata\""), "{e}");
    let e = store
        .row(&Document::new("x"), &[1.0], None, Some("ns"), &bare)
        .unwrap_err();
    assert!(e.to_string().contains("for namespaces"), "{e}");

    let hit = |v: Value| v.as_object().unwrap().clone();
    let doc = store.to_document(
        hit(json!({ "distance": 0.25, "pk": "id1", "text": "hi",
                    "metadata": "{\"source\":\"a.txt\",\"n\":1}", "year": 2024, "other": null })),
        &schema,
    );
    assert_eq!(doc.page_content, "hi");
    assert_eq!(doc.id.as_deref(), Some("id1"));
    assert_eq!(doc.score, -0.25, "L2: -distance");
    assert_eq!(
        serde_json::to_value(&doc.metadata).unwrap(),
        json!({ "source": "a.txt", "n": 1, "year": 2024 })
    );

    // Odd shapes: object metadata, non-object JSON, invalid JSON, non-string text.
    for (metadata, expected) in [
        (json!({ "a": 1 }), json!({ "a": 1 })),
        (json!("[1,2]"), json!({ "metadata": [1, 2] })),
        (json!("not json"), json!({ "metadata": "not json" })),
        (json!(5), json!({ "metadata": 5 })),
        (Value::Null, json!({})),
    ] {
        let doc = store.to_document(
            hit(json!({ "distance": 1.0, "text": 7, "metadata": metadata })),
            &Schema {
                metric: "COSINE".into(),
                ..schema.clone()
            },
        );
        assert_eq!(serde_json::to_value(&doc.metadata).unwrap(), expected);
        assert_eq!(doc.page_content, "7");
        assert_eq!(doc.score, 1.0);
    }
    let doc = store.to_document(hit(json!({ "text": null })), &schema);
    assert_eq!((doc.page_content.as_str(), doc.score), ("", -0.0));
}

#[tokio::test]
async fn first_write_creates_the_collection_and_inserts_in_batches() {
    let mut server = Server::new_async().await;
    // Missing at first, then the freshly created (loaded) collection.
    let describes = Arc::new(AtomicUsize::new(0));
    let calls = describes.clone();
    let describe_mock = server
        .mock("POST", "/v2/vectordb/collections/describe")
        .match_header("authorization", "Bearer root:Milvus")
        .match_body(Matcher::Json(
            json!({ "collectionName": "docs", "dbName": "rag" }),
        ))
        .with_body_from_request(move |_| {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                not_found().into_bytes()
            } else {
                describe("LoadStateLoaded", "IP", false)
                    .to_string()
                    .into_bytes()
            }
        })
        .expect(2)
        .create_async()
        .await;
    let create = server
        .mock("POST", "/v2/vectordb/collections/create")
        .match_body(Matcher::PartialJson(json!({
            "collectionName": "docs",
            "dbName": "rag",
            "schema": {
                "autoId": false,
                "enableDynamicField": false,
                "fields": [
                    { "fieldName": "pk", "dataType": "VarChar", "isPrimary": true,
                      "elementTypeParams": { "max_length": 64 } },
                    { "fieldName": "text", "dataType": "VarChar",
                      "elementTypeParams": { "max_length": 1000 } },
                    { "fieldName": "vector", "dataType": "FloatVector",
                      "elementTypeParams": { "dim": 2 } },
                    { "fieldName": "metadata", "dataType": "JSON" },
                    { "fieldName": "namespace", "dataType": "VarChar", "isPartitionKey": true,
                      "elementTypeParams": { "max_length": 256 } },
                ],
            },
            "indexParams": [{ "fieldName": "vector", "indexName": "vector",
                              "metricType": "IP", "indexType": "AUTOINDEX" }],
            "params": { "consistencyLevel": "Bounded" },
        })))
        .with_body(ok(json!({})))
        .expect(1)
        .create_async()
        .await;
    let store = builder(&server)
        .token("root:Milvus")
        .db_name("rag")
        .metric("ip")
        .consistency_level("Bounded")
        .max_text_length(1000)
        .namespace("team")
        .build()
        .unwrap();

    let docs: Vec<Document> = (0..150)
        .map(|i| Document::new(if i % 2 == 0 { "alpha" } else { "beta" }))
        .collect();
    let rows = Arc::new(Mutex::new(Vec::<Value>::new()));
    let seen = rows.clone();
    let insert = server
        .mock("POST", "/v2/vectordb/entities/insert")
        .match_header("authorization", "Bearer root:Milvus")
        .match_body(Matcher::PartialJson(
            json!({ "collectionName": "docs", "dbName": "rag" }),
        ))
        .with_body_from_request(move |request| {
            let batch = body_of(request)["data"].as_array().unwrap().clone();
            let count = batch.len();
            seen.lock().unwrap().push(Value::Array(batch));
            ok(json!({ "insertCount": count })).into_bytes()
        })
        .expect(2)
        .create_async()
        .await;

    let ids = store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();
    describe_mock.assert_async().await;
    create.assert_async().await;
    insert.assert_async().await;
    assert_eq!(ids.len(), 150);
    assert!(ids.iter().all(|id| uuid::Uuid::parse_str(id).is_ok()));
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), 150);

    let batches = rows.lock().unwrap().clone();
    let sizes: Vec<_> = batches
        .iter()
        .map(|b| b.as_array().unwrap().len())
        .collect();
    assert_eq!(sizes, [100, 50]);
    let sent: Vec<&Value> = batches.iter().flat_map(|b| b.as_array().unwrap()).collect();
    for (i, (row, id)) in sent.iter().zip(&ids).enumerate() {
        let (text, vector) = if i % 2 == 0 {
            ("alpha", json!([1.0, 0.0]))
        } else {
            ("beta", json!([0.0, 1.0]))
        };
        assert_eq!(
            **row,
            json!({ "pk": id, "text": text, "vector": vector, "metadata": {}, "namespace": "team" }),
            "row {i}"
        );
    }

    // The schema is cached: the next write doesn't describe again.
    let _insert_again = server
        .mock("POST", "/v2/vectordb/entities/insert")
        .with_body(ok(json!({ "insertCount": 1 })))
        .create_async()
        .await;
    let no_describe = server
        .mock("POST", "/v2/vectordb/collections/describe")
        .expect(0)
        .create_async()
        .await;
    store
        .add_documents(&docs[..1], &VecStoreOptions::default())
        .await
        .unwrap();
    no_describe.assert_async().await;
}

#[tokio::test]
async fn auto_id_collections_return_milvus_ids_and_get_loaded() {
    let mut server = Server::new_async().await;
    let _describe = server
        .mock("POST", "/v2/vectordb/collections/describe")
        .with_body(describe("LoadStateNotLoad", "COSINE", true).to_string())
        .expect(1)
        .create_async()
        .await;
    let load = server
        .mock("POST", "/v2/vectordb/collections/load")
        .match_body(Matcher::Json(json!({ "collectionName": "docs" })))
        .with_body(ok(json!({})))
        .expect(1)
        .create_async()
        .await;
    let state = server
        .mock("POST", "/v2/vectordb/collections/get_load_state")
        .with_body(ok(
            json!({ "loadState": "LoadStateLoaded", "loadProgress": 100 }),
        ))
        .expect(1)
        .create_async()
        .await;
    let insert = server
        .mock("POST", "/v2/vectordb/entities/insert")
        .match_request(|request| {
            let rows = body_of(request)["data"].clone();
            rows.as_array()
                .unwrap()
                .iter()
                .all(|r| r.get("pk").is_none())
        })
        .with_body(ok(
            json!({ "insertCount": 2, "insertIds": [469936236673630413u64, "b"] }),
        ))
        .expect(1)
        .create_async()
        .await;
    let store = builder(&server).create_collection(false).build().unwrap();
    let ids = store
        .add_documents(
            &[Document::new("alpha"), Document::new("beta")],
            &VecStoreOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(ids, ["469936236673630413", "b"]);
    load.assert_async().await;
    state.assert_async().await;
    insert.assert_async().await;

    // A short id list is an error.
    let _short = server
        .mock("POST", "/v2/vectordb/entities/insert")
        .with_body(ok(json!({ "insertCount": 1, "insertIds": [1] })))
        .create_async()
        .await;
    let err = store
        .add_documents(
            &[Document::new("alpha"), Document::new("beta")],
            &VecStoreOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("1 ids for 2 rows"), "{err}");

    // autoId keys can't take caller ids; nothing is written.
    let no_write = server
        .mock("POST", Matcher::Regex("entities/(insert|upsert)".into()))
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
    assert!(err.to_string().contains("autoId"), "{err}");
    no_write.assert_async().await;
}

#[tokio::test]
async fn caller_ids_are_upserted_as_primary_keys() {
    let mut server = Server::new_async().await;
    let _describe = server
        .mock("POST", "/v2/vectordb/collections/describe")
        .with_body(describe("LoadStateLoaded", "COSINE", false).to_string())
        .create_async()
        .await;
    let insert = server
        .mock("POST", "/v2/vectordb/entities/insert")
        .expect(0)
        .create_async()
        .await;
    let upsert = server
        .mock("POST", "/v2/vectordb/entities/upsert")
        .match_request(|request| {
            let rows = body_of(request)["data"].clone();
            let rows = rows.as_array().unwrap();
            rows.len() == 2
                && rows[0]["pk"] == json!("doc-1")
                && rows[1]["pk"]
                    .as_str()
                    .is_some_and(|id| uuid::Uuid::parse_str(id).is_ok())
        })
        .with_body(ok(json!({ "upsertCount": 2 })))
        .expect(1)
        .create_async()
        .await;
    let store = builder(&server).create_collection(false).build().unwrap();
    let ids = store
        .add_documents(
            &[
                Document::new("alpha").with_id("doc-1"),
                Document::new("beta"),
            ],
            &VecStoreOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(ids[0], "doc-1");
    assert!(uuid::Uuid::parse_str(&ids[1]).is_ok(), "{ids:?}");
    upsert.assert_async().await;
    insert.assert_async().await;

    let err = store
        .add_documents(
            &[Document::new("alpha").with_id("")],
            &VecStoreOptions::default().with_embedder(crate::test_utils::PanickingEmbedder),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("must not be empty"), "{err}");
}

#[tokio::test(start_paused = true)]
async fn loading_is_polled_until_done_or_timeout() {
    let mut server = Server::new_async().await;
    let _describe = server
        .mock("POST", "/v2/vectordb/collections/describe")
        .with_body(describe("LoadStateLoading", "COSINE", false).to_string())
        .create_async()
        .await;
    let _load = server
        .mock("POST", "/v2/vectordb/collections/load")
        .with_body(ok(json!({})))
        .create_async()
        .await;
    let loading = server
        .mock("POST", "/v2/vectordb/collections/get_load_state")
        .with_body(ok(
            json!({ "loadState": "LoadStateLoading", "loadProgress": 40 }),
        ))
        .expect_at_least(2)
        .create_async()
        .await;
    let store = builder(&server)
        .load_timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    let err = store
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("wasn't loaded after 1s"), "{err}");
    assert!(err.to_string().contains("40"), "{err}");
    loading.assert_async().await;
    assert!(store.schema.get().is_none(), "nothing cached on failure");
}

#[tokio::test]
async fn search_builds_the_request_and_maps_hits() {
    let mut server = Server::new_async().await;
    let _describe = server
        .mock("POST", "/v2/vectordb/collections/describe")
        .with_body(describe("LoadStateLoaded", "COSINE", false).to_string())
        .expect(1)
        .create_async()
        .await;
    let search = server
        .mock("POST", "/v2/vectordb/entities/search")
        .match_body(Matcher::Json(json!({
            "collectionName": "docs",
            "data": [[0.6, 0.8]],
            "annsField": "vector",
            "limit": 3,
            "outputFields": ["text", "metadata", "year"],
            "filter": "namespace == {__lc_namespace} and (year > {y})",
            "exprParams": { "y": 2000, NAMESPACE_PARAM: "team" },
        })))
        .with_body(ok(json!([
            { "distance": 0.9, "pk": "a", "text": "alpha", "metadata": "{\"source\":\"a\"}", "year": 2024 },
            { "distance": 0.5, "pk": "b", "text": "beta", "metadata": "{}", "year": 2001 },
            "garbage",
        ])))
        .expect(3)
        .create_async()
        .await;
    let store = builder(&server).namespace("team").build().unwrap();
    let options = || {
        VecStoreOptions::default()
            .with_filters(json!({ "expr": "year > {y}", "params": { "y": 2000 } }))
    };
    let opt = options();
    let docs = store.similarity_search("query", 3, &opt).await.unwrap();
    assert_eq!(docs.len(), 2);
    assert_eq!(
        (docs[0].page_content.as_str(), docs[0].score),
        ("alpha", 0.9)
    );
    assert_eq!(docs[0].metadata["source"], "a");
    assert_eq!(docs[0].metadata["year"], 2024);
    assert!(!docs[0].metadata.contains_key("pk"));

    let docs = store
        .similarity_search("query", 3, &options().with_score_threshold(0.6))
        .await
        .unwrap();
    assert_eq!(docs.len(), 1, "only 0.9 passes 0.6");
    // Inclusive: a score equal to the threshold stays.
    let docs = store
        .similarity_search("query", 3, &options().with_score_threshold(0.5))
        .await
        .unwrap();
    assert_eq!(docs.len(), 2);
    search.assert_async().await;

    // limit 0 never reaches Milvus (it rejects topk 0).
    let store = builder(&server)
        .embedder(PanickingEmbedder)
        .build()
        .unwrap();
    assert!(store
        .similarity_search("q", 0, &VecStoreOptions::default())
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn search_without_filters_sends_none() {
    let mut server = Server::new_async().await;
    let _describe = server
        .mock("POST", "/v2/vectordb/collections/describe")
        .with_body(describe("LoadStateLoaded", "L2", false).to_string())
        .create_async()
        .await;
    let search = server
        .mock("POST", "/v2/vectordb/entities/search")
        .match_request(|request| {
            let body = body_of(request);
            body.get("filter").is_none() && body.get("exprParams").is_none()
        })
        .with_body(ok(json!([{ "distance": 0.4, "text": "beta" }])))
        .expect(1)
        .create_async()
        .await;
    let store = builder(&server).build().unwrap();
    let docs = store
        .similarity_search("query", 1, &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(docs[0].score, -0.4);
    search.assert_async().await;

    // A plain filter string has no exprParams either.
    let search = server
        .mock("POST", "/v2/vectordb/entities/search")
        .match_request(|request| {
            let body = body_of(request);
            body["filter"] == json!("year > 1") && body.get("exprParams").is_none()
        })
        .with_body(ok(json!([])))
        .expect(1)
        .create_async()
        .await;
    store
        .similarity_search(
            "query",
            1,
            &VecStoreOptions::default().with_filters(json!("year > 1")),
        )
        .await
        .unwrap();
    search.assert_async().await;
}

#[tokio::test]
async fn missing_collections_and_errors() {
    let mut server = Server::new_async().await;
    let _describe = server
        .mock("POST", "/v2/vectordb/collections/describe")
        .with_body(not_found())
        .create_async()
        .await;
    let store = builder(&server).build().unwrap();
    // Search before any write: nothing, no error.
    assert!(store
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap()
        .is_empty());

    // Writes don't create it when create_collection is off.
    let store = builder(&server).create_collection(false).build().unwrap();
    let err = store
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("create_collection is off"),
        "{err}"
    );

    // Empty input never calls the embedder or Milvus.
    let store = builder(&server)
        .embedder(PanickingEmbedder)
        .build()
        .unwrap();
    assert!(store
        .add_documents(&[], &VecStoreOptions::default())
        .await
        .unwrap()
        .is_empty());

    // Other Milvus errors pass through.
    let mut server = Server::new_async().await;
    let _describe = server
        .mock("POST", "/v2/vectordb/collections/describe")
        .with_body(json!({ "code": 800, "message": "database not found[database=x]" }).to_string())
        .create_async()
        .await;
    let store = builder(&server).build().unwrap();
    let err = store
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "Milvus error 800: database not found[database=x]"
    );
    assert_eq!(err.downcast_ref::<MilvusError>().unwrap().code, 800);

    // HTTP errors and invalid bodies.
    let mut server = Server::new_async().await;
    let _describe = server
        .mock("POST", "/v2/vectordb/collections/describe")
        .with_status(401)
        .with_body("unauthorized")
        .create_async()
        .await;
    let err = builder(&server)
        .build()
        .unwrap()
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("401") && err.to_string().contains("unauthorized"),
        "{err}"
    );

    let mut server = Server::new_async().await;
    let _describe = server
        .mock("POST", "/v2/vectordb/collections/describe")
        .with_body("<html>")
        .create_async()
        .await;
    let err = builder(&server)
        .build()
        .unwrap()
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("invalid Milvus answer"), "{err}");

    // An embedder returning the wrong number of vectors.
    let short = ShortEmbedder;
    let store = builder(&server).embedder(short).build().unwrap();
    let err = store
        .add_documents(
            &[Document::new("a"), Document::new("b")],
            &VecStoreOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("1 vectors for 2 documents"),
        "{err}"
    );
}

fn live_url() -> String {
    std::env::var("MILVUS_TEST_URL").expect("set MILVUS_TEST_URL")
}

/// Needs Milvus 2.4+, e.g. the standalone container from
/// <https://milvus.io/docs/install_standalone-docker.md>, then
/// `MILVUS_TEST_URL=http://localhost:19530 cargo test --features milvus -- --ignored milvus_live`.
#[tokio::test]
#[ignore]
async fn milvus_live_round_trip() {
    let suffix = format!(
        "{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let collection = format!("lcr_{suffix}");
    let live = |c: &str| {
        StoreBuilder::new()
            .url(&live_url())
            .token(&std::env::var("MILVUS_TEST_TOKEN").unwrap_or("root:Milvus".into()))
            .collection(c)
            .embedder(embedder())
    };
    let store = live(&collection).build().unwrap();
    let opt = VecStoreOptions::default();

    // Nothing yet.
    assert!(store
        .similarity_search("query", 2, &opt)
        .await
        .unwrap()
        .is_empty());

    // Two stores racing to create the same collection both succeed.
    let other = live(&collection).build().unwrap();
    let first =
        [Document::new("alpha").with_metadata([("source".to_string(), json!("a.txt"))].into())];
    let second = [Document::new("beta").with_metadata(
        [
            ("source".to_string(), json!("b.txt")),
            ("tags".to_string(), json!(["x", { "y": null }])),
        ]
        .into(),
    )];
    let (a, b) = tokio::join!(
        store.add_documents(&first, &opt),
        other.add_documents(&second, &opt),
    );
    assert_eq!(a.unwrap().len(), 1);
    assert_eq!(b.unwrap().len(), 1);

    // Strong consistency: visible right away, cosine scores, metadata round-trips.
    let docs = store.similarity_search("query", 5, &opt).await.unwrap();
    assert_eq!(
        docs.iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>(),
        ["beta", "alpha"]
    );
    assert!((docs[0].score - 0.8).abs() < 1e-5, "{}", docs[0].score);
    assert_eq!(docs[0].metadata["tags"], json!(["x", { "y": null }]));
    assert_eq!(docs[1].metadata["source"], "a.txt");
    assert!(!docs[0].metadata.contains_key("namespace"));

    // Filters: strings and templates; bad expressions are Milvus errors.
    let filtered = store
        .similarity_search(
            "query",
            5,
            &VecStoreOptions::default().with_filters(json!("metadata[\"source\"] == \"a.txt\"")),
        )
        .await
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].page_content, "alpha");
    let filtered = store
        .similarity_search(
            "query",
            5,
            &VecStoreOptions::default().with_filters(
                json!({ "expr": "metadata[\"source\"] in {s}", "params": { "s": ["b.txt"] } }),
            ),
        )
        .await
        .unwrap();
    assert_eq!(filtered[0].page_content, "beta");
    let err = store
        .similarity_search(
            "query",
            5,
            &VecStoreOptions::default().with_filters(json!("metadata[ ==")),
        )
        .await
        .unwrap_err();
    assert!(err.downcast_ref::<MilvusError>().is_some(), "{err}");

    // Namespaces: exact matches only, and quotes can't escape the template.
    for ns in ["tenant", "tenant-b", "t\" or namespace != \""] {
        store
            .add_documents(
                &[Document::new("alpha").with_metadata([("ns".to_string(), json!(ns))].into())],
                &VecStoreOptions::default().with_name_space(ns),
            )
            .await
            .unwrap();
    }
    for ns in ["tenant", "tenant-b", "t\" or namespace != \""] {
        let hits = store
            .similarity_search("query", 10, &VecStoreOptions::default().with_name_space(ns))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "{ns}");
        assert_eq!(hits[0].metadata["ns"], ns);
    }
    // The builder namespace scopes by default, and combines with filters.
    let scoped = live(&collection).namespace("tenant").build().unwrap();
    let hits = scoped
        .similarity_search(
            "query",
            10,
            &VecStoreOptions::default().with_filters(json!("metadata[\"ns\"] == \"tenant\"")),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    let hits = scoped
        .similarity_search(
            "query",
            10,
            &VecStoreOptions::default().with_filters(json!("metadata[\"ns\"] == \"tenant-b\"")),
        )
        .await
        .unwrap();
    assert!(hits.is_empty());
    // Unscoped searches see everything.
    assert_eq!(
        store
            .similarity_search("query", 10, &opt)
            .await
            .unwrap()
            .len(),
        5
    );
    // Thresholds.
    let hits = store
        .similarity_search(
            "query",
            10,
            &VecStoreOptions::default().with_score_threshold(0.7),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].page_content, "beta");
    // Per-call embedder.
    let flipped = FakeEmbedder::new(2, &[("query", vec![1.0, 0.0])]);
    let hits = store
        .similarity_search(
            "query",
            1,
            &VecStoreOptions::default().with_embedder(flipped),
        )
        .await
        .unwrap();
    assert_eq!(hits[0].page_content, "alpha");

    // Caller ids are upserted: writing one again replaces the row, and searches return it.
    let ids_ns = VecStoreOptions::default().with_name_space("ids");
    store
        .add_documents(
            &[Document::new("alpha")
                .with_id("doc-1")
                .with_metadata([("old".to_string(), json!(true))].into())],
            &ids_ns,
        )
        .await
        .unwrap();
    let ids = store
        .add_documents(&[Document::new("beta").with_id("doc-1")], &ids_ns)
        .await
        .unwrap();
    assert_eq!(ids, ["doc-1"]);
    let hits = store.similarity_search("query", 5, &ids_ns).await.unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0].id.as_deref(), Some("doc-1"));
    assert_eq!(hits[0].page_content, "beta");
    assert!(!hits[0].metadata.contains_key("old"), "{hits:?}");

    // An existing L2 collection (created elsewhere, autoId, not loaded) works as is.
    let client = reqwest::Client::new();
    let post = |path: &str, body: Value| {
        let request = client
            .post(format!("{}/v2/vectordb/{path}", live_url()))
            .bearer_auth("root:Milvus")
            .json(&body);
        async move { request.send().await.unwrap().json::<Value>().await.unwrap() }
    };
    let l2 = format!("lcr_l2_{suffix}");
    let created = post(
        "collections/create",
        json!({
            "collectionName": l2,
            "schema": { "autoId": true, "enableDynamicField": false, "fields": [
                { "fieldName": "id", "dataType": "Int64", "isPrimary": true, "autoID": true },
                { "fieldName": "text", "dataType": "VarChar", "elementTypeParams": { "max_length": 100 } },
                { "fieldName": "vector", "dataType": "FloatVector", "elementTypeParams": { "dim": 2 } },
                { "fieldName": "page", "dataType": "Int64", "nullable": true },
            ]},
            "indexParams": [{ "fieldName": "vector", "indexName": "vector",
                              "metricType": "L2", "indexType": "AUTOINDEX" }],
            "params": { "consistencyLevel": "Strong" },
        }),
    )
    .await;
    assert_eq!(created["code"], 0, "{created}");
    assert_eq!(
        post("collections/release", json!({ "collectionName": l2 })).await["code"],
        0
    );
    let l2_store = live(&l2).create_collection(false).build().unwrap();
    let ids = l2_store
        .add_documents(&[Document::new("alpha"), Document::new("beta")], &opt)
        .await
        .unwrap();
    assert_eq!(ids.len(), 2);
    assert!(ids.iter().all(|id| id.parse::<i64>().is_ok()), "{ids:?}");
    let hits = l2_store.similarity_search("query", 2, &opt).await.unwrap();
    assert_eq!(hits[0].page_content, "beta");
    assert!((hits[0].score - -0.4).abs() < 1e-5, "{}", hits[0].score);
    assert!(hits[0].score > hits[1].score);
    // No metadata/namespace fields: those writes fail clearly.
    let err = l2_store
        .add_documents(
            &[Document::new("alpha").with_metadata([("k".to_string(), json!(1))].into())],
            &opt,
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("no JSON field"), "{err}");
    let err = l2_store
        .add_documents(&[Document::new("alpha").with_id("doc-1")], &opt)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("autoId"), "{err}");

    // A collection with a different dimension is a clear Milvus error.
    let wide = live(&collection)
        .embedder(FakeEmbedder::new(3, &[("alpha", vec![1.0, 0.0, 0.0])]))
        .build()
        .unwrap();
    let err = wide
        .add_documents(&[Document::new("alpha")], &opt)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("dimension"), "{err}");

    for c in [&collection, &l2] {
        assert_eq!(
            post("collections/drop", json!({ "collectionName": c })).await["code"],
            0
        );
    }
}
