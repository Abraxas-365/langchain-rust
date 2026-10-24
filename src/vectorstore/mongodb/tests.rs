use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use futures::TryStreamExt;
use mongodb::{
    bson::{doc, oid::ObjectId, Bson, DateTime, Document as BsonDocument},
    options::ClientOptions,
    Client, Collection,
};
use serde_json::{json, Value};

use super::{
    store::{num_candidates, wait_until_queryable},
    *,
};
use crate::{
    schemas::Document,
    test_utils::FakeEmbedder,
    vectorstore::{VecStoreOptions, VectorStore},
};

/// A collection handle that never connects (nothing here runs an operation).
fn offline_collection() -> Collection<BsonDocument> {
    let options = ClientOptions::builder().build();
    Client::with_options(options)
        .unwrap()
        .database("app")
        .collection("docs")
}

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

fn store(builder: StoreBuilder) -> Store {
    builder
        .embedder(embedder())
        .build_with_collection(offline_collection())
        .unwrap()
}

#[tokio::test]
async fn builder_validates_fields() {
    let err = |b: StoreBuilder| {
        b.build_with_collection(offline_collection())
            .err()
            .unwrap()
            .to_string()
    };
    assert_eq!(err(StoreBuilder::new()), "embedder is required");
    for (bad, expected) in [
        (StoreBuilder::new().text_key(""), "invalid"),
        (StoreBuilder::new().embedding_key("$vec"), "invalid"),
        (StoreBuilder::new().metadata_key("meta.data"), "invalid"),
        (StoreBuilder::new().namespace_key("_id"), "invalid"),
        (StoreBuilder::new().text_key("embedding"), "used twice"),
        (
            StoreBuilder::new().metadata_key("__vector_score"),
            "used twice",
        ),
    ] {
        let e = err(bad.embedder(embedder()));
        assert!(e.contains(expected), "{e}");
    }
    let e = StoreBuilder::new()
        .collection("docs")
        .embedder(embedder())
        .build()
        .await
        .err()
        .unwrap();
    assert_eq!(e.to_string(), "database is required");
    let e = StoreBuilder::new()
        .database("app")
        .embedder(embedder())
        .build()
        .await
        .err()
        .unwrap();
    assert_eq!(e.to_string(), "collection is required");

    let store = store(StoreBuilder::new().index_name("my_index"));
    assert_eq!(store.index_name(), "my_index");
    assert_eq!(store.collection().name(), "docs");
    assert_eq!(
        self::store(StoreBuilder::new()).index_name(),
        "vector_index"
    );
}

#[tokio::test]
async fn index_definition_has_vector_namespace_and_filter_paths() {
    let store = store(
        StoreBuilder::new()
            .embedding_key("vec")
            .namespace_key("tenant")
            .filter_paths(["metadata.source", "metadata.year"])
            .similarity(MongoSimilarity::DotProduct),
    );
    assert_eq!(
        store.index_definition(3),
        doc! { "fields": [
            { "type": "vector", "path": "vec", "numDimensions": 3_i64, "similarity": "dotProduct" },
            { "type": "filter", "path": "tenant" },
            { "type": "filter", "path": "metadata.source" },
            { "type": "filter", "path": "metadata.year" },
        ] }
    );
    assert_eq!(MongoSimilarity::default().as_str(), "cosine");
    assert_eq!(MongoSimilarity::Euclidean.as_str(), "euclidean");
}

#[tokio::test]
async fn documents_are_stored_with_typed_metadata() {
    let store = store(StoreBuilder::new().text_key("body"));
    let doc = Document::new("alpha").with_metadata(
        [
            ("source".to_string(), json!("a.txt")),
            ("year".to_string(), json!(2024)),
            ("big".to_string(), json!(5_000_000_000_i64)),
            ("ratio".to_string(), json!(0.5)),
            ("tags".to_string(), json!(["x", 1])),
            ("nested".to_string(), json!({ "a": true, "b": null })),
        ]
        .into(),
    );
    assert_eq!(
        store.to_bson(&doc, &[1.0, -0.5], Some("tenant-a")).unwrap(),
        doc! {
            "body": "alpha",
            "embedding": [1.0, -0.5],
            "namespace": "tenant-a",
            "metadata": {
                "big": 5_000_000_000_i64,
                "nested": { "a": true, "b": Bson::Null },
                "ratio": 0.5,
                "source": "a.txt",
                "tags": ["x", 1],
                "year": 2024,
            },
        }
    );
    // Deterministic field order (HashMap iteration isn't).
    let stored = store.to_bson(&doc, &[1.0], None).unwrap();
    assert_eq!(
        stored
            .get_document("metadata")
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        ["big", "nested", "ratio", "source", "tags", "year"]
    );
    assert_eq!(
        stored.keys().collect::<Vec<_>>(),
        ["body", "embedding", "metadata"]
    );
    assert_eq!(
        store
            .to_bson(&Document::new("beta"), &[0.0, 1.0], None)
            .unwrap(),
        doc! { "body": "beta", "embedding": [0.0, 1.0], "metadata": {} },
        "no namespace field without a namespace"
    );
    // Extended-JSON-looking metadata that isn't valid is rejected up front.
    let bad = Document::new("x")
        .with_metadata([("when".to_string(), json!({ "$date": "not a date" }))].into());
    let err = store.to_bson(&bad, &[0.0, 1.0], None).unwrap_err();
    assert!(err.to_string().contains("\"when\""), "{err}");
}

#[tokio::test]
async fn filters_combine_namespace_and_mql() {
    let store = store(StoreBuilder::new());
    assert_eq!(store.filter(None, None).unwrap(), None);
    assert_eq!(store.filter(None, Some(&json!(null))).unwrap(), None);
    assert_eq!(store.filter(None, Some(&json!({}))).unwrap(), None);
    assert_eq!(
        store.filter(Some("t-a"), None).unwrap(),
        Some(doc! { "namespace": { "$eq": "t-a" } })
    );
    assert_eq!(
        store
            .filter(None, Some(&json!({ "metadata.year": { "$gte": 2020 } })))
            .unwrap(),
        Some(doc! { "metadata.year": { "$gte": 2020 } })
    );
    assert_eq!(
        store
            .filter(Some("t-a"), Some(&json!({ "metadata.source": "a.txt" })))
            .unwrap(),
        Some(doc! { "$and": [
            { "namespace": { "$eq": "t-a" } },
            { "metadata.source": "a.txt" },
        ] })
    );
    for bad in [json!("metadata.year > 3"), json!([1]), json!(3)] {
        let err = store.filter(None, Some(&bad)).unwrap_err();
        assert!(err.to_string().contains("MQL object"), "{bad}: {err}");
    }
    let err = store
        .filter(None, Some(&json!({ "d": { "$date": "nope" } })))
        .unwrap_err();
    assert!(err.to_string().contains("invalid MongoDB filter"), "{err}");
}

#[tokio::test]
async fn pipeline_searches_projects_and_scores() {
    let store = store(
        StoreBuilder::new()
            .index_name("vi")
            .embedding_key("vec")
            .num_candidates_multiplier(20),
    );
    assert_eq!(
        store.pipeline(&[0.5, 1.0], 3, Some(doc! { "namespace": { "$eq": "n" } })),
        vec![
            doc! { "$vectorSearch": {
                "index": "vi",
                "path": "vec",
                "queryVector": [0.5, 1.0],
                "numCandidates": 60_i64,
                "limit": 3_i64,
                "filter": { "namespace": { "$eq": "n" } },
            } },
            doc! { "$project": { "vec": 0, "__vector_score": { "$meta": "vectorSearchScore" } } },
        ]
    );
    let without_filter = store.pipeline(&[0.5, 1.0], 3, None);
    assert!(!without_filter[0]
        .get_document("$vectorSearch")
        .unwrap()
        .contains_key("filter"));

    assert_eq!(num_candidates(4, 10), 40);
    assert_eq!(num_candidates(5000, 10), 10_000, "Atlas caps candidates");
    assert_eq!(num_candidates(20_000, 10), 20_000, "never below limit");
    assert_eq!(num_candidates(usize::MAX, 10), usize::MAX);
    let candidates = |b: StoreBuilder| {
        self::store(b).pipeline(&[1.0], 7, None)[0]
            .get_document("$vectorSearch")
            .unwrap()
            .get_i64("numCandidates")
            .unwrap()
    };
    assert_eq!(candidates(StoreBuilder::new()), 70, "default multiplier 10");
    assert_eq!(
        candidates(StoreBuilder::new().num_candidates_multiplier(0)),
        7,
        "never fewer candidates than limit"
    );
}

#[tokio::test]
async fn results_become_documents() {
    let store = store(StoreBuilder::new());
    let id = ObjectId::new();
    let when = DateTime::from_millis(1_700_000_000_000);
    let doc = store
        .to_document(doc! {
            "_id": id,
            "text": "alpha",
            "namespace": "t-a",
            "metadata": {
                "source": "a.txt",
                "year": 2024,
                "big": 5_000_000_000_i64,
                "ref": id,
                "when": when,
                "nested": { "ids": [id] },
            },
            "__vector_score": 0.875,
        })
        .unwrap();
    assert_eq!(doc.page_content, "alpha");
    assert_eq!(doc.score, 0.875);
    assert_eq!(doc.id, Some(id.to_hex()));
    assert_eq!(
        doc.metadata,
        HashMap::<String, Value>::from([
            ("source".to_string(), json!("a.txt")),
            ("year".to_string(), json!(2024)),
            ("big".to_string(), json!(5_000_000_000_i64)),
            ("ref".to_string(), json!(id.to_hex())),
            ("when".to_string(), json!("2023-11-14T22:13:20Z")),
            ("nested".to_string(), json!({ "ids": [id.to_hex()] })),
        ])
    );

    let bare = store
        .to_document(doc! { "__vector_score": 1_i32, "text": "beta" })
        .unwrap();
    assert_eq!(bare.score, 1.0);
    assert!(bare.metadata.is_empty());
    let empty = store.to_document(doc! { "__vector_score": 2_i64 }).unwrap();
    assert_eq!(empty.page_content, "");
    assert_eq!(empty.score, 2.0);

    let err = store.to_document(doc! { "text": "alpha" }).unwrap_err();
    assert!(err.to_string().contains("without a score"), "{err}");
    let err = store
        .to_document(doc! { "text": 3, "__vector_score": 0.5 })
        .unwrap_err();
    assert!(err.to_string().contains("not a string"), "{err}");

    let stored = store
        .stored_document(doc! { "_id": "doc-1", "text": "beta", "embedding": [1.0] })
        .unwrap();
    assert_eq!(stored.id.as_deref(), Some("doc-1"));
    assert_eq!(stored.page_content, "beta");
    assert_eq!(stored.score, 0.0);
    assert!(stored.metadata.is_empty());
}

#[tokio::test]
async fn id_filters_match_string_and_object_ids_within_the_scope() {
    let store = store(StoreBuilder::new());
    let oid = ObjectId::new();
    let ids = vec!["doc-1".to_string(), oid.to_hex()];
    assert_eq!(
        store.id_filter(&ids, None, None).unwrap(),
        doc! { "_id": { "$in": ["doc-1", oid.to_hex(), oid] } }
    );
    assert_eq!(
        store
            .id_filter(
                &ids[..1],
                Some("t-a"),
                Some(&json!({ "metadata.year": 2024 }))
            )
            .unwrap(),
        doc! { "$and": [
            { "_id": { "$in": ["doc-1"] } },
            { "$and": [
                { "namespace": { "$eq": "t-a" } },
                { "metadata.year": 2024 },
            ] },
        ] }
    );
    assert_eq!(
        store.id_filter(&ids[..1], Some("t-a"), None).unwrap(),
        doc! { "$and": [
            { "_id": { "$in": ["doc-1"] } },
            { "namespace": { "$eq": "t-a" } },
        ] }
    );
    assert!(store.id_filter(&ids, None, Some(&json!("nope"))).is_err());

    // Empty ids and bad filters never reach the (unreachable) server.
    let opt = VecStoreOptions::default();
    store.delete(&[], &opt).await.unwrap();
    assert!(store.get_by_ids(&[], &opt).await.unwrap().is_empty());
    let bad = VecStoreOptions::default().with_filters(json!(1));
    assert!(store.delete(&ids, &bad).await.is_err());
    assert!(store.get_by_ids(&ids, &bad).await.is_err());
}

#[tokio::test]
async fn empty_input_zero_limit_and_bad_filters_skip_the_network() {
    // The collection points at a server that doesn't exist: any operation would fail.
    let store = store(StoreBuilder::new());
    let opt = VecStoreOptions::default();
    assert!(store.add_documents(&[], &opt).await.unwrap().is_empty());
    assert!(store
        .similarity_search("query", 0, &opt)
        .await
        .unwrap()
        .is_empty());
    let err = store
        .similarity_search(
            "query",
            3,
            &VecStoreOptions::default().with_filters(json!("nope")),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("MQL object"), "{err}");
    let err = store
        .add_documents(
            &[Document::new("x")
                .with_metadata([("d".to_string(), json!({ "$date": "nope" }))].into())],
            &opt,
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not valid BSON"), "{err}");
    let err = store
        .add_documents(
            &[Document::new("alpha"), Document::new("beta").with_id("")],
            &VecStoreOptions::default().with_embedder(crate::test_utils::PanickingEmbedder),
        )
        .await
        .unwrap_err();
    assert_eq!(err.to_string(), "document id must not be empty");

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
    let err = store
        .add_documents(
            &[Document::new("alpha"), Document::new("beta")],
            &VecStoreOptions::default().with_embedder(ShortEmbedder),
        )
        .await
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "embedder returned 1 vectors for 2 documents"
    );
}

#[tokio::test(start_paused = true)]
async fn waiting_for_the_index() {
    let status = |polls: Arc<AtomicUsize>, ready_at: usize, gone: bool| {
        move || {
            let n = polls.fetch_add(1, Ordering::SeqCst) + 1;
            async move {
                if gone {
                    return Ok(None);
                }
                Ok(Some(n >= ready_at))
            }
        }
    };
    // Becomes queryable on the third poll.
    let polls = Arc::new(AtomicUsize::new(0));
    let started = tokio::time::Instant::now();
    wait_until_queryable(
        status(polls.clone(), 3, false),
        Duration::from_millis(500),
        Duration::from_secs(10),
    )
    .await
    .unwrap();
    assert_eq!(polls.load(Ordering::SeqCst), 3);
    assert_eq!(started.elapsed(), Duration::from_secs(1));

    let polls = Arc::new(AtomicUsize::new(0));
    let err = wait_until_queryable(
        status(polls.clone(), usize::MAX, false),
        Duration::from_millis(500),
        Duration::from_secs(2),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("not queryable after 2s"), "{err}");
    // Polls at 0, 0.5, 1, 1.5 and 2s, then gives up instead of sleeping past the deadline.
    assert_eq!(polls.load(Ordering::SeqCst), 5);

    let err = wait_until_queryable(
        status(Arc::new(AtomicUsize::new(0)), 1, true),
        Duration::from_millis(500),
        Duration::from_secs(2),
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("disappeared"), "{err}");

    let err = wait_until_queryable(
        || async { Err::<Option<bool>, _>("boom".into()) },
        Duration::from_millis(500),
        Duration::from_secs(2),
    )
    .await
    .unwrap_err();
    assert_eq!(err.to_string(), "boom");
}

/// Against MongoDB Atlas or the local Atlas image:
///
/// ```sh
/// docker run -d -p 27017:27017 mongodb/mongodb-atlas-local:8
/// MONGODB_VECTOR_TEST_URI='mongodb://localhost:27017/?directConnection=true' \
///   cargo test --features mongodb -- --ignored mongodb_vector_live
/// ```
#[tokio::test]
#[ignore = "needs MONGODB_VECTOR_TEST_URI"]
async fn mongodb_vector_live_round_trip() {
    let uri = std::env::var("MONGODB_VECTOR_TEST_URI").expect("MONGODB_VECTOR_TEST_URI");
    let database = format!("lcr_test_{}", ObjectId::new().to_hex());
    let store = StoreBuilder::new()
        .uri(&uri)
        .database(&database)
        .collection("docs")
        .embedder(embedder())
        .filter_paths(["metadata.source", "metadata.year"])
        .build()
        .await
        .unwrap();

    // Before anything is written: no collection, no index, no results (not an error).
    assert!(store
        .similarity_search("alpha", 5, &VecStoreOptions::default())
        .await
        .unwrap()
        .is_empty());

    let tenant = |ns: &str| VecStoreOptions::default().with_name_space(ns);
    let ids = store
        .add_documents(
            &[
                Document::new("alpha").with_metadata(
                    [
                        ("source".to_string(), json!("a.txt")),
                        ("year".to_string(), json!(2024)),
                        ("tags".to_string(), json!(["x"])),
                    ]
                    .into(),
                ),
                Document::new("beta").with_metadata(
                    [
                        ("source".to_string(), json!("b.txt")),
                        ("year".to_string(), json!(2019)),
                    ]
                    .into(),
                ),
            ],
            &tenant("tenant-a"),
        )
        .await
        .unwrap();
    assert_eq!(ids.len(), 2);
    let stored = store
        .collection()
        .find_one(doc! { "_id": ObjectId::parse_str(&ids[0]).unwrap() })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        stored.get_str("text").unwrap(),
        "alpha",
        "ids in input order"
    );
    store
        .add_documents(&[Document::new("alpha")], &tenant("tenant-b"))
        .await
        .unwrap();

    // Atlas indexes asynchronously.
    let search = |query: &'static str, opt: VecStoreOptions<Value>, expected: usize| {
        let store = &store;
        async move {
            for _ in 0..100 {
                let found = store.similarity_search(query, 5, &opt).await.unwrap();
                if found.len() >= expected {
                    return found;
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            panic!("{query}: fewer than {expected} results after 20s");
        }
    };

    let found = search("alpha", tenant("tenant-a"), 2).await;
    assert_eq!(
        found
            .iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "beta"]
    );
    // Cosine scores are normalized to (1 + cos) / 2.
    assert!((found[0].score - 1.0).abs() < 1e-5, "{}", found[0].score);
    assert!((found[1].score - 0.5).abs() < 1e-5, "{}", found[1].score);
    assert_eq!(found[0].metadata["source"], json!("a.txt"));
    assert_eq!(found[0].metadata["year"], json!(2024));
    assert_eq!(found[0].metadata["tags"], json!(["x"]));

    let by_source = search(
        "alpha",
        tenant("tenant-a").with_filters(json!({ "metadata.source": "b.txt" })),
        1,
    )
    .await;
    assert_eq!(by_source.len(), 1);
    assert_eq!(by_source[0].page_content, "beta");

    let by_year = search(
        "beta",
        tenant("tenant-a").with_filters(json!({ "metadata.year": { "$gte": 2020 } })),
        1,
    )
    .await;
    assert_eq!(by_year.len(), 1);
    assert_eq!(by_year[0].page_content, "alpha");

    let thresholded = store
        .similarity_search("alpha", 5, &tenant("tenant-a").with_score_threshold(0.9))
        .await
        .unwrap();
    assert_eq!(thresholded.len(), 1);

    let all = search("alpha", VecStoreOptions::default(), 3).await;
    assert_eq!(all.len(), 3);

    // A per-call embedder is used for the query: "alpha" now embeds like "beta".
    let swapped = FakeEmbedder::new(2, &[("alpha", vec![0.0, 1.0])]);
    let found = store
        .similarity_search("alpha", 1, &tenant("tenant-a").with_embedder(swapped))
        .await
        .unwrap();
    assert_eq!(found[0].page_content, "beta");

    // The builder's namespace applies to writes and searches without an override.
    let scoped = StoreBuilder::new()
        .uri(&uri)
        .database(&database)
        .collection("docs")
        .embedder(embedder())
        .namespace("tenant-b")
        .build()
        .await
        .unwrap();
    let found = search("beta", VecStoreOptions::default(), 1).await;
    assert_eq!(found.len(), 3, "unscoped store sees every tenant");
    let found = scoped
        .similarity_search("beta", 5, &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(
        found
            .iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>(),
        ["alpha"]
    );
    // A per-call namespace beats the builder's on writes too.
    scoped
        .add_documents(&[Document::new("beta")], &tenant("tenant-d"))
        .await
        .unwrap();
    let found = search("beta", tenant("tenant-d"), 1).await;
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].page_content, "beta");

    // Caller ids become string `_id`s; writing one again replaces the document.
    let ids = store
        .add_documents(
            &[
                Document::new("alpha")
                    .with_id("doc-1")
                    .with_metadata([("old".to_string(), json!(true))].into()),
                Document::new("beta"),
            ],
            &tenant("tenant-e"),
        )
        .await
        .unwrap();
    assert_eq!(ids[0], "doc-1");
    assert!(ObjectId::parse_str(&ids[1]).is_ok(), "{ids:?}");
    let ids = store
        .add_documents(
            &[Document::new("beta").with_id("doc-1")],
            &tenant("tenant-e"),
        )
        .await
        .unwrap();
    assert_eq!(ids, ["doc-1"]);
    let replaced = store
        .collection()
        .find_one(doc! { "_id": "doc-1" })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(replaced.get_str("text").unwrap(), "beta");
    assert!(replaced
        .get_document("metadata")
        .unwrap()
        .get("old")
        .is_none());
    let found = search("beta", tenant("tenant-e"), 2).await;
    assert_eq!(found.len(), 2, "the replace didn't add a document");
    assert!(
        found.iter().any(|d| d.id.as_deref() == Some("doc-1")),
        "{found:?}"
    );

    // get_by_ids / delete: caller ids, generated ObjectId hex, namespace scope.
    let generated = found
        .iter()
        .find_map(|d| d.id.clone().filter(|id| id != "doc-1"))
        .unwrap();
    let ask = vec![generated.clone(), "nope".to_string(), "doc-1".to_string()];
    let got = store.get_by_ids(&ask, &tenant("tenant-e")).await.unwrap();
    let got_ids: Vec<_> = got.iter().map(|d| d.id.clone().unwrap()).collect();
    assert_eq!(got_ids, [generated.clone(), "doc-1".to_string()]);
    assert_eq!(got[1].page_content, "beta");
    assert_eq!(got[1].score, 0.0);
    // Hex ids match case-insensitively and come back lowercase.
    let upper = store
        .get_by_ids(&[generated.to_uppercase()], &tenant("tenant-e"))
        .await
        .unwrap();
    assert_eq!(upper.len(), 1, "{upper:?}");
    assert_eq!(upper[0].id.as_deref(), Some(generated.as_str()));
    assert!(store
        .get_by_ids(&ask, &tenant("tenant-a"))
        .await
        .unwrap()
        .is_empty());
    store.delete(&ask, &tenant("tenant-a")).await.unwrap();
    assert_eq!(
        store
            .get_by_ids(&ask, &VecStoreOptions::default())
            .await
            .unwrap()
            .len(),
        2
    );
    store.delete(&ask, &tenant("tenant-e")).await.unwrap();
    assert!(store
        .get_by_ids(&ask, &VecStoreOptions::default())
        .await
        .unwrap()
        .is_empty());

    // Two stores racing to create differently-defined indexes on a fresh collection:
    // the loser gets "duplicate index names" and uses the winner's index.
    let racer = |similarity| {
        let uri = uri.clone();
        let database = database.clone();
        async move {
            let store = StoreBuilder::new()
                .uri(&uri)
                .database(&database)
                .collection("race")
                .embedder(embedder())
                .similarity(similarity)
                .build()
                .await
                .unwrap();
            store
                .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
                .await
                .map(|_| ())
                .map_err(|e| e.to_string())
        }
    };
    let (a, b) = tokio::join!(
        racer(MongoSimilarity::Cosine),
        racer(MongoSimilarity::Euclidean)
    );
    assert_eq!((a, b), (Ok(()), Ok(())));

    // A second store on the same collection reuses the index.
    let again = StoreBuilder::new()
        .uri(&uri)
        .database(&database)
        .collection("docs")
        .embedder(embedder())
        .build()
        .await
        .unwrap();
    again
        .add_documents(&[Document::new("beta")], &tenant("tenant-c"))
        .await
        .unwrap();
    let found = search("beta", tenant("tenant-c"), 1).await;
    assert_eq!(found[0].page_content, "beta");

    // With create_index(false) the store never creates one.
    let unmanaged = StoreBuilder::new()
        .uri(&uri)
        .database(&database)
        .collection("other")
        .embedder(embedder())
        .create_index(false)
        .build()
        .await
        .unwrap();
    unmanaged
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap();
    let indexes: Vec<_> = unmanaged
        .collection()
        .list_search_indexes()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    assert!(indexes.is_empty(), "{indexes:?}");

    // A different definition under the same name is used as is.
    let different = StoreBuilder::new()
        .uri(&uri)
        .database(&database)
        .collection("docs")
        .embedder(embedder())
        .similarity(MongoSimilarity::Euclidean)
        .build()
        .await
        .unwrap();
    different
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap();

    store
        .collection()
        .client()
        .database(&database)
        .drop()
        .await
        .unwrap();
}
