//! Offline tests against an in-memory SurrealDB (`kv-mem`, dev-dependency only).

use ::surrealdb::engine::local::{Db, Mem};
use ::surrealdb::Surreal;
use serde_json::{json, Value};

use super::{Store, StoreBuilder};
use crate::{
    schemas::Document,
    test_utils::FakeEmbedder,
    vectorstore::{VecStoreOptions, VectorStore},
};

fn embedder() -> FakeEmbedder {
    FakeEmbedder::new(
        3,
        &[
            ("cats purr", vec![1.0, 0.0, 0.0]),
            ("dogs bark", vec![0.0, 1.0, 0.0]),
            ("kittens purr", vec![0.9, 0.1, 0.0]),
            ("cat", vec![1.0, 0.0, 0.0]),
        ],
    )
}

async fn mem_db() -> Surreal<Db> {
    let db = Surreal::new::<Mem>(()).await.unwrap();
    db.use_ns("test").use_db("test").await.unwrap();
    db
}

fn docs() -> Vec<Document> {
    vec![
        Document::new("cats purr").with_metadata([("k".to_string(), json!("a"))].into()),
        Document::new("dogs bark").with_metadata([("k".to_string(), json!("b"))].into()),
        Document::new("kittens purr"),
    ]
}

async fn add_and_search(store: &Store<Db>) {
    let opts = VecStoreOptions::<Value>::default();
    let ids = store.add_documents(&docs(), &opts).await.unwrap();
    assert_eq!(ids.len(), 3);
    assert!(ids.iter().all(|id| !id.is_empty()));

    let results = store.similarity_search("cat", 2, &opts).await.unwrap();
    assert_eq!(results.len(), 2, "{results:?}");
    assert_eq!(results[0].page_content, "cats purr");
    assert_eq!(results[0].id.as_deref(), Some(ids[0].as_str()));
    assert!((results[0].score - 1.0).abs() < 1e-9);
    assert_eq!(results[0].metadata["k"], json!("a"));
    assert_eq!(results[1].page_content, "kittens purr");
    assert!(results[0].score >= results[1].score);

    // "dogs bark" is orthogonal to the query (cosine 0) and is cut by the threshold.
    let thresholded = store
        .similarity_search("cat", 10, &VecStoreOptions::new().with_score_threshold(0.5))
        .await
        .unwrap();
    assert_eq!(thresholded.len(), 2);

    get_and_delete(store, &ids).await;
}

/// `ids` are what `add_documents` returned for `docs()`.
async fn get_and_delete(store: &Store<Db>, ids: &[String]) {
    let opts = VecStoreOptions::<Value>::default();
    // Request order, missing ids skipped, a repeated id returned once.
    let ask = vec![
        ids[2].clone(),
        "missing".into(),
        ids[0].clone(),
        ids[2].clone(),
    ];
    let got = store.get_by_ids(&ask, &opts).await.unwrap();
    let got_ids: Vec<_> = got.iter().map(|d| d.id.clone().unwrap()).collect();
    assert_eq!(got_ids, [ids[2].clone(), ids[0].clone()]);
    assert_eq!(got[0].page_content, "kittens purr");
    assert_eq!(got[1].page_content, "cats purr");
    assert_eq!(got[1].metadata["k"], json!("a"));
    assert_eq!(got[1].score, 0.0);

    store.delete(&ask, &opts).await.unwrap();
    assert!(store.get_by_ids(&ask, &opts).await.unwrap().is_empty());
    let left = store.similarity_search("cat", 10, &opts).await.unwrap();
    let left: Vec<_> = left.iter().map(|d| d.page_content.as_str()).collect();
    assert_eq!(left, ["dogs bark"]);

    // Empty input is a no-op; scopes that can't narrow an id lookup are rejected.
    store.delete(&[], &opts).await.unwrap();
    assert!(store.get_by_ids(&[], &opts).await.unwrap().is_empty());
    for bad in [
        VecStoreOptions::<Value>::default().with_name_space("t"),
        VecStoreOptions::<Value>::default().with_filters(json!({"k": "b"})),
    ] {
        assert!(store.get_by_ids(&ids[1..2], &bad).await.is_err());
        assert!(store.delete(&ids[1..2], &bad).await.is_err());
    }
    assert_eq!(store.get_by_ids(&ids[1..2], &opts).await.unwrap().len(), 1);
}

#[tokio::test]
async fn schemafull_shared_table_round_trip() {
    let store = StoreBuilder::new()
        .db(mem_db().await)
        .vector_dimensions(3)
        .embedder(embedder())
        .build()
        .await
        .unwrap();
    store.initialize().await.unwrap();

    add_and_search(&store).await;
}

#[tokio::test]
async fn shared_table_isolates_collections() {
    let db = mem_db().await;
    let build = |name: &str| {
        StoreBuilder::new()
            .db(db.clone())
            .collection_name(name)
            .vector_dimensions(3)
            .embedder(embedder())
            .build()
    };
    let a = build("a").await.unwrap();
    let b = build("b").await.unwrap();
    a.initialize().await.unwrap();

    let opts = VecStoreOptions::<Value>::default();
    let a_ids = a
        .add_documents(&[Document::new("cats purr")], &opts)
        .await
        .unwrap();
    let b_ids = b
        .add_documents(&[Document::new("dogs bark")], &opts)
        .await
        .unwrap();

    let results = a.similarity_search("cat", 10, &opts).await.unwrap();
    assert_eq!(results.len(), 1, "{results:?}");
    assert_eq!(results[0].page_content, "cats purr");
    assert_eq!(results[0].metadata["collection"], json!("a"));

    // Id lookups don't cross collections on the shared table.
    assert!(a.get_by_ids(&b_ids, &opts).await.unwrap().is_empty());
    a.delete(&b_ids, &opts).await.unwrap();
    let still_b = b.get_by_ids(&b_ids, &opts).await.unwrap();
    assert_eq!(still_b.len(), 1);
    assert_eq!(still_b[0].page_content, "dogs bark");
    b.delete(&b_ids, &opts).await.unwrap();
    assert!(b.get_by_ids(&b_ids, &opts).await.unwrap().is_empty());
    assert_eq!(a.get_by_ids(&a_ids, &opts).await.unwrap().len(), 1);
}

#[tokio::test]
async fn compatibility_mode_table_per_collection() {
    let store = StoreBuilder::new_with_compatiblity()
        .db(mem_db().await)
        .embedder(embedder())
        .build()
        .await
        .unwrap();
    store.initialize().await.unwrap();

    add_and_search(&store).await;
}

#[tokio::test]
async fn schemafull_rejects_wrong_dimensions() {
    let store = StoreBuilder::new()
        .db(mem_db().await)
        .vector_dimensions(5)
        .embedder(embedder())
        .build()
        .await
        .unwrap();
    store.initialize().await.unwrap();

    let err = store
        .add_documents(&[Document::new("cats purr")], &VecStoreOptions::default())
        .await;
    assert!(err.is_err());
}

#[tokio::test]
async fn builder_requires_db_and_embedder() {
    let no_embedder = StoreBuilder::<Db>::new().db(mem_db().await).build().await;
    assert!(no_embedder.is_err());
    let no_db = StoreBuilder::<Db>::new().embedder(embedder()).build().await;
    assert!(no_db.is_err());
}
