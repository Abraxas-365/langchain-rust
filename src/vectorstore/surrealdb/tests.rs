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
    a.add_documents(&[Document::new("cats purr")], &opts)
        .await
        .unwrap();
    b.add_documents(&[Document::new("dogs bark")], &opts)
        .await
        .unwrap();

    let results = a.similarity_search("cat", 10, &opts).await.unwrap();
    assert_eq!(results.len(), 1, "{results:?}");
    assert_eq!(results[0].page_content, "cats purr");
    assert_eq!(results[0].metadata["collection"], json!("a"));
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
