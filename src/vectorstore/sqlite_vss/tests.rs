use std::str::FromStr;

use serde_json::json;
use sqlx::{
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
    Pool, Sqlite,
};

use super::StoreBuilder;
use crate::{
    schemas::Document,
    test_utils::FakeEmbedder,
    vectorstore::{VecStoreOptions, VectorStore},
};

/// An in-memory database with sqlite-vss loaded from `SQLITE_VSS_EXTENSION_DIR` (the
/// directory with `vector0` and `vss0`, e.g. from the `sqlite-vss` Python wheel).
async fn pool() -> Pool<Sqlite> {
    let dir = std::env::var("SQLITE_VSS_EXTENSION_DIR").expect("SQLITE_VSS_EXTENSION_DIR");
    let options = SqliteConnectOptions::from_str("sqlite::memory:").unwrap();
    // SAFETY: test-only, loads the extensions the developer pointed at.
    let options = unsafe {
        options
            .extension(format!("{dir}/vector0"))
            .extension(format!("{dir}/vss0"))
    };
    SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await
        .unwrap()
}

/// SQLITE_VSS_EXTENSION_DIR=/path/to/sqlite_vss \
///   cargo test --features sqlite-vss --lib -- --ignored sqlite_vss_live
#[tokio::test]
#[ignore]
async fn sqlite_vss_live_round_trip() {
    let embedder = FakeEmbedder::new(
        3,
        &[
            ("cats purr", vec![1.0, 0.0, 0.0]),
            ("dogs bark", vec![0.0, 1.0, 0.0]),
            ("kittens purr", vec![0.9, 0.1, 0.0]),
            ("cat", vec![1.0, 0.0, 0.0]),
        ],
    );
    let pool = pool().await;
    let store = StoreBuilder::new()
        .pool(pool.clone())
        .table("docs")
        .vector_dimensions(3)
        .embedder(embedder)
        .build()
        .await
        .unwrap();
    store.initialize().await.unwrap();
    let opts = VecStoreOptions::default();
    let ids = store
        .add_documents(
            &[
                Document::new("cats purr").with_metadata([("k".into(), json!("a"))].into()),
                Document::new("dogs bark"),
                Document::new("kittens purr"),
            ],
            &opts,
        )
        .await
        .unwrap();

    let found = store.similarity_search("cat", 2, &opts).await.unwrap();
    assert_eq!(found[0].page_content, "cats purr");
    assert_eq!(found[0].id.as_deref(), Some(ids[0].as_str()));

    let ask = vec![ids[2].clone(), "999".into(), "x".into(), ids[0].clone()];
    let got = store.get_by_ids(&ask, &opts).await.unwrap();
    let got: Vec<_> = got
        .iter()
        .map(|d| (d.id.clone().unwrap(), d.page_content.clone(), d.score))
        .collect();
    assert_eq!(
        got,
        [
            (ids[2].clone(), "kittens purr".to_string(), 0.0),
            (ids[0].clone(), "cats purr".to_string(), 0.0)
        ]
    );

    store.delete(&ask, &opts).await.unwrap();
    assert!(store.get_by_ids(&ask, &opts).await.unwrap().is_empty());
    let vectors: i64 = sqlx::query_scalar("SELECT count(*) FROM vss_docs")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(vectors, 1, "the vectors went with the rows");
    let left = store.similarity_search("cat", 10, &opts).await.unwrap();
    assert_eq!(left.len(), 1);
    assert_eq!(left[0].page_content, "dogs bark");
}
