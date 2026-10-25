mod builder;
mod pgvector;

pub use builder::*;
pub use pgvector::*;

// pgLockIDEmbeddingTable is used for advisor lock to fix issue arising from concurrent
// creation of the embedding table.The same value represents the same lock.
const PG_LOCK_ID_EMBEDDING_TABLE: i64 = 1573678846307946494;
// pgLockIDCollectionTable is used for advisor lock to fix issue arising from concurrent
// creation of the collection table.The same value represents the same lock.
const PG_LOCK_ID_COLLECTION_TABLE: i64 = 1573678846307946495;
// pgLockIDExtension is used for advisor lock to fix issue arising from concurrent creation
// of the vector extension. The value is deliberately set to the same as python langchain
// https://github.com/langchain-ai/langchain/blob/v0.0.340/libs/langchain/langchain/vectorstores/pgvector.py#L167
const PG_LOCKID_EXTENSION: i64 = 1573678846307946496;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_trait::async_trait;
    use serde_json::{json, Value};
    use sqlx::{AssertSqlSafe, Row};
    use tokio::sync::OnceCell;

    use crate::{
        embedding::{Embedder, EmbedderError},
        schemas::Document,
        test_utils::FakeEmbedder,
        vectorstore::{pgvector::StoreBuilder, VecStoreOptions, VectorStore},
    };

    use super::Store;

    static PG_URL: OnceCell<String> = OnceCell::const_new();

    async fn postgres_url() -> String {
        PG_URL
            .get_or_init(|| async {
                if let Ok(url) = std::env::var("PGVECTOR_TEST_URL") {
                    if !url.is_empty() {
                        return url;
                    }
                }
                let out = std::process::Command::new("docker")
                    .args([
                        "run",
                        "-d",
                        "--rm",
                        "-e",
                        "POSTGRES_PASSWORD=pw",
                        "-p",
                        "0:5432",
                        "pgvector/pgvector:pg17",
                    ])
                    .output()
                    .expect("docker");
                let id = String::from_utf8(out.stdout).unwrap().trim().to_string();
                let port = std::process::Command::new("docker")
                    .args(["port", &id, "5432/tcp"])
                    .output()
                    .unwrap();
                let port = String::from_utf8(port.stdout).unwrap();
                let port = port
                    .lines()
                    .next()
                    .unwrap()
                    .rsplit(':')
                    .next()
                    .unwrap()
                    .to_string();
                format!("postgres://postgres:pw@127.0.0.1:{port}/postgres")
            })
            .await
            .clone()
    }

    fn embedder() -> FakeEmbedder {
        FakeEmbedder::new(
            3,
            &[
                ("near", vec![1.0, 0.0, 0.0]),
                ("mid", vec![1.0, 1.0, 0.0]),
                ("far", vec![-1.0, 0.0, 0.0]),
                ("query", vec![1.0, 0.05, 0.0]),
            ],
        )
    }

    async fn build_store(
        url: &str,
        collection: &str,
        embed_table: &str,
        coll_table: &str,
        embedder: FakeEmbedder,
    ) -> Store {
        let mut store = None;
        for _ in 0..60 {
            match StoreBuilder::new()
                .embedder(embedder.clone())
                .connection_url(url)
                .vector_dimensions(3)
                .collection_name(collection)
                .collection_table_name(coll_table)
                .embedder_table_name(embed_table)
                .build()
                .await
            {
                Ok(store_ok) => {
                    store = Some(store_ok);
                    break;
                }
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
            }
        }
        store.expect("postgres did not come up")
    }

    fn table_names(label: &str) -> (String, String, String) {
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        (
            format!("col_{label}_{suffix}"),
            format!("lc_emb_{suffix}"),
            format!("lc_coll_{suffix}"),
        )
    }

    async fn drop_tables(store: &Store) {
        let embed = &store.embedder_table_name;
        let coll = &store.collection_table_name;
        let _ = sqlx::query(AssertSqlSafe(format!("DROP TABLE IF EXISTS {embed}")))
            .execute(&store.pool)
            .await;
        let _ = sqlx::query(AssertSqlSafe(format!("DROP TABLE IF EXISTS {coll}")))
            .execute(&store.pool)
            .await;
    }

    async fn row_count(store: &Store) -> i64 {
        let sql = format!(
            "SELECT COUNT(*) FROM {} WHERE collection_id = $1",
            store.embedder_table_name
        );
        let row = sqlx::query(AssertSqlSafe(sql))
            .bind(&store.collection_uuid)
            .fetch_one(&store.pool)
            .await
            .unwrap();
        row.try_get(0).unwrap()
    }

    async fn stored_row(store: &Store, id: &str) -> Option<(String, Value, String)> {
        let sql = format!(
            "SELECT document, cmetadata, embedding::text FROM {} WHERE uuid = $1",
            store.embedder_table_name
        );
        let row = sqlx::query(AssertSqlSafe(sql))
            .bind(id)
            .fetch_optional(&store.pool)
            .await
            .unwrap()?;
        Some((
            row.try_get(0).unwrap(),
            row.try_get(1).unwrap(),
            row.try_get(2).unwrap(),
        ))
    }

    struct FailEmbedder;

    #[async_trait]
    impl Embedder for FailEmbedder {
        async fn embed_documents(&self, _: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
            Err(EmbedderError::FastEmbedError("embed failed".into()))
        }

        async fn embed_query(&self, _: &str) -> Result<Vec<f64>, EmbedderError> {
            Err(EmbedderError::FastEmbedError("embed failed".into()))
        }
    }

    /// Needs Docker. Run with `cargo test --features postgres -- --ignored pgvector`.
    /// Or point PGVECTOR_TEST_URL at an existing postgres with the `vector` extension.
    #[tokio::test]
    #[ignore]
    async fn pgvector_similarity_search_returns_closest_first() {
        let url = postgres_url().await;
        let (collection, embed_table, coll_table) = table_names("order");
        let store = build_store(&url, &collection, &embed_table, &coll_table, embedder()).await;

        let docs = ["far", "mid", "near"].map(Document::new);
        store
            .add_documents(&docs, &VecStoreOptions::default())
            .await
            .unwrap();

        let found = store
            .similarity_search("query", 2, &VecStoreOptions::default())
            .await
            .unwrap();
        let contents: Vec<_> = found.iter().map(|d| d.page_content.as_str()).collect();
        assert_eq!(contents, ["near", "mid"]);
        assert!(found[0].score <= found[1].score, "score is cosine distance");
        assert!(found.iter().all(|doc| doc.id.is_some()));
        drop_tables(&store).await;
    }

    /// pgvecto.rs names its extension `vectors` but keeps pgvector's `vector` type and
    /// `<=>`. Run against it with e.g.
    /// `docker run -d -e POSTGRES_PASSWORD=pw -p 5433:5432 tensorchord/pgvecto-rs:pg17-v0.4.0`
    /// and `PGVECTORS_TEST_URL=postgres://postgres:pw@127.0.0.1:5433/postgres cargo test
    /// --features postgres --lib pgvecto_rs -- --ignored`.
    #[tokio::test]
    #[ignore]
    async fn pgvecto_rs_extension_name() {
        let Ok(url) = std::env::var("PGVECTORS_TEST_URL") else {
            eprintln!("PGVECTORS_TEST_URL not set, skipping");
            return;
        };
        let (collection, embed_table, coll_table) = table_names("vectors");
        let builder = || {
            StoreBuilder::new()
                .embedder(embedder())
                .connection_url(&url)
                .vector_dimensions(3)
                .collection_name(&collection)
                .collection_table_name(&coll_table)
                .embedder_table_name(&embed_table)
        };
        // There is no `vector` extension on pgvecto.rs, so the default fails.
        let err = builder().build().await.err().unwrap().to_string();
        assert!(err.contains("vector"), "{err}");

        let store = builder().extension_name("vectors").build().await.unwrap();
        let ids = store
            .add_documents(
                &["far", "mid", "near"].map(Document::new),
                &VecStoreOptions::default(),
            )
            .await
            .unwrap();
        let found = store
            .similarity_search("query", 2, &VecStoreOptions::default())
            .await
            .unwrap();
        let contents: Vec<_> = found.iter().map(|d| d.page_content.as_str()).collect();
        assert_eq!(contents, ["near", "mid"]);
        store
            .delete(&ids[2..], &VecStoreOptions::default())
            .await
            .unwrap();
        assert_eq!(row_count(&store).await, 2);
        drop_tables(&store).await;
    }

    /// Arbitrary ids, generated UUIDs, in-batch replacement, and search ordering.
    #[tokio::test]
    #[ignore]
    async fn pgvector_custom_ids_round_trip() {
        let url = postgres_url().await;
        let (collection, embed_table, coll_table) = table_names("ids");
        let store = build_store(&url, &collection, &embed_table, &coll_table, embedder()).await;
        let opts = VecStoreOptions::default();

        let weird = "source/system:abc 123";
        let inserted = store
            .add_documents(
                &[Document::new("near")
                    .with_id(weird)
                    .with_metadata(HashMap::from([("k".to_string(), json!("old"))]))],
                &opts,
            )
            .await
            .unwrap();
        assert_eq!(inserted, vec![weird.to_string()]);
        let (document, metadata, embedding) = stored_row(&store, weird).await.unwrap();
        assert_eq!(document, "near");
        assert_eq!(metadata["k"], json!("old"));

        let updated = store
            .add_documents(
                &[Document::new("mid")
                    .with_id(weird)
                    .with_metadata(HashMap::from([("k".to_string(), json!("new"))]))],
                &opts,
            )
            .await
            .unwrap();
        assert_eq!(updated, vec![weird.to_string()]);
        assert_eq!(row_count(&store).await, 1);
        let (document, metadata, embedding_after) = stored_row(&store, weird).await.unwrap();
        assert_eq!(document, "mid");
        assert_eq!(metadata["k"], json!("new"));
        assert_ne!(embedding, embedding_after);

        let found = store.similarity_search("query", 1, &opts).await.unwrap();
        assert_eq!(found[0].id.as_deref(), Some(weird));
        assert_eq!(found[0].page_content, "mid");
        assert_eq!(found[0].metadata["k"], json!("new"));

        let generated = store
            .add_documents(&[Document::new("far")], &opts)
            .await
            .unwrap();
        assert!(
            uuid::Uuid::parse_str(&generated[0]).is_ok(),
            "{}",
            generated[0]
        );
        let generated_hit = store.similarity_search("far", 1, &opts).await.unwrap();
        assert_eq!(generated_hit[0].id.as_deref(), Some(generated[0].as_str()));
        assert!(stored_row(&store, &generated[0]).await.is_some());

        let mixed = store
            .add_documents(
                &[
                    Document::new("near").with_id("custom-mixed"),
                    Document::new("far"),
                ],
                &opts,
            )
            .await
            .unwrap();
        assert_eq!(mixed[0], "custom-mixed");
        assert!(uuid::Uuid::parse_str(&mixed[1]).is_ok());

        let dup = store
            .add_documents(
                &[
                    Document::new("near")
                        .with_id("dup")
                        .with_metadata(HashMap::from([("k".to_string(), json!("1"))])),
                    Document::new("far")
                        .with_id("dup")
                        .with_metadata(HashMap::from([("k".to_string(), json!("2"))])),
                ],
                &opts,
            )
            .await
            .unwrap();
        assert_eq!(dup, vec!["dup".to_string(), "dup".to_string()]);
        let (document, metadata, _) = stored_row(&store, "dup").await.unwrap();
        assert_eq!(document, "far");
        assert_eq!(metadata["k"], json!("2"));
        assert_eq!(row_count(&store).await, 5);

        let ordered = store
            .add_documents(
                &[
                    Document::new("far").with_id("id-far"),
                    Document::new("mid").with_id("id-mid"),
                    Document::new("near").with_id("id-near"),
                ],
                &opts,
            )
            .await
            .unwrap();
        assert_eq!(
            ordered,
            vec![
                "id-far".to_string(),
                "id-mid".to_string(),
                "id-near".to_string()
            ]
        );
        let ranked = store.similarity_search("query", 10, &opts).await.unwrap();
        let pos = |id: &str| {
            ranked
                .iter()
                .position(|doc| doc.id.as_deref() == Some(id))
                .unwrap()
        };
        assert!(pos("id-near") < pos("id-mid"));
        assert!(pos("id-mid") < pos("id-far"));
        assert!(ranked[pos("id-near")].score <= ranked[pos("id-mid")].score);
        assert!(ranked[pos("id-mid")].score <= ranked[pos("id-far")].score);

        let fail_opts = VecStoreOptions::default().with_embedder(FailEmbedder);
        let empty = store
            .add_documents(&[Document::new("near").with_id("")], &fail_opts)
            .await
            .unwrap_err()
            .to_string();
        assert!(empty.contains("empty"), "{empty}");
        assert!(!empty.contains("embed failed"), "{empty}");

        let embed_err = store
            .add_documents(
                &[Document::new("near").with_id("never-written")],
                &fail_opts,
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(embed_err.contains("embed failed"), "{embed_err}");
        assert!(stored_row(&store, "never-written").await.is_none());

        drop_tables(&store).await;
    }

    /// The same id in a second collection must not overwrite the first, and the
    /// failing batch rolls back.
    #[tokio::test]
    #[ignore]
    async fn pgvector_cross_collection_id_collision_rolls_back() {
        let url = postgres_url().await;
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let embed_table = format!("lc_emb_{suffix}");
        let coll_table = format!("lc_coll_{suffix}");
        let embedder = embedder();
        let store_a = build_store(
            &url,
            &format!("col_a_{suffix}"),
            &embed_table,
            &coll_table,
            embedder.clone(),
        )
        .await;
        let store_b = build_store(
            &url,
            &format!("col_b_{suffix}"),
            &embed_table,
            &coll_table,
            embedder,
        )
        .await;
        let opts = VecStoreOptions::default();

        store_a
            .add_documents(
                &[Document::new("near")
                    .with_id("shared-id")
                    .with_metadata(HashMap::from([("k".to_string(), json!("a"))]))],
                &opts,
            )
            .await
            .unwrap();

        let err = store_b
            .add_documents(
                &[
                    Document::new("mid").with_id("fresh-id"),
                    Document::new("far")
                        .with_id("shared-id")
                        .with_metadata(HashMap::from([("k".to_string(), json!("b"))])),
                ],
                &opts,
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("shared-id"), "{err}");
        assert!(err.contains("another collection"), "{err}");
        assert!(stored_row(&store_b, "fresh-id").await.is_none());
        let (document, metadata, _) = stored_row(&store_a, "shared-id").await.unwrap();
        assert_eq!(document, "near");
        assert_eq!(metadata["k"], json!("a"));
        assert_eq!(row_count(&store_a).await, 1);
        assert_eq!(row_count(&store_b).await, 0);

        drop_tables(&store_a).await;
    }

    /// get_by_ids / delete: request order, missing ids, collection scoping, options.
    #[tokio::test]
    #[ignore]
    async fn pgvector_get_by_ids_and_delete() {
        let url = postgres_url().await;
        let suffix = uuid::Uuid::new_v4().simple().to_string();
        let embed_table = format!("lc_emb_{suffix}");
        let coll_table = format!("lc_coll_{suffix}");
        let store_a = build_store(
            &url,
            &format!("col_a_{suffix}"),
            &embed_table,
            &coll_table,
            embedder(),
        )
        .await;
        let store_b = build_store(
            &url,
            &format!("col_b_{suffix}"),
            &embed_table,
            &coll_table,
            embedder(),
        )
        .await;
        let opts = VecStoreOptions::default();
        let ids = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        store_a
            .add_documents(
                &[
                    Document::new("near")
                        .with_id("a1")
                        .with_metadata(HashMap::from([("k".to_string(), json!(1))])),
                    Document::new("far").with_id("a2"),
                ],
                &opts,
            )
            .await
            .unwrap();
        store_b
            .add_documents(&[Document::new("mid").with_id("b1")], &opts)
            .await
            .unwrap();

        let got = store_a
            .get_by_ids(&ids(&["a2", "missing", "b1", "a1"]), &opts)
            .await
            .unwrap();
        let got_ids: Vec<_> = got.iter().map(|d| d.id.as_deref().unwrap()).collect();
        assert_eq!(
            got_ids,
            ["a2", "a1"],
            "request order, other collection hidden"
        );
        assert_eq!(got[1].page_content, "near");
        assert_eq!(got[1].metadata["k"], json!(1));
        assert_eq!(got[1].score, 0.0);
        assert!(store_a.get_by_ids(&[], &opts).await.unwrap().is_empty());

        let ns = VecStoreOptions::default().with_name_space("x");
        assert!(store_a.get_by_ids(&ids(&["a1"]), &ns).await.is_err());
        assert!(store_a.delete(&ids(&["a1"]), &ns).await.is_err());

        // b1 belongs to the other collection: untouched.
        store_a
            .delete(&ids(&["a1", "b1", "missing"]), &opts)
            .await
            .unwrap();
        assert!(stored_row(&store_a, "a1").await.is_none());
        assert!(stored_row(&store_a, "a2").await.is_some());
        assert!(stored_row(&store_b, "b1").await.is_some());
        store_a.delete(&[], &opts).await.unwrap();
        let found = store_a.similarity_search("query", 10, &opts).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id.as_deref(), Some("a2"));

        drop_tables(&store_a).await;
    }
}
