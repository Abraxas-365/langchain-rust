//! Offline tests for the helpers and builder checks, plus live tests against MariaDB
//! (`#[ignore]`): set `MARIADB_TEST_URL` (e.g. `mysql://root:pw@127.0.0.1:3306/lcr`) or let
//! them start a `mariadb:11.8` container through docker.

use std::collections::HashMap;

use serde_json::{json, Value};
use sqlx::{AssertSqlSafe, Row};
use tokio::sync::OnceCell;

use super::*;
use crate::{
    embedding::{Embedder, EmbedderError},
    schemas::Document,
    test_utils::{FakeEmbedder, PanickingEmbedder},
    vectorstore::{VecStoreOptions, VectorStore},
};

/// Options whose embedder panics: proves a call returns before embedding anything.
fn panicking() -> MariaDBOptions {
    MariaDBOptions::default().with_embedder(PanickingEmbedder)
}

/// Returns one vector however many documents it gets.
struct ShortEmbedder;

#[async_trait::async_trait]
impl Embedder for ShortEmbedder {
    async fn embed_documents(&self, _: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        Ok(vec![vec![1.0, 0.0, 0.0]])
    }

    async fn embed_query(&self, _: &str) -> Result<Vec<f64>, EmbedderError> {
        Ok(vec![1.0, 0.0, 0.0])
    }
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

#[test]
fn identifiers() {
    for ok in ["docs", "_t", "Lc_Emb_2", &"a".repeat(64)] {
        assert!(check_identifier("t", ok).is_ok(), "{ok}");
    }
    for bad in [
        "",
        "1docs",
        "my-table",
        "a b",
        "t;DROP TABLE x",
        "`t`",
        "tábla",
        &"a".repeat(65),
    ] {
        let err = check_identifier("embedding_table_name", bad)
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("embedding_table_name"), "{bad}: {err}");
    }
}

#[test]
fn vector_text_is_f32_json() {
    assert_eq!(vector_text(&[]), "[]");
    assert_eq!(vector_text(&[1.0, 0.5, -2.0, 0.0]), "[1,0.5,-2,0]");
    assert_eq!(vector_text(&[1.0 / 3.0]), "[0.33333334]");
}

#[test]
fn option_parsing() {
    let opt = MariaDBOptions::default();
    assert_eq!(filter_json(&opt).unwrap(), None);
    assert_eq!(score_threshold(&opt).unwrap(), None);

    let empty = MariaDBOptions::default().with_filters(json!({}));
    assert_eq!(filter_json(&empty).unwrap(), None);
    let filter = MariaDBOptions::default().with_filters(json!({"lang": "en", "n": 2}));
    let parsed: Value = serde_json::from_str(&filter_json(&filter).unwrap().unwrap()).unwrap();
    assert_eq!(parsed, json!({"lang": "en", "n": 2}));
    for bad in [json!("lang = 'en'"), json!(["en"]), json!(null)] {
        let opt = MariaDBOptions::default().with_filters(bad.clone());
        let err = filter_json(&opt).unwrap_err().to_string();
        assert!(err.contains("JSON object"), "{bad}: {err}");
    }

    for (t, expected) in [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)] {
        let opt = MariaDBOptions::default().with_score_threshold(t);
        assert_eq!(score_threshold(&opt).unwrap(), Some(expected));
    }
    for t in [-0.01, 1.01, f32::NAN] {
        let opt = MariaDBOptions::default().with_score_threshold(t);
        assert!(score_threshold(&opt).is_err(), "{t}");
    }
}

#[tokio::test]
async fn builder_checks_run_before_connecting() {
    // Nothing listens on port 9; reaching the connection would fail differently.
    let url = "mysql://root:pw@127.0.0.1:9/x";
    let base = || {
        StoreBuilder::new()
            .embedder(embedder())
            .connection_url(url)
            .vector_dimensions(3)
    };
    let cases: Vec<(StoreBuilder, &str)> = vec![
        (
            StoreBuilder::new().connection_url(url).vector_dimensions(3),
            "embedder is required",
        ),
        (base().vector_dimensions(0), "vector_dimensions"),
        (
            base().embedding_table_name("bad-name"),
            "embedding_table_name",
        ),
        (base().collection_table_name("x;y"), "collection_table_name"),
        (
            base()
                .embedding_table_name("same")
                .collection_table_name("same"),
            "must differ",
        ),
        (base().collection_name(""), "collection_name"),
        (base().collection_name("é".repeat(256)), "collection_name"),
        (
            StoreBuilder::new()
                .embedder(embedder())
                .vector_dimensions(3),
            "connection_url or pool is required",
        ),
    ];
    for (builder, expected) in cases {
        let err = match builder.build().await {
            Ok(_) => panic!("{expected}: built"),
            Err(err) => err.to_string(),
        };
        assert!(err.contains(expected), "expected {expected:?}, got {err}");
    }
    // 255 multi-byte chars is fine (the check counts chars, not bytes): it gets as far as
    // using the pool. A lazy pool with a short timeout keeps the failure quick.
    let pool = sqlx::mysql::MySqlPoolOptions::new()
        .acquire_timeout(std::time::Duration::from_millis(300))
        .connect_lazy(url)
        .unwrap();
    let err = match base()
        .pool(pool)
        .collection_name("é".repeat(255))
        .build()
        .await
    {
        Ok(_) => panic!("connected to port 9?"),
        Err(err) => err.to_string(),
    };
    assert!(!err.contains("collection_name"), "{err}");
}

// ---- live ----

static URL: OnceCell<String> = OnceCell::const_new();

async fn mariadb_url() -> String {
    URL.get_or_init(|| async {
        if let Ok(url) = std::env::var("MARIADB_TEST_URL") {
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
                "MARIADB_ROOT_PASSWORD=pw",
                "-e",
                "MARIADB_DATABASE=lcr",
                "-p",
                "0:3306",
                "mariadb:11.8",
            ])
            .output()
            .expect("docker");
        let id = String::from_utf8(out.stdout).unwrap().trim().to_string();
        let port = std::process::Command::new("docker")
            .args(["port", &id, "3306/tcp"])
            .output()
            .unwrap();
        let port = String::from_utf8(port.stdout).unwrap();
        let port = port.lines().next().unwrap().rsplit(':').next().unwrap();
        format!("mysql://root:pw@127.0.0.1:{port}/lcr")
    })
    .await
    .clone()
}

struct Tables {
    emb: String,
    coll: String,
}

fn tables() -> Tables {
    let suffix = uuid::Uuid::new_v4().simple().to_string();
    Tables {
        emb: format!("lc_emb_{suffix}"),
        coll: format!("lc_coll_{suffix}"),
    }
}

async fn build(t: &Tables, collection: &str, embedder: FakeEmbedder) -> Store {
    let url = mariadb_url().await;
    let mut last = None;
    // A fresh container takes a few seconds to accept connections.
    for _ in 0..60 {
        match StoreBuilder::new()
            .embedder(embedder.clone())
            .connection_url(&url)
            .vector_dimensions(3)
            .collection_name(collection)
            .embedding_table_name(&t.emb)
            .collection_table_name(&t.coll)
            .build()
            .await
        {
            Ok(store) => return store,
            Err(err) => last = Some(err.to_string()),
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    panic!("mariadb did not come up: {last:?}");
}

async fn row_count(store: &Store) -> i64 {
    sqlx::query(AssertSqlSafe(format!(
        "SELECT COUNT(*) FROM {}",
        store.embedding_table_name
    )))
    .fetch_one(&store.pool)
    .await
    .unwrap()
    .get(0)
}

fn doc(text: &str, meta: Value) -> Document {
    let metadata: HashMap<String, Value> = serde_json::from_value(meta).unwrap();
    Document::new(text).with_metadata(metadata)
}

fn contents(docs: &[Document]) -> Vec<&str> {
    docs.iter().map(|d| d.page_content.as_str()).collect()
}

#[tokio::test]
#[ignore = "needs MariaDB 11.7+ (MARIADB_TEST_URL or docker)"]
async fn mariadb_similarity_search_options() {
    let t = tables();
    let store = build(&t, "main", embedder()).await;
    let ids = store
        .add_documents(
            &[
                doc("far", json!({"lang": "en", "n": 1})),
                doc("near", json!({"lang": "en", "n": 2, "tags": ["a", "b"]})),
                doc("mid", json!({"lang": "es", "n": 3})),
            ],
            &MariaDBOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(ids.len(), 3);
    assert!(ids.iter().all(|id| uuid::Uuid::parse_str(id).is_ok()));

    let opt = MariaDBOptions::default();
    let found = store.similarity_search("query", 10, &opt).await.unwrap();
    assert_eq!(contents(&found), ["near", "mid", "far"]);
    assert_eq!(found[0].id.as_deref(), Some(ids[1].as_str()));
    assert_eq!(found[0].metadata["tags"], json!(["a", "b"]));
    assert!(
        found[0].score > 0.99 && found[0].score <= 1.0,
        "{}",
        found[0].score
    );
    assert!(found[2].score < -0.99, "{}", found[2].score);
    assert!(found.windows(2).all(|w| w[0].score >= w[1].score));

    let top = store.similarity_search("query", 1, &opt).await.unwrap();
    assert_eq!(contents(&top), ["near"]);
    assert!(store
        .similarity_search("query", 0, &panicking())
        .await
        .unwrap()
        .is_empty());

    // cos(query, mid) ~ 0.74: a 0.7 threshold keeps it, 0.8 drops it.
    let at = |t: f32| MariaDBOptions::default().with_score_threshold(t);
    let kept = store
        .similarity_search("query", 10, &at(0.7))
        .await
        .unwrap();
    assert_eq!(contents(&kept), ["near", "mid"]);
    let kept = store
        .similarity_search("query", 10, &at(0.8))
        .await
        .unwrap();
    assert_eq!(contents(&kept), ["near"]);

    let with = |f: Value| MariaDBOptions::default().with_filters(f);
    let en = store
        .similarity_search("query", 10, &with(json!({"lang": "en"})))
        .await
        .unwrap();
    assert_eq!(contents(&en), ["near", "far"]);
    let both = store
        .similarity_search("query", 10, &with(json!({"lang": "en", "n": 1})))
        .await
        .unwrap();
    assert_eq!(contents(&both), ["far"]);
    let tag = store
        .similarity_search("query", 10, &with(json!({"tags": "b"})))
        .await
        .unwrap();
    assert_eq!(contents(&tag), ["near"]);
    // A quote in a filter value is data, not SQL.
    let quoted = store
        .similarity_search("query", 10, &with(json!({"lang": "en' OR '1'='1"})))
        .await
        .unwrap();
    assert!(quoted.is_empty());
    let combined = MariaDBOptions::default()
        .with_filters(json!({"lang": "en"}))
        .with_score_threshold(0.5);
    let found = store
        .similarity_search("query", 10, &combined)
        .await
        .unwrap();
    assert_eq!(contents(&found), ["near"]);

    // name_space searches another collection in the same tables.
    let other = build(&t, "other", embedder()).await;
    other
        .add_documents(&[doc("mid", json!({}))], &MariaDBOptions::default())
        .await
        .unwrap();
    let ns = MariaDBOptions::default().with_name_space("other");
    let found = store.similarity_search("query", 10, &ns).await.unwrap();
    assert_eq!(contents(&found), ["mid"]);
    let missing = MariaDBOptions::default().with_name_space("nope");
    assert!(store
        .similarity_search("query", 10, &missing)
        .await
        .unwrap()
        .is_empty());

    store.drop_tables().await.unwrap();
}

#[tokio::test]
#[ignore = "needs MariaDB 11.7+ (MARIADB_TEST_URL or docker)"]
async fn mariadb_ids_get_and_delete() {
    let t = tables();
    let store = build(&t, "main", embedder()).await;
    let opt = MariaDBOptions::default();
    let ids = store
        .add_documents(
            &[
                doc("near", json!({"v": 1})).with_id("a"),
                doc("far", json!({})),
            ],
            &opt,
        )
        .await
        .unwrap();
    assert_eq!(ids[0], "a");

    // Re-adding an id updates the row in place.
    store
        .add_documents(&[doc("mid", json!({"v": 2})).with_id("a")], &opt)
        .await
        .unwrap();
    assert_eq!(row_count(&store).await, 2);
    let got = store
        .get_by_ids(&[ids[1].clone(), "missing".into(), "a".into()], &opt)
        .await
        .unwrap();
    assert_eq!(contents(&got), ["far", "mid"]);
    assert_eq!(got[1].metadata["v"], json!(2));
    assert_eq!(got[1].score, 0.0);
    assert_eq!(got[1].id.as_deref(), Some("a"));
    // The vector was updated too: "a" now scores as mid (~0.74), not near (~1).
    let found = store.similarity_search("query", 1, &opt).await.unwrap();
    assert_eq!(contents(&found), ["mid"]);
    assert!(found[0].score < 0.8, "{}", found[0].score);

    // Another collection can't see, delete or take over these ids.
    let other = build(&t, "other", embedder()).await;
    assert!(other
        .get_by_ids(&["a".into()], &opt)
        .await
        .unwrap()
        .is_empty());
    other.delete(&["a".into()], &opt).await.unwrap();
    assert_eq!(row_count(&store).await, 2);
    let err = other
        .add_documents(
            &[
                doc("near", json!({})).with_id("fresh"),
                doc("near", json!({})).with_id("a"),
            ],
            &opt,
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("`a` already belongs to another collection"),
        "{err}"
    );
    // Nothing from the failed batch was written.
    assert_eq!(row_count(&store).await, 2);
    assert_eq!(
        store.get_by_ids(&["a".into()], &opt).await.unwrap()[0].page_content,
        "mid"
    );

    store
        .delete(&["a".into(), "missing".into()], &opt)
        .await
        .unwrap();
    assert_eq!(row_count(&store).await, 1);
    store.delete(&[], &opt).await.unwrap();
    assert!(store.get_by_ids(&[], &opt).await.unwrap().is_empty());

    // More ids than one IN chunk.
    let many: Vec<Document> = (0..1005)
        .map(|i| doc("far", json!({})).with_id(format!("m{i}")))
        .collect();
    let many_ids = store.add_documents(&many, &opt).await.unwrap();
    let got = store.get_by_ids(&many_ids, &opt).await.unwrap();
    assert_eq!(got.len(), 1005);
    assert_eq!(got[1004].id.as_deref(), Some("m1004"));
    store.delete(&many_ids, &opt).await.unwrap();
    assert_eq!(row_count(&store).await, 1);

    store.drop_tables().await.unwrap();
}

#[tokio::test]
#[ignore = "needs MariaDB 11.7+ (MARIADB_TEST_URL or docker)"]
async fn mariadb_validation_and_collections() {
    let t = tables();
    let store = build(&t, "main", embedder()).await;
    let opt = MariaDBOptions::default();

    assert!(store
        .add_documents(&[], &panicking())
        .await
        .unwrap()
        .is_empty());
    for (bad, expected) in [
        (doc("x", json!({})).with_id(""), "must not be empty"),
        (
            doc("x", json!({})).with_id("i".repeat(256)),
            "longer than 255",
        ),
    ] {
        let err = store
            .add_documents(&[bad], &opt)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains(expected), "{err}");
    }
    let ok = store
        .add_documents(&[doc("x", json!({})).with_id("i".repeat(255))], &opt)
        .await
        .unwrap();
    assert_eq!(ok[0].len(), 255);
    // The limit counts chars, as the varchar column does: 255 multi-byte chars fit.
    let wide_id = "é".repeat(255);
    store
        .add_documents(&[doc("x", json!({})).with_id(wide_id.clone())], &opt)
        .await
        .unwrap();
    assert_eq!(
        store.get_by_ids(&[wide_id], &opt).await.unwrap()[0].page_content,
        "x"
    );

    // An embedder that drops a vector is an error, not a silent partial write.
    let short = MariaDBOptions::default().with_embedder(ShortEmbedder);
    let err = store
        .add_documents(&[doc("x", json!({})), doc("y", json!({}))], &short)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("returned 1 vectors for 2 documents"), "{err}");
    assert_eq!(row_count(&store).await, 2);

    // Writes and lookups refuse query-only options.
    for opt in [
        MariaDBOptions::default().with_filters(json!({"a": 1})),
        MariaDBOptions::default().with_score_threshold(0.5),
        MariaDBOptions::default().with_name_space("other"),
    ] {
        for err in [
            store
                .add_documents(&[doc("x", json!({}))], &opt)
                .await
                .unwrap_err(),
            store.delete(&["a".into()], &opt).await.unwrap_err(),
            store.get_by_ids(&["a".into()], &opt).await.unwrap_err(),
        ] {
            assert!(err.to_string().contains("not supported"), "{err}");
        }
    }
    let bad = MariaDBOptions::default().with_score_threshold(2.0);
    assert!(store.similarity_search("query", 1, &bad).await.is_err());

    // Wrong-size vectors fail before any SQL.
    let wide = FakeEmbedder::new(4, &[]);
    let wide_opt = VecStoreOptions::default().with_embedder(wide.clone());
    let err = store
        .add_documents(&[doc("x", json!({}))], &wide_opt)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("returned 4 dimensions"), "{err}");
    let err = store
        .similarity_search("query", 1, &wide_opt)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("returned 4 dimensions"), "{err}");

    // An existing table keeps its size.
    let url = mariadb_url().await;
    let err = match StoreBuilder::new()
        .embedder(wide)
        .connection_url(&url)
        .vector_dimensions(4)
        .embedding_table_name(&t.emb)
        .collection_table_name(&t.coll)
        .build()
        .await
    {
        Ok(_) => panic!("built with 4 dims on a vector(3) table"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("vector(3)") && err.contains("is 4"), "{err}");

    // Building again reuses the collection, updates its metadata, keeps the documents.
    let rebuilt = StoreBuilder::new()
        .embedder(embedder())
        .connection_url(&url)
        .vector_dimensions(3)
        .collection_name("main")
        .collection_metadata(HashMap::from([("owner".to_string(), json!("me"))]))
        .embedding_table_name(&t.emb)
        .collection_table_name(&t.coll)
        .build()
        .await
        .unwrap();
    assert_eq!(rebuilt.collection_id, store.collection_id);
    assert_eq!(row_count(&store).await, 2);
    let meta: String = sqlx::query(AssertSqlSafe(format!(
        "SELECT cmetadata FROM {} WHERE name = 'main'",
        t.coll
    )))
    .fetch_one(&store.pool)
    .await
    .unwrap()
    .get(0);
    assert_eq!(
        serde_json::from_str::<Value>(&meta).unwrap(),
        json!({"owner": "me"})
    );

    // pre_delete_collection empties it (documents go with it) under a new id, and
    // leaves other collections alone.
    let other = build(&t, "other", embedder()).await;
    other
        .add_documents(&[doc("near", json!({}))], &opt)
        .await
        .unwrap();
    let fresh = StoreBuilder::new()
        .embedder(embedder())
        .connection_url(&url)
        .vector_dimensions(3)
        .collection_name("main")
        .pre_delete_collection(true)
        .embedding_table_name(&t.emb)
        .collection_table_name(&t.coll)
        .build()
        .await
        .unwrap();
    assert_ne!(fresh.collection_id, store.collection_id);
    assert!(fresh
        .similarity_search("query", 10, &opt)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(row_count(&fresh).await, 1);
    assert_eq!(
        contents(&other.similarity_search("query", 10, &opt).await.unwrap()),
        ["near"]
    );

    // Concurrent builds of a new collection agree on one row.
    let builds = (0..4).map(|_| build(&t, "racy", embedder()));
    let stores = futures::future::join_all(builds).await;
    assert!(stores
        .iter()
        .all(|s| s.collection_id == stores[0].collection_id));

    store.drop_tables().await.unwrap();
}
