use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use ::redis::{
    aio::ConnectionLike, make_extension_error, Arg, Cmd, Pipeline, RedisFuture, RedisResult,
    Value as RedisValue,
};
use futures::FutureExt;
use serde_json::{json, Value};

use super::*;
use crate::{
    schemas::Document,
    test_utils::FakeEmbedder,
    vectorstore::{VecStoreOptions, VectorStore},
};

type Handler = dyn Fn(&[String]) -> RedisResult<RedisValue> + Send + Sync;

/// A connection that records every command (arguments as lossy strings) and answers
/// through `handler`. Pipelines are recorded command by command.
#[derive(Clone)]
struct FakeRedis {
    log: Arc<Mutex<Vec<Vec<String>>>>,
    raw: Arc<Mutex<Vec<Vec<Vec<u8>>>>>,
    handler: Arc<Handler>,
}

impl FakeRedis {
    fn new(handler: impl Fn(&[String]) -> RedisResult<RedisValue> + Send + Sync + 'static) -> Self {
        Self {
            log: Default::default(),
            raw: Default::default(),
            handler: Arc::new(handler),
        }
    }

    fn record(&self, cmd: &Cmd) -> Vec<String> {
        let raw: Vec<Vec<u8>> = cmd
            .args_iter()
            .filter_map(|arg| match arg {
                Arg::Simple(bytes) => Some(bytes.to_vec()),
                _ => None,
            })
            .collect();
        let args: Vec<String> = raw
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect();
        self.log.lock().unwrap().push(args.clone());
        self.raw.lock().unwrap().push(raw);
        args
    }

    fn commands(&self) -> Vec<Vec<String>> {
        self.log.lock().unwrap().clone()
    }

    fn names(&self) -> Vec<String> {
        self.commands().iter().map(|c| c[0].clone()).collect()
    }
}

impl ConnectionLike for FakeRedis {
    fn req_packed_command<'a>(&'a mut self, cmd: &'a Cmd) -> RedisFuture<'a, RedisValue> {
        let args = self.record(cmd);
        futures::future::ready((self.handler)(&args)).boxed()
    }

    fn req_packed_commands<'a>(
        &'a mut self,
        pipeline: &'a Pipeline,
        offset: usize,
        count: usize,
    ) -> RedisFuture<'a, Vec<RedisValue>> {
        let mut replies = Vec::new();
        for cmd in pipeline.cmd_iter() {
            let args = self.record(cmd);
            match (self.handler)(&args) {
                Ok(v) => replies.push(v),
                Err(e) => return futures::future::ready(Err(e)).boxed(),
            }
        }
        let replies = replies.into_iter().skip(offset).take(count).collect();
        futures::future::ready(Ok(replies)).boxed()
    }

    fn get_db(&self) -> i64 {
        0
    }
}

fn bulk(s: &str) -> RedisValue {
    RedisValue::BulkString(s.as_bytes().to_vec())
}

fn not_found() -> RedisResult<RedisValue> {
    Err(make_extension_error(
        "SEARCH_INDEX_NOT_FOUND".into(),
        Some("Index not found: docs".into()),
    ))
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

fn store(conn: &FakeRedis, builder: StoreBuilder) -> Store<FakeRedis> {
    builder
        .index_name("docs")
        .embedder(embedder())
        .build_with_connection(conn.clone())
        .unwrap()
}

/// Real embedders reject empty input, so the store must not call them for it.
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

#[test]
fn builder_validates_names() {
    let conn = FakeRedis::new(|_| Ok(RedisValue::Okay));
    let build = |b: StoreBuilder| {
        b.build_with_connection(conn.clone())
            .err()
            .map(|e| e.to_string())
    };
    assert_eq!(
        build(StoreBuilder::new().embedder(embedder())).as_deref(),
        Some("index_name is required")
    );
    assert_eq!(
        build(StoreBuilder::new().index_name("docs")).as_deref(),
        Some("embedder is required")
    );
    for (bad, expected) in [
        (StoreBuilder::new().content_key("page content"), "invalid"),
        (StoreBuilder::new().vector_key("v-1"), "invalid"),
        (StoreBuilder::new().namespace_key("ns}"), "invalid"),
        (
            StoreBuilder::new().metadata_schema([MetadataField::tag("a|b")]),
            "invalid",
        ),
        (
            StoreBuilder::new()
                .metadata_schema([MetadataField::tag("x"), MetadataField::numeric("x")]),
            "duplicate",
        ),
        (
            StoreBuilder::new().metadata_schema([MetadataField::text("content")]),
            "duplicate",
        ),
    ] {
        let err = build(bad.index_name("docs").embedder(embedder())).unwrap();
        assert!(err.contains(expected), "{err}");
    }
    let store = store(&conn, StoreBuilder::new());
    assert_eq!(store.key_prefix(), "doc:docs:");
    assert_eq!(store.index_name(), "docs");
    let custom = StoreBuilder::new()
        .index_name("other")
        .key_prefix("kb:")
        .embedder(embedder())
        .build_with_connection(conn.clone())
        .unwrap();
    assert_eq!(custom.key_prefix(), "kb:");
}

#[tokio::test]
async fn first_write_creates_the_index_then_pipelines_hsets() {
    let conn = FakeRedis::new(|args| match args[0].as_str() {
        "FT.INFO" => not_found(),
        _ => Ok(RedisValue::Okay),
    });
    let store = store(
        &conn,
        StoreBuilder::new()
            .key_prefix("kb:")
            .namespace("tenant-a")
            .metadata_schema([
                MetadataField::tag("source"),
                MetadataField::numeric("year"),
                MetadataField::text("title"),
            ])
            .distance(RedisDistance::L2)
            .algorithm(VectorAlgorithm::Flat),
    );
    let docs = [
        Document::new("alpha").with_metadata(
            [
                ("source".to_string(), json!("a.txt")),
                ("year".to_string(), json!(2024)),
                ("tags".to_string(), json!(["x"])),
                ("gone".to_string(), json!(null)),
                // Reserved names can't overwrite the document.
                ("content".to_string(), json!("evil")),
                ("content_vector".to_string(), json!("evil")),
                ("namespace".to_string(), json!("other")),
                ("__vector_score".to_string(), json!(0)),
            ]
            .into(),
        ),
        Document::new("beta"),
    ];
    let keys = store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();
    // The index is checked once per store.
    store
        .add_documents(&docs[1..], &VecStoreOptions::default())
        .await
        .unwrap();

    assert_eq!(keys.len(), 2);
    assert!(keys.iter().all(|k| k.starts_with("kb:")), "{keys:?}");
    assert_ne!(keys[0], keys[1]);
    let commands = conn.commands();
    assert_eq!(
        conn.names(),
        ["FT.INFO", "FT.CREATE", "HSET", "HSET", "HSET"]
    );
    assert_eq!(commands[0], ["FT.INFO", "docs"]);
    assert_eq!(
        commands[1].join(" "),
        "FT.CREATE docs ON HASH PREFIX 1 kb: SCHEMA content TEXT namespace TAG \
         source TAG year NUMERIC title TEXT content_vector VECTOR FLAT 6 TYPE FLOAT32 DIM 2 \
         DISTANCE_METRIC L2"
    );
    let raw = conn.raw.lock().unwrap().clone();
    assert_eq!(commands[2][1], keys[0]);
    let mut without_vector = commands[2][2..].to_vec();
    without_vector.remove(3);
    assert_eq!(
        without_vector,
        [
            "content",
            "alpha",
            "content_vector",
            "namespace",
            "tenant-a",
            "source",
            "a.txt",
            "tags",
            r#"["x"]"#,
            "year",
            "2024",
        ]
    );
    assert_eq!(raw[2][5], [0, 0, 128, 63, 0, 0, 0, 0], "f32 LE vector");
    assert_eq!(commands[3][1], keys[1]);
    assert_eq!(commands[3].len(), 8, "{:?}", commands[3]);
    assert_eq!(raw[3][5], [0, 0, 0, 0, 0, 0, 128, 63]);

    // A per-call namespace wins over the default one.
    let conn = FakeRedis::new(|_| Ok(RedisValue::Okay));
    store_with_ready_index(&conn, StoreBuilder::new().namespace("default"))
        .add_documents(
            &[Document::new("alpha")],
            &VecStoreOptions::default().with_name_space("call"),
        )
        .await
        .unwrap();
    assert_eq!(conn.commands()[0][6..], ["namespace", "call"]);
}

fn store_with_ready_index(conn: &FakeRedis, builder: StoreBuilder) -> Store<FakeRedis> {
    let store = store(conn, builder);
    store
        .index_ready
        .store(true, std::sync::atomic::Ordering::Release);
    store
}

#[tokio::test]
async fn existing_index_is_reused_and_a_racing_create_is_fine() {
    let conn = FakeRedis::new(|args| match args[0].as_str() {
        "FT.INFO" => Ok(RedisValue::Array(vec![bulk("index_name"), bulk("docs")])),
        _ => Ok(RedisValue::Okay),
    });
    store(&conn, StoreBuilder::new())
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(conn.names(), ["FT.INFO", "HSET"]);

    let conn = FakeRedis::new(|args| match args[0].as_str() {
        "FT.INFO" => not_found(),
        "FT.CREATE" => Err(make_extension_error(
            "SEARCH_INDEX_EXISTS".into(),
            Some("Index already exists".into()),
        )),
        _ => Ok(RedisValue::Okay),
    });
    store(&conn, StoreBuilder::new())
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(conn.names(), ["FT.INFO", "FT.CREATE", "HSET"]);

    // Redis Stack 7 says "Unknown index name" instead of an error code.
    let conn = FakeRedis::new(|args| match args[0].as_str() {
        "FT.INFO" => Err(make_extension_error(
            "ERR".into(),
            Some("Unknown index name".into()),
        )),
        _ => Ok(RedisValue::Okay),
    });
    store(&conn, StoreBuilder::new())
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(conn.names(), ["FT.INFO", "FT.CREATE", "HSET"]);
    // Defaults: HNSW with cosine distance.
    assert_eq!(
        conn.commands()[1][12..].join(" "),
        "content_vector VECTOR HNSW 6 TYPE FLOAT32 DIM 2 DISTANCE_METRIC COSINE"
    );
}

#[tokio::test]
async fn write_errors_are_returned() {
    let conn = FakeRedis::new(|args| match args[0].as_str() {
        "FT.INFO" => Err(make_extension_error(
            "NOPERM".into(),
            Some("this user has no permissions to run the 'ft.info' command".into()),
        )),
        _ => Ok(RedisValue::Okay),
    });
    let err = store(&conn, StoreBuilder::new())
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("no permissions"), "{err}");
    assert_eq!(conn.names(), ["FT.INFO"]);

    let conn = FakeRedis::new(|args| match args[0].as_str() {
        "FT.INFO" => not_found(),
        "FT.CREATE" => Err(make_extension_error(
            "ERR".into(),
            Some("unknown command 'FT.CREATE'".into()),
        )),
        _ => Ok(RedisValue::Okay),
    });
    let store = store(&conn, StoreBuilder::new());
    let err = store
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("unknown command"), "{err}");
    assert_eq!(conn.names(), ["FT.INFO", "FT.CREATE"]);
    // A failed creation isn't remembered as done.
    let _ = store
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await;
    assert_eq!(
        conn.names(),
        ["FT.INFO", "FT.CREATE", "FT.INFO", "FT.CREATE"]
    );

    let conn = FakeRedis::new(|args| match args[0].as_str() {
        "HSET" => Err(make_extension_error(
            "OOM".into(),
            Some("command not allowed when used memory > 'maxmemory'".into()),
        )),
        _ => Ok(RedisValue::Okay),
    });
    let err = store_with_ready_index(&conn, StoreBuilder::new())
        .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("maxmemory"), "{err}");

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
    let conn = FakeRedis::new(|_| Ok(RedisValue::Okay));
    let err = store_with_ready_index(&conn, StoreBuilder::new())
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
    assert!(conn.names().is_empty());
}

#[tokio::test]
async fn writes_are_batched_and_empty_input_does_nothing() {
    let conn = FakeRedis::new(|args| match args[0].as_str() {
        "FT.INFO" => Ok(RedisValue::Array(vec![])),
        _ => Ok(RedisValue::Int(4)),
    });
    let store = store(&conn, StoreBuilder::new());
    assert!(store
        .add_documents(
            &[],
            &VecStoreOptions::default().with_embedder(PanickingEmbedder)
        )
        .await
        .unwrap()
        .is_empty());
    assert!(conn.names().is_empty());

    let docs: Vec<Document> = (0..1201)
        .map(|i| Document::new(format!("doc {i}")))
        .collect();
    let keys = store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(keys.len(), 1201);
    let hsets: Vec<String> = conn
        .commands()
        .into_iter()
        .filter(|c| c[0] == "HSET")
        .map(|c| c[1].clone())
        .collect();
    assert_eq!(hsets, keys, "every key written once, in order");
}

#[tokio::test]
async fn similarity_search_builds_the_knn_query_and_parses_resp2() {
    let conn = FakeRedis::new(|args| match args[0].as_str() {
        "FT.SEARCH" => Ok(RedisValue::Array(vec![
            RedisValue::Int(2),
            bulk("doc:docs:1"),
            RedisValue::Array(vec![
                bulk("__vector_score"),
                bulk("0.1"),
                bulk("content_vector"),
                // Valid UTF-8 on purpose: it must be dropped by name, not by decoding.
                RedisValue::BulkString(vec![0; 8]),
                bulk("content"),
                bulk("alpha"),
                bulk("source"),
                bulk("a.txt"),
                bulk("year"),
                bulk("2024"),
                bulk("zip"),
                bulk("12345"),
            ]),
            bulk("doc:docs:2"),
            RedisValue::Array(vec![
                bulk("__vector_score"),
                bulk("0.5"),
                bulk("content"),
                bulk("beta"),
            ]),
        ])),
        _ => Ok(RedisValue::Okay),
    });
    let store = store(
        &conn,
        StoreBuilder::new().metadata_schema([MetadataField::numeric("year")]),
    );
    let docs = store
        .similarity_search("query", 2, &VecStoreOptions::default())
        .await
        .unwrap();

    let commands = conn.commands();
    assert_eq!(conn.names(), ["FT.SEARCH"], "no index check on reads");
    let search = &commands[0];
    assert_eq!(
        search[1..6],
        [
            "docs",
            "*=>[KNN 2 @content_vector $vector AS __vector_score]",
            "PARAMS",
            "2",
            "vector"
        ]
    );
    assert_eq!(
        conn.raw.lock().unwrap()[0][6],
        [154, 153, 25, 63, 205, 204, 76, 63],
        "query vector as f32 LE"
    );
    assert_eq!(
        search[7..].join(" "),
        "SORTBY __vector_score ASC LIMIT 0 2 DIALECT 2"
    );

    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].page_content, "alpha");
    assert!((docs[0].score - 0.9).abs() < 1e-12, "{}", docs[0].score);
    assert_eq!(
        docs[0].metadata,
        HashMap::<String, Value>::from([
            ("source".to_string(), json!("a.txt")),
            ("year".to_string(), json!(2024)),
            // Not declared numeric, so it stays a string.
            ("zip".to_string(), json!("12345")),
        ])
    );
    assert_eq!(docs[1].page_content, "beta");
    assert_eq!(docs[1].score, 0.5);
    assert!(docs[1].metadata.is_empty());

    // A per-call embedder embeds the query.
    store
        .similarity_search(
            "unknown",
            2,
            &VecStoreOptions::default()
                .with_embedder(FakeEmbedder::new(2, &[("unknown", vec![1.0, 0.0])])),
        )
        .await
        .unwrap();
    assert_eq!(
        conn.raw.lock().unwrap()[1][6],
        [0, 0, 128, 63, 0, 0, 0, 0],
        "per-call embedder vector"
    );
}

#[tokio::test]
async fn similarity_search_parses_resp3_and_applies_the_threshold() {
    let hit = |key: &str, score: &str, content: &str| {
        RedisValue::Map(vec![
            (bulk("id"), bulk(key)),
            (
                bulk("extra_attributes"),
                RedisValue::Map(vec![
                    (bulk("__vector_score"), bulk(score)),
                    (bulk("content"), bulk(content)),
                ]),
            ),
            (bulk("values"), RedisValue::Array(vec![])),
        ])
    };
    let reply = RedisValue::Map(vec![
        (bulk("attributes"), RedisValue::Array(vec![])),
        (bulk("total_results"), RedisValue::Int(2)),
        (
            bulk("results"),
            RedisValue::Array(vec![
                hit("doc:docs:1", "0.25", "alpha"),
                hit("doc:docs:2", "0.26", "beta"),
            ]),
        ),
    ]);
    let conn = FakeRedis::new(move |_| Ok(reply.clone()));
    let store = store(&conn, StoreBuilder::new());
    let all = store
        .similarity_search("query", 5, &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[1].page_content, "beta");
    let docs = store
        .similarity_search(
            "query",
            5,
            &VecStoreOptions::default().with_score_threshold(0.75),
        )
        .await
        .unwrap();
    assert_eq!(
        docs.iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>(),
        ["alpha"],
        "a score equal to the threshold is kept"
    );
}

#[tokio::test]
async fn filters_namespaces_and_their_escaping() {
    let conn = FakeRedis::new(|_| Ok(RedisValue::Array(vec![RedisValue::Int(0)])));
    let store = store(&conn, StoreBuilder::new().namespace("default-ns"));
    let query = |opt: VecStoreOptions<Value>| {
        let store = &store;
        let conn = conn.clone();
        async move {
            store.similarity_search("query", 3, &opt).await.unwrap();
            conn.commands().last().unwrap()[2].clone()
        }
    };

    assert_eq!(
        query(VecStoreOptions::default()).await,
        r"(@namespace:{default\-ns})=>[KNN 3 @content_vector $vector AS __vector_score]"
    );
    assert_eq!(
        query(
            VecStoreOptions::default()
                .with_name_space("t 1")
                .with_filters(json!({ "year": 2024, "source": "a b.txt", "draft": false }))
        )
        .await,
        r"(@namespace:{t\ 1} @draft:{false} @source:{a\ b\.txt} @year:[2024 2024])=>[KNN 3 @content_vector $vector AS __vector_score]"
    );
    assert_eq!(
        query(VecStoreOptions::default().with_filters(json!("@year:[2020 +inf] | @source:{x}")))
            .await,
        r"(@namespace:{default\-ns} (@year:[2020 +inf] | @source:{x}))=>[KNN 3 @content_vector $vector AS __vector_score]"
    );

    let plain = super::store::Store::filter_expression;
    let no_ns = self::store(&conn, StoreBuilder::new());
    assert_eq!(plain(&no_ns, None, Some(&json!("*"))).unwrap(), "*");
    assert_eq!(plain(&no_ns, None, Some(&json!(" "))).unwrap(), "*");
    assert_eq!(plain(&no_ns, None, Some(&json!(null))).unwrap(), "*");
    assert_eq!(plain(&no_ns, None, None).unwrap(), "*");
    assert_eq!(
        plain(&no_ns, None, Some(&json!({ "n": 1.5 }))).unwrap(),
        "(@n:[1.5 1.5])"
    );
    for bad in [json!({ "a-b": 1 }), json!({ "a": [1] }), json!(3)] {
        let err = store
            .similarity_search(
                "query",
                3,
                &VecStoreOptions::default().with_filters(bad.clone()),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Redis filter"), "{bad}: {err}");
    }
}

#[tokio::test]
async fn search_edge_cases() {
    // No index yet: empty, not an error.
    let conn = FakeRedis::new(|_| not_found());
    assert!(store(&conn, StoreBuilder::new())
        .similarity_search("query", 3, &VecStoreOptions::default())
        .await
        .unwrap()
        .is_empty());

    // limit 0 never reaches Redis or the embedder.
    let conn = FakeRedis::new(|_| panic!("no command expected"));
    assert!(store(&conn, StoreBuilder::new())
        .similarity_search(
            "query",
            0,
            &VecStoreOptions::default().with_embedder(PanickingEmbedder)
        )
        .await
        .unwrap()
        .is_empty());

    // Other errors and malformed replies surface.
    let conn = FakeRedis::new(|_| {
        Err(make_extension_error(
            "SEARCH_SYNTAX".into(),
            Some("Syntax error at offset 3".into()),
        ))
    });
    let err = store(&conn, StoreBuilder::new())
        .similarity_search("query", 3, &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("Syntax error"), "{err}");

    for reply in [
        RedisValue::Int(3),
        RedisValue::Array(vec![RedisValue::Int(1), bulk("doc:docs:1")]),
        RedisValue::Map(vec![(bulk("total_results"), RedisValue::Int(0))]),
    ] {
        let conn = FakeRedis::new(move |_| Ok(reply.clone()));
        let err = store(&conn, StoreBuilder::new())
            .similarity_search("query", 3, &VecStoreOptions::default())
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("unexpected FT.SEARCH reply"),
            "{err}"
        );
    }

    let conn = FakeRedis::new(|_| {
        Ok(RedisValue::Array(vec![
            RedisValue::Int(1),
            bulk("doc:docs:1"),
            RedisValue::Array(vec![bulk("content"), bulk("alpha")]),
        ]))
    });
    let err = store(&conn, StoreBuilder::new())
        .similarity_search("query", 3, &VecStoreOptions::default())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("without __vector_score"), "{err}");
}

/// Against Redis 8 (or Redis Stack):
///
/// ```sh
/// docker run -d -p 6379:6379 redis:8
/// REDIS_VECTOR_TEST_URL=redis://127.0.0.1:6379/ cargo test --features redis -- --ignored redis_vector_live
/// ```
#[tokio::test]
#[ignore = "needs REDIS_VECTOR_TEST_URL"]
async fn redis_vector_live_round_trip() {
    let url = std::env::var("REDIS_VECTOR_TEST_URL").expect("REDIS_VECTOR_TEST_URL");
    let index = format!("lcr_test_{}", uuid::Uuid::new_v4().simple());
    let store = StoreBuilder::new()
        .url(&url)
        .index_name(&index)
        .embedder(embedder())
        .metadata_schema([MetadataField::tag("source"), MetadataField::numeric("year")])
        .build()
        .await
        .unwrap();

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
    // A namespace sharing the prefix must not leak into "tenant-a".
    store
        .add_documents(&[Document::new("alpha")], &tenant("tenant-ab"))
        .await
        .unwrap();

    let found = store
        .similarity_search("alpha", 5, &tenant("tenant-a"))
        .await
        .unwrap();
    assert_eq!(
        found
            .iter()
            .map(|d| d.page_content.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "beta"]
    );
    assert!((found[0].score - 1.0).abs() < 1e-5, "{}", found[0].score);
    assert!(found[1].score.abs() < 1e-5, "{}", found[1].score);
    assert_eq!(found[0].metadata["source"], json!("a.txt"));
    assert_eq!(found[0].metadata["year"], json!(2024));
    assert_eq!(found[0].metadata["tags"], json!(r#"["x"]"#));
    assert_eq!(found[0].metadata["namespace"], json!("tenant-a"));
    assert!(!found[0].metadata.contains_key("content_vector"));

    let by_tag = store
        .similarity_search(
            "alpha",
            5,
            &tenant("tenant-a").with_filters(json!({ "source": "b.txt" })),
        )
        .await
        .unwrap();
    assert_eq!(by_tag.len(), 1);
    assert_eq!(by_tag[0].page_content, "beta");

    let by_range = store
        .similarity_search(
            "beta",
            5,
            &tenant("tenant-a").with_filters(json!("@year:[2020 +inf]")),
        )
        .await
        .unwrap();
    assert_eq!(by_range.len(), 1);
    assert_eq!(by_range[0].page_content, "alpha");

    // Without a namespace every document matches; FT.SEARCH would stop at 10 without LIMIT.
    let many: Vec<Document> = (0..12).map(|i| Document::new(format!("doc {i}"))).collect();
    store
        .add_documents(&many, &VecStoreOptions::default())
        .await
        .unwrap();
    let all = store
        .similarity_search("alpha", 20, &VecStoreOptions::default())
        .await
        .unwrap();
    assert_eq!(all.len(), 15);

    let client = ::redis::Client::open(url.as_str()).unwrap();
    let mut conn = client.get_multiplexed_async_connection().await.unwrap();
    let () = ::redis::cmd("FT.DROPINDEX")
        .arg(&index)
        .arg("DD")
        .query_async(&mut conn)
        .await
        .unwrap();
}
