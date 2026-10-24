use async_trait::async_trait;
use opensearch::http::request::JsonBody;
use opensearch::http::response::Response;
use opensearch::indices::{IndicesCreateParts, IndicesDeleteParts};
use opensearch::{BulkParts, MgetParts, SearchParts};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

pub use opensearch::auth::Credentials;
pub use opensearch::cert::CertificateValidation;
pub use opensearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
pub use opensearch::OpenSearch;

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{vectorstore::in_id_order, VecStoreOptions, VectorStore},
};

pub struct Store {
    pub client: OpenSearch,
    pub embedder: Arc<dyn Embedder>,
    pub k: i32,
    pub index: String,
    pub vector_field: String,
    pub content_field: String,
}

// https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/
// https://opensearch.org/blog/efficient-filters-in-knn/
// https://opensearch.org/docs/latest/clients/rust/

impl Store {
    pub async fn delete_index(&self) -> Result<Response, Box<dyn Error>> {
        let response = self
            .client
            .indices()
            .delete(IndicesDeleteParts::Index(&[&self.index]))
            .send()
            .await?;

        let result = response.error_for_status_code().map_err(|e| Box::new(e))?;

        Ok(result)
    }

    pub async fn create_index(&self) -> Result<Response, Box<dyn Error>> {
        let body = json!({
            "settings": {
                "index.knn": true,
                "knn.algo_param": {
                    "ef_search": "512"
                },
            },
            "mappings": {
                "properties": {
                    &self.vector_field: {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "engine": "faiss",
                            "name": "hnsw",
                            "space_type": "l2",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 16
                            }
                        }
                    },
                    &self.content_field: {
                        "type": "text"
                    },
                    "metadata": {
                        "properties": {
                            "source": {
                                "type": "text",
                            }
                        }
                    }
                }
            }
        });

        let response = self
            .client
            .indices()
            .create(IndicesCreateParts::Index(&self.index))
            .body(body)
            .send()
            .await?;

        let result = response.error_for_status_code().map_err(|e| Box::new(e))?;

        Ok(result)
    }
}

#[async_trait]
impl VectorStore for Store {
    type Options = VecStoreOptions<Value>;

    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &Self::Options,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        // Reject before embedding or sending so a bad id cannot look successful.
        for doc in docs {
            if doc.id.as_ref().is_some_and(|id| id.is_empty()) {
                return Err("document id must not be empty".into());
            }
        }

        let texts: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let vectors = embedder.embed_documents(&texts).await?;

        if vectors.len() != docs.len() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Number of vectors and documents do not match",
            )));
        }

        let mut body: Vec<JsonBody<_>> = Vec::with_capacity(docs.len() * 2);

        for (doc, vector) in docs.iter().zip(vectors.iter()) {
            // `index` replaces an existing `_id`. Omit the key when the caller did
            // not supply one so OpenSearch still generates an id. Never send `_id: null`.
            let operation = match &doc.id {
                Some(id) => json!({"index": {"_id": id}}),
                None => json!({"index": {}}),
            };
            body.push(operation.into());

            let document = json!({
                &self.content_field: doc.page_content,
                "metadata": doc.metadata,
                &self.vector_field: vector,
            });
            body.push(document.into());
        }

        let response = self
            .client
            .bulk(BulkParts::Index(&self.index))
            .body(body)
            .send()
            .await?
            .error_for_status_code()
            .map_err(|e| Box::new(e))?;

        let response_body = response.json::<Value>().await?;
        let items = response_body
            .get("items")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("opensearch bulk response missing items: {response_body}"))?;

        let mut ids = Vec::with_capacity(items.len());
        for item in items {
            ids.push(bulk_item_id(item)?);
        }
        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &Self::Options,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let query_vector = self.embedder.embed_query(query).await?;
        let query = build_similarity_search_query(
            query_vector,
            &self.vector_field,
            limit,
            self.k,
            opt.filters.clone(),
        );

        let response = self
            .client
            .search(SearchParts::Index(&[&self.index]))
            .from(0)
            .size(limit as i64)
            .body(query)
            .send()
            .await?;

        let status = response.status_code();
        let response_body = response.json::<Value>().await?;
        if !status.is_success() {
            return Err(format!("opensearch search failed ({status}): {response_body}").into());
        }

        parse_search_hits(&response_body, &self.content_field)
    }

    /// `_bulk` deletes, 100 ids per request. Ids that don't exist (and a missing index)
    /// are fine. `filters` and `name_space` are rejected: they can't narrow a delete by
    /// id here, and silently ignoring a scope on a delete would be worse.
    async fn delete(&self, ids: &[String], opt: &Self::Options) -> Result<(), Box<dyn Error>> {
        reject_id_options(opt)?;
        for batch in ids.chunks(ID_BATCH_SIZE) {
            let body: Vec<JsonBody<Value>> = batch
                .iter()
                .map(|id| json!({"delete": {"_id": id}}).into())
                .collect();
            let response = self
                .client
                .bulk(BulkParts::Index(&self.index))
                .body(body)
                .send()
                .await?;
            let response_body = success_json(response, "delete").await?;
            let items = response_body
                .get("items")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    format!("opensearch bulk response missing items: {response_body}")
                })?;
            for item in items {
                let result = &item["delete"];
                let failed = match result["status"].as_u64() {
                    Some(404) => false,
                    Some(status) => status >= 400 || result.get("error").is_some(),
                    None => true,
                };
                if failed {
                    return Err(format!("opensearch delete failed: {result}").into());
                }
            }
        }
        Ok(())
    }

    /// `_mget` without the vector field, 100 ids per request; documents come back in
    /// request order (score 0). A missing index has no documents.
    async fn get_by_ids(
        &self,
        ids: &[String],
        opt: &Self::Options,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        reject_id_options(opt)?;
        let excludes = [self.vector_field.as_str()];
        let mut docs = Vec::with_capacity(ids.len());
        for batch in ids.chunks(ID_BATCH_SIZE) {
            let response = self
                .client
                .mget(MgetParts::Index(&self.index))
                ._source_excludes(&excludes)
                .body(json!({ "ids": batch }))
                .send()
                .await?;
            if response.status_code().as_u16() == 404 {
                let body = response.text().await?;
                if body.contains("index_not_found_exception") {
                    return Ok(Vec::new());
                }
                return Err(format!("opensearch mget failed (404): {body}").into());
            }
            let response_body = success_json(response, "mget").await?;
            docs.extend(parse_mget_docs(&response_body, &self.content_field)?);
        }
        Ok(in_id_order(ids, docs))
    }
}

/// Ids per `_bulk` delete / `_mget` request.
const ID_BATCH_SIZE: usize = 100;

fn reject_id_options(opt: &VecStoreOptions<Value>) -> Result<(), Box<dyn Error>> {
    if opt.filters.is_some() || opt.name_space.is_some() {
        return Err("opensearch delete/get_by_ids don't support filters or name_space".into());
    }
    Ok(())
}

/// The JSON body of a successful response; any other status is an error carrying the
/// raw body (which may not be JSON, e.g. from a proxy).
async fn success_json(response: Response, what: &str) -> Result<Value, Box<dyn Error>> {
    let status = response.status_code();
    let body = response.text().await?;
    if !status.is_success() {
        return Err(format!("opensearch {what} failed ({status}): {body}").into());
    }
    serde_json::from_str(&body)
        .map_err(|e| format!("opensearch {what}: invalid JSON response ({e}): {body}").into())
}

/// The found documents of an `_mget` response.
fn parse_mget_docs(body: &Value, content_field: &str) -> Result<Vec<Document>, Box<dyn Error>> {
    let docs = body
        .get("docs")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("unexpected opensearch mget response, missing docs: {body}"))?;
    docs.iter()
        .filter(|doc| doc["found"].as_bool() == Some(true))
        .map(|doc| {
            let mut hit = doc.clone();
            hit["_score"] = json!(0.0);
            parse_hit(&hit, content_field)
        })
        .collect()
}

/// Turns an OpenSearch `_search` response body into documents.
fn parse_search_hits(body: &Value, content_field: &str) -> Result<Vec<Document>, Box<dyn Error>> {
    let hits = body
        .pointer("/hits/hits")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("unexpected opensearch response, missing hits.hits: {body}"))?;

    hits.iter()
        .map(|hit| parse_hit(hit, content_field))
        .collect()
}

fn parse_hit(hit: &Value, content_field: &str) -> Result<Document, Box<dyn Error>> {
    let source = &hit["_source"];
    let page_content = source[content_field]
        .as_str()
        .ok_or_else(|| format!("hit has no string field `{content_field}`: {hit}"))?
        .to_string();
    let metadata = match &source["metadata"] {
        Value::Object(map) => map.clone().into_iter().collect(),
        _ => HashMap::new(),
    };
    let score = hit["_score"].as_f64().unwrap_or_default();
    let id = match hit.get("_id") {
        Some(Value::String(id)) => Some(id.clone()),
        _ => None,
    };
    Ok(Document {
        page_content,
        metadata,
        score,
        id,
    })
}

/// Id of one successful bulk item, in request order.
///
/// A per-item error or a missing/non-string `_id` is a `Result` error. A rejected
/// custom id must not be reported as inserted.
fn bulk_item_id(item: &Value) -> Result<String, Box<dyn Error>> {
    let index = item
        .get("index")
        .ok_or_else(|| format!("opensearch bulk item missing index result: {item}"))?;
    if let Some(error) = index.get("error").filter(|error| !error.is_null()) {
        return Err(format!("opensearch bulk item failed: {error}").into());
    }
    if let Some(status) = index.get("status").and_then(Value::as_u64) {
        if status >= 400 {
            return Err(format!("opensearch bulk item failed ({status}): {index}").into());
        }
    }
    match index.get("_id") {
        Some(Value::String(id)) => Ok(id.clone()),
        _ => Err(format!("opensearch bulk item missing string _id: {item}").into()),
    }
}

fn build_similarity_search_query(
    embedded_query: Vec<f64>,
    vector_field: &str,
    size: usize,
    k: i32,
    maybe_filter: Option<Value>,
) -> Value {
    match maybe_filter {
        Some(filter) => {
            json!({
              "size": size,
              "query": {
                "knn": {
                  vector_field: {
                    "vector": embedded_query,
                    "k": k,
                    "filter": filter,
                  }
                }
              }
            })
        }
        None => {
            json!({
              "size": size,
              "query": {
                "knn": {
                  vector_field: {
                    "vector": embedded_query,
                    "k": k,
                  }
                }
              }
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_search_hits;
    use serde_json::json;

    #[test]
    fn parses_hits_into_documents() {
        let body = json!({"hits": {"hits": [
            {"_score": 0.9, "_source": {"text": "rust", "metadata": {"lang": "en"}}},
            {"_score": 0.5, "_source": {"text": "go"}}
        ]}});
        let docs = parse_search_hits(&body, "text").unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].page_content, "rust");
        assert_eq!(docs[0].score, 0.9);
        assert_eq!(docs[0].metadata["lang"], json!("en"));
        assert!(docs[1].metadata.is_empty());
        assert!(docs[0].id.is_none());
        assert!(docs[1].id.is_none());
    }

    #[test]
    fn search_hits_keep_a_string_id_and_ignore_a_missing_one() {
        let body = json!({"hits": {"hits": [
            {"_id": "custom-1", "_score": 1.0, "_source": {"text": "rust", "metadata": {"lang": "en"}}},
            {"_score": 0.2, "_source": {"text": "go", "metadata": {"lang": "en"}}}
        ]}});
        let docs = parse_search_hits(&body, "text").unwrap();
        assert_eq!(docs[0].id.as_deref(), Some("custom-1"));
        assert_eq!(docs[0].metadata["lang"], json!("en"));
        assert!(docs[1].id.is_none());
        assert_eq!(docs[1].page_content, "go");
    }

    // #308: an error body (or a mapping without the content field) used to panic.
    #[test]
    fn error_bodies_are_errors_not_panics() {
        let err = json!({"error": {"type": "index_not_found_exception"}, "status": 404});
        assert!(parse_search_hits(&err, "text").is_err());

        let wrong_field = json!({"hits": {"hits": [{"_score": 1.0, "_source": {"body": "x"}}]}});
        let msg = parse_search_hits(&wrong_field, "text")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("`text`"), "{msg}");
    }

    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use opensearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
    use opensearch::OpenSearch;
    use url::Url;

    use crate::embedding::{Embedder, EmbedderError};
    use crate::schemas::Document;
    use crate::test_utils::FakeEmbedder;
    use crate::vectorstore::{VecStoreOptions, VectorStore};

    use super::{CertificateValidation, Store};

    fn client_for(server: &mockito::ServerGuard) -> OpenSearch {
        let url = Url::parse(&server.url()).unwrap();
        let pool = SingleNodeConnectionPool::new(url);
        let transport = TransportBuilder::new(pool)
            .disable_proxy()
            .cert_validation(CertificateValidation::None)
            .build()
            .unwrap();
        OpenSearch::new(transport)
    }

    fn store_for(server: &mockito::ServerGuard, embedder: impl Embedder + 'static) -> Store {
        Store {
            client: client_for(server),
            embedder: Arc::new(embedder),
            k: 2,
            index: "custom_index".to_string(),
            vector_field: "embedding_vec".to_string(),
            content_field: "body_text".to_string(),
        }
    }

    fn embedder() -> FakeEmbedder {
        FakeEmbedder::new(
            2,
            &[
                ("alpha", vec![0.25, 0.5]),
                ("beta", vec![0.75, 1.0]),
                ("gamma", vec![0.1, 0.2]),
            ],
        )
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

    fn ndjson(body: &str) -> Vec<serde_json::Value> {
        body.lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).unwrap_or_else(|err| panic!("{line}: {err}")))
            .collect()
    }

    #[tokio::test]
    async fn bulk_request_mixes_explicit_and_generated_ids() {
        let mut server = mockito::Server::new_async().await;
        let captured = Arc::new(Mutex::new(String::new()));
        let captured_match = captured.clone();
        server
            .mock("POST", "/custom_index/_bulk")
            .match_request(move |request| {
                *captured_match.lock().unwrap() = request.utf8_lossy_body().unwrap().to_string();
                true
            })
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "errors": false,
                    "items": [
                        {"index": {"_id": "custom-1", "status": 201, "result": "created"}},
                        {"index": {"_id": "server-gen-2", "status": 201, "result": "created"}}
                    ]
                })
                .to_string(),
            )
            .create_async()
            .await;

        let store = store_for(&server, embedder());
        let docs = vec![
            Document::new("alpha").with_id("custom-1").with_metadata(
                std::collections::HashMap::from([("source".to_string(), json!("cli"))]),
            ),
            Document::new("beta"),
        ];
        let ids = store
            .add_documents(&docs, &VecStoreOptions::default())
            .await
            .unwrap();
        assert_eq!(
            ids,
            vec!["custom-1".to_string(), "server-gen-2".to_string()]
        );

        let body = captured.lock().unwrap().clone();
        assert!(!body.contains("\"_id\":null"), "{body}");
        assert!(!body.contains("\"_id\": null"), "{body}");
        let lines = ndjson(&body);
        assert_eq!(lines.len(), 4, "{body}");
        assert_eq!(lines[0]["index"]["_id"], json!("custom-1"));
        assert!(lines[1].get("_id").is_none(), "{body}");
        assert_eq!(lines[1]["body_text"], json!("alpha"));
        assert_eq!(lines[1]["metadata"], json!({"source": "cli"}));
        assert_eq!(lines[1]["embedding_vec"], json!([0.25, 0.5]));
        assert!(lines[2]["index"].get("_id").is_none(), "{body}");
        assert_eq!(lines[2]["index"], json!({}));
        assert_eq!(lines[3]["body_text"], json!("beta"));
        assert_eq!(lines[3]["metadata"], json!({}));
        assert_eq!(lines[3]["embedding_vec"], json!([0.75, 1.0]));
        assert!(lines[3].get("_id").is_none());
    }

    #[tokio::test]
    async fn bulk_item_and_http_failures_are_errors() {
        let cases = [
            json!({"errors": true, "items": [
                {"index": {"_id": "custom-1", "status": 201}},
                {"index": {"status": 400, "error": {"type": "mapper_parsing_exception", "reason": "bad id"}}}
            ]}),
            json!({"errors": false, "items": [{"index": {"status": 201}}]}),
            json!({"errors": false, "items": [{"index": {"_id": 12, "status": 201}}]}),
            json!({"errors": false, "items": [{"index": {"_id": null, "status": 201}}]}),
            json!({"errors": false}),
        ];
        for body in cases {
            let mut server = mockito::Server::new_async().await;
            server
                .mock("POST", "/custom_index/_bulk")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(body.to_string())
                .create_async()
                .await;
            let store = store_for(&server, embedder());
            let err = store
                .add_documents(
                    &[Document::new("alpha").with_id("custom-1")],
                    &VecStoreOptions::default(),
                )
                .await
                .unwrap_err();
            assert!(!err.to_string().is_empty(), "{body}");
        }

        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/custom_index/_bulk")
            .with_status(500)
            .with_body("unavailable")
            .create_async()
            .await;
        let store = store_for(&server, embedder());
        let err = store
            .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
            .await
            .unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[tokio::test]
    async fn empty_id_and_embed_errors_do_not_send_bulk() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", mockito::Matcher::Any)
            .expect(0)
            .with_status(500)
            .create_async()
            .await;

        let store = store_for(&server, embedder());
        let err = store
            .add_documents(
                &[Document::new("alpha").with_id("")],
                &VecStoreOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("empty"), "{err}");

        let store = store_for(&server, FailEmbedder);
        let err = store
            .add_documents(&[Document::new("alpha")], &VecStoreOptions::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("embed failed"), "{err}");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn get_by_ids_uses_mget_without_vectors() {
        let mut server = mockito::Server::new_async().await;
        let mget = server
            .mock("POST", "/custom_index/_mget")
            .match_query(mockito::Matcher::UrlEncoded(
                "_source_excludes".into(),
                "embedding_vec".into(),
            ))
            .match_body(mockito::Matcher::Json(json!({"ids": ["b", "nope", "a"]})))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({"docs": [
                    {"_id": "b", "found": true, "_source": {"body_text": "beta", "metadata": {"k": 1}}},
                    {"_id": "nope", "found": false},
                    {"_id": "a", "found": true, "_source": {"body_text": "alpha"}},
                ]})
                .to_string(),
            )
            .create_async()
            .await;
        let store = store_for(&server, embedder());
        let ids = ["b", "nope", "a"].map(String::from);
        let docs = store
            .get_by_ids(&ids, &VecStoreOptions::default())
            .await
            .unwrap();
        mget.assert_async().await;
        let got: Vec<_> = docs.iter().map(|d| d.id.as_deref().unwrap()).collect();
        assert_eq!(got, ["b", "a"]);
        assert_eq!(docs[0].page_content, "beta");
        assert_eq!(docs[0].metadata["k"], json!(1));
        assert_eq!(docs[0].score, 0.0);
        assert!(docs[1].metadata.is_empty());

        // A missing index has no documents; other 404s and errors are reported.
        for (body, missing) in [
            (
                json!({"error": {"type": "index_not_found_exception"}, "status": 404}),
                true,
            ),
            (json!({"error": "no such route"}), false),
        ] {
            let mut server = mockito::Server::new_async().await;
            server
                .mock("POST", "/custom_index/_mget")
                .with_status(404)
                .with_header("content-type", "application/json")
                .match_query(mockito::Matcher::Any)
                .with_body(body.to_string())
                .create_async()
                .await;
            let got = store_for(&server, embedder())
                .get_by_ids(&ids, &VecStoreOptions::default())
                .await;
            match got {
                Ok(docs) => assert!(missing && docs.is_empty(), "{docs:?}"),
                Err(err) => assert!(!missing && err.to_string().contains("no such route")),
            }
        }

        // More than 100 ids: one `_mget` per 100.
        let mut server = mockito::Server::new_async().await;
        let batches = server
            .mock("POST", "/custom_index/_mget")
            .match_query(mockito::Matcher::Any)
            .with_header("content-type", "application/json")
            .with_body(json!({"docs": []}).to_string())
            .expect(2)
            .create_async()
            .await;
        let many: Vec<String> = (0..101).map(|i: usize| i.to_string()).collect();
        assert!(store_for(&server, embedder())
            .get_by_ids(&many, &VecStoreOptions::default())
            .await
            .unwrap()
            .is_empty());
        batches.assert_async().await;
    }

    #[tokio::test]
    async fn delete_bulk_deletes_and_tolerates_missing_ids() {
        let mut server = mockito::Server::new_async().await;
        let captured = Arc::new(Mutex::new(String::new()));
        let captured_match = captured.clone();
        server
            .mock("POST", "/custom_index/_bulk")
            .match_request(move |request| {
                *captured_match.lock().unwrap() = request.utf8_lossy_body().unwrap().to_string();
                true
            })
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({"errors": false, "items": [
                    {"delete": {"_id": "a", "status": 200, "result": "deleted"}},
                    {"delete": {"_id": "nope", "status": 404, "result": "not_found"}},
                ]})
                .to_string(),
            )
            .create_async()
            .await;
        let store = store_for(&server, embedder());
        store
            .delete(
                &["a".to_string(), "nope".to_string()],
                &VecStoreOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(
            ndjson(&captured.lock().unwrap()),
            [
                json!({"delete": {"_id": "a"}}),
                json!({"delete": {"_id": "nope"}})
            ]
        );

        for body in [
            json!({"errors": true, "items": [{"delete": {"_id": "a", "status": 403, "error": "blocked"}}]}),
            json!({"errors": true, "items": [{"delete": {"_id": "a", "error": "no status"}}]}),
            json!({"errors": true, "items": [{"delete": {"_id": "a", "status": 200, "error": "odd"}}]}),
            json!({"errors": false}),
        ] {
            let mut server = mockito::Server::new_async().await;
            server
                .mock("POST", "/custom_index/_bulk")
                .with_status(200)
                .with_header("content-type", "application/json")
                .with_body(body.to_string())
                .create_async()
                .await;
            let err = store_for(&server, embedder())
                .delete(&["a".to_string()], &VecStoreOptions::default())
                .await
                .unwrap_err();
            assert!(!err.to_string().is_empty(), "{body}");
        }

        // More than 100 ids: one `_bulk` per 100.
        let mut server = mockito::Server::new_async().await;
        let batches = server
            .mock("POST", "/custom_index/_bulk")
            .with_header("content-type", "application/json")
            .with_body(json!({"errors": false, "items": []}).to_string())
            .expect(2)
            .create_async()
            .await;
        let many: Vec<String> = (0..101).map(|i: usize| i.to_string()).collect();
        store_for(&server, embedder())
            .delete(&many, &VecStoreOptions::default())
            .await
            .unwrap();
        batches.assert_async().await;
    }

    #[tokio::test]
    async fn id_lookups_reject_scopes_and_skip_empty_input() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", mockito::Matcher::Any)
            .expect(0)
            .create_async()
            .await;
        let store = store_for(&server, embedder());
        let opt = VecStoreOptions::default();
        store.delete(&[], &opt).await.unwrap();
        assert!(store.get_by_ids(&[], &opt).await.unwrap().is_empty());
        let ids = ["a".to_string()];
        for scoped in [
            VecStoreOptions::default().with_name_space("t"),
            VecStoreOptions::default().with_filters(json!({"term": {"k": 1}})),
        ] {
            assert!(store.delete(&ids, &scoped).await.is_err());
            assert!(store.get_by_ids(&ids, &scoped).await.is_err());
        }
        mock.assert_async().await;
    }

    /// Re-index one explicit id against a local OpenSearch and check a single updated record.
    ///
    /// ```text
    /// OPENSEARCH_TEST_URL=https://localhost:9200 \
    /// cargo test --features opensearch --lib -- --ignored opensearch_custom_id_replaces_document
    /// ```
    ///
    /// `OPENSEARCH_TEST_USERNAME` / `OPENSEARCH_TEST_PASSWORD` default to the local
    /// demo cluster in `examples/vector_store_opensearch.rs`.
    #[tokio::test]
    #[ignore]
    async fn opensearch_custom_id_replaces_document() {
        let url = std::env::var("OPENSEARCH_TEST_URL")
            .unwrap_or_else(|_| "https://localhost:9200".to_string());
        let username =
            std::env::var("OPENSEARCH_TEST_USERNAME").unwrap_or_else(|_| "admin".to_string());
        let password = std::env::var("OPENSEARCH_TEST_PASSWORD")
            .unwrap_or_else(|_| "ZxPZ3cZL0ky1bzVu+~N".to_string());

        let parsed = Url::parse(&url).unwrap();
        let pool = SingleNodeConnectionPool::new(parsed);
        let mut transport = TransportBuilder::new(pool)
            .disable_proxy()
            .auth(super::Credentials::Basic(username, password));
        if url.starts_with("https") {
            transport = transport.cert_validation(CertificateValidation::None);
        }
        let client = OpenSearch::new(transport.build().unwrap());

        let index = format!("custom-ids-{}", uuid_suffix());
        let embedder = embedder();
        let store = Store {
            client,
            embedder: Arc::new(embedder),
            k: 2,
            index: index.clone(),
            vector_field: "embedding_vec".to_string(),
            content_field: "body_text".to_string(),
        };

        let outcome = reindex_custom_id(&store).await;
        let _ = store.delete_index().await;
        outcome.unwrap();

        // No index: nothing to get or delete.
        let ids = ["stable-id".to_string()];
        let opts = VecStoreOptions::default();
        assert!(store.get_by_ids(&ids, &opts).await.unwrap().is_empty());
        store.delete(&ids, &opts).await.unwrap();
        let _ = store.delete_index().await;
    }

    fn uuid_suffix() -> String {
        // Avoid a hard dependency on the uuid crate, which is feature-gated.
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("{nanos}")
    }

    async fn reindex_custom_id(store: &Store) -> Result<(), Box<dyn std::error::Error>> {
        use opensearch::indices::IndicesCreateParts;

        let _ = store.delete_index().await;
        let mapping = json!({
            "settings": {"index.knn": true},
            "mappings": {
                "properties": {
                    "embedding_vec": {
                        "type": "knn_vector",
                        "dimension": 2,
                        "method": {
                            "engine": "faiss",
                            "name": "hnsw",
                            "space_type": "l2",
                            "parameters": {"ef_construction": 128, "m": 16}
                        }
                    },
                    "body_text": {"type": "text"},
                    "metadata": {"type": "object"}
                }
            }
        });
        store
            .client
            .indices()
            .create(IndicesCreateParts::Index(&store.index))
            .body(mapping)
            .send()
            .await?
            .error_for_status_code()?;

        let opts = VecStoreOptions::default();
        let first = store
            .add_documents(
                &[Document::new("alpha").with_id("stable-id").with_metadata(
                    std::collections::HashMap::from([("rev".to_string(), json!(1))]),
                )],
                &opts,
            )
            .await?;
        assert_eq!(first, vec!["stable-id".to_string()]);

        let replaced = store
            .add_documents(
                &[Document::new("beta").with_id("stable-id").with_metadata(
                    std::collections::HashMap::from([("rev".to_string(), json!(2))]),
                )],
                &opts,
            )
            .await?;
        assert_eq!(replaced, vec!["stable-id".to_string()]);
        refresh(store).await?;

        let found = search_all(store).await?;
        let hits = found["hits"]["hits"]
            .as_array()
            .ok_or("missing hits")?
            .clone();
        if hits.len() != 1 {
            return Err(format!("expected one document after re-index, got {found}").into());
        }
        assert_eq!(hits[0]["_id"], json!("stable-id"));
        assert_eq!(hits[0]["_source"]["body_text"], json!("beta"));
        assert_eq!(hits[0]["_source"]["metadata"]["rev"], json!(2));
        assert_eq!(hits[0]["_source"]["embedding_vec"], json!([0.75, 1.0]));

        let generated = store
            .add_documents(&[Document::new("gamma")], &opts)
            .await?;
        assert_eq!(generated.len(), 1);
        assert!(!generated[0].is_empty());
        refresh(store).await?;
        let searched = store.similarity_search("gamma", 5, &opts).await?;
        let matched = searched
            .iter()
            .find(|doc| doc.page_content == "gamma")
            .ok_or("generated document was not searchable")?;
        if matched.id.as_deref() != Some(generated[0].as_str()) {
            return Err(format!("search id {:?} != returned {}", matched.id, generated[0]).into());
        }

        let ask = [
            generated[0].clone(),
            "nope".to_string(),
            "stable-id".to_string(),
        ];
        let got = store.get_by_ids(&ask, &opts).await?;
        let got_ids: Vec<_> = got.iter().map(|d| d.id.clone().unwrap()).collect();
        assert_eq!(got_ids, [generated[0].clone(), "stable-id".to_string()]);
        assert_eq!(got[1].page_content, "beta");
        assert_eq!(got[1].metadata["rev"], json!(2));
        store.delete(&ask[1..], &opts).await?;
        refresh(store).await?;
        let left = store.get_by_ids(&ask, &opts).await?;
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].id.as_deref(), Some(generated[0].as_str()));
        Ok(())
    }

    async fn refresh(store: &Store) -> Result<(), Box<dyn std::error::Error>> {
        use opensearch::indices::IndicesRefreshParts;
        store
            .client
            .indices()
            .refresh(IndicesRefreshParts::Index(&[&store.index]))
            .send()
            .await?
            .error_for_status_code()?;
        Ok(())
    }

    async fn search_all(store: &Store) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        use opensearch::SearchParts;
        let response = store
            .client
            .search(SearchParts::Index(&[&store.index]))
            .body(json!({"size": 10, "query": {"match_all": {}}}))
            .send()
            .await?
            .error_for_status_code()?;
        Ok(response.json().await?)
    }
}
