use async_trait::async_trait;
use opensearch::http::request::JsonBody;
use opensearch::http::response::Response;
use opensearch::indices::{IndicesCreateParts, IndicesDeleteParts};
use opensearch::{BulkParts, SearchParts};
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
    vectorstore::{VecStoreOptions, VectorStore},
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
            let operation = json!({"index": {}});
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

        let ids = response_body["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| serde_json::from_value::<String>(item["index"]["_id"].clone()).unwrap())
            .collect::<Vec<_>>();

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
}

/// Turns an OpenSearch `_search` response body into documents.
fn parse_search_hits(body: &Value, content_field: &str) -> Result<Vec<Document>, Box<dyn Error>> {
    let hits = body
        .pointer("/hits/hits")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("unexpected opensearch response, missing hits.hits: {body}"))?;

    hits.iter()
        .map(|hit| {
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
            Ok(Document {
                page_content,
                metadata,
                score,
            })
        })
        .collect()
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
}
