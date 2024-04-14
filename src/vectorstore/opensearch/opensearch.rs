use async_trait::async_trait;
use opensearch::http::request::JsonBody;
use opensearch::http::response::Response;
use opensearch::indices::{IndicesCreateParts, IndicesDeleteParts};
pub use opensearch::OpenSearch;
use opensearch::{BulkParts, SearchParts};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

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
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &VecStoreOptions,
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
        opt: &VecStoreOptions,
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
            .size(3)
            .body(query)
            .send()
            .await?;

        let response_body = response.json::<Value>().await?;

        let aoss_documents = response_body["hits"]["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|raw_value| {
                serde_json::from_value::<HashMap<String, Value>>(raw_value.clone()).unwrap()
            })
            .collect::<Vec<_>>();

        let documents = aoss_documents
            .into_iter()
            .map(|item| {
                let page_content =
                    serde_json::from_value::<String>(item["_source"][&self.content_field].clone())
                        .unwrap();
                let metadata = serde_json::from_value::<HashMap<String, Value>>(
                    item["_source"]["metadata"].clone(),
                )
                .unwrap();
                let score = serde_json::from_value::<f64>(item["_score"].clone()).unwrap();
                Document {
                    page_content,
                    metadata,
                    score,
                }
            })
            .collect();

        Ok(documents)
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
