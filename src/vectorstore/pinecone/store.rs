use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{document_ids, VecStoreOptions, VectorStore},
};

/// Vectors per upsert request (Pinecone's recommended batch size; the hard limit is 1000).
pub(super) const UPSERT_BATCH_SIZE: usize = 100;

/// [Pinecone](https://www.pinecone.io/) vector store over the data-plane REST API.
/// Build it with [`StoreBuilder`](super::StoreBuilder).
///
/// - The index must already exist, with the embedder's dimension.
/// - The document text is stored in metadata under `text_key` (default `text`, like
///   langchaingo and LangChain), so indexes are interchangeable with those libraries.
/// - Options: `filters` is a Pinecone metadata filter (e.g. `json!({"genre": {"$eq": "drama"}})`);
///   `name_space` overrides the store's namespace for that call; `score_threshold` keeps
///   matches with `score >= threshold`; `embedder` overrides the store's.
/// - `Document::score` is Pinecone's score (similarity for cosine/dotproduct indexes).
/// - Pinecone metadata values must be strings, numbers, booleans or lists of strings.
///   Nulls are dropped and other values are stored as JSON strings.
pub struct Store {
    pub(super) client: Client,
    pub(super) host: String,
    pub(super) api_key: String,
    pub(super) api_version: String,
    pub(super) embedder: Arc<dyn Embedder>,
    pub(super) namespace: String,
    pub(super) text_key: String,
}

pub type PineconeOptions = VecStoreOptions<Value>;

#[derive(Deserialize)]
struct QueryResponse {
    #[serde(default)]
    matches: Vec<ScoredVector>,
}

#[derive(Deserialize)]
struct ScoredVector {
    #[serde(default)]
    id: String,
    #[serde(default)]
    score: f64,
    #[serde(default)]
    metadata: Option<Map<String, Value>>,
}

/// Pinecone accepts strings, numbers, booleans and lists of strings as metadata values.
fn pinecone_metadata_value(value: &Value) -> Option<Value> {
    match value {
        Value::Null => None,
        Value::Bool(_) | Value::Number(_) | Value::String(_) => Some(value.clone()),
        Value::Array(items) if items.iter().all(Value::is_string) => Some(value.clone()),
        Value::Array(_) | Value::Object(_) => Some(Value::String(value.to_string())),
    }
}

impl Store {
    async fn post(&self, path: &str, body: &Value) -> Result<Value, Box<dyn Error>> {
        let response = self
            .client
            .post(format!("{}{path}", self.host))
            .header("Api-Key", &self.api_key)
            .header("X-Pinecone-Api-Version", &self.api_version)
            .json(body)
            .send()
            .await?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            let message = serde_json::from_str::<Value>(&text)
                .ok()
                .and_then(|v| v["message"].as_str().map(str::to_string))
                .unwrap_or(text);
            return Err(format!("Pinecone error ({status}): {message}").into());
        }
        Ok(serde_json::from_str(&text).unwrap_or(Value::Null))
    }

    fn namespace<'a>(&'a self, opt: &'a PineconeOptions) -> &'a str {
        opt.name_space.as_deref().unwrap_or(&self.namespace)
    }

    fn metadata(&self, doc: &Document) -> Map<String, Value> {
        let mut metadata: Map<String, Value> = doc
            .metadata
            .iter()
            .filter_map(|(k, v)| pinecone_metadata_value(v).map(|v| (k.clone(), v)))
            .collect();
        metadata.insert(self.text_key.clone(), json!(doc.page_content));
        metadata
    }
}

#[async_trait]
impl VectorStore for Store {
    type Options = PineconeOptions;

    /// Embeds and upserts the documents in batches, returning their ids.
    ///
    /// A [`Document::id`] is used as the vector id, so writing it again overwrites
    /// that record (values and metadata).
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &PineconeOptions,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        if docs.is_empty() {
            return Ok(vec![]);
        }
        let ids = document_ids(docs, || uuid::Uuid::new_v4().to_string())?;
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let texts: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();
        let embeddings = embedder.embed_documents(&texts).await?;
        if embeddings.len() != docs.len() {
            return Err(format!(
                "embedder returned {} vectors for {} documents",
                embeddings.len(),
                docs.len()
            )
            .into());
        }
        let vectors: Vec<Value> = docs
            .iter()
            .zip(&embeddings)
            .zip(&ids)
            .map(|((doc, values), id)| {
                json!({ "id": id, "values": values, "metadata": self.metadata(doc) })
            })
            .collect();

        let namespace = self.namespace(opt);
        for batch in vectors.chunks(UPSERT_BATCH_SIZE) {
            self.post(
                "/vectors/upsert",
                &json!({ "vectors": batch, "namespace": namespace }),
            )
            .await?;
        }
        Ok(ids)
    }

    /// Returns up to `limit` matches for `query`, best first.
    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &PineconeOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let vector = embedder.embed_query(query).await?;

        let mut body = json!({
            "vector": vector,
            "topK": limit,
            "namespace": self.namespace(opt),
            "includeMetadata": true,
            "includeValues": false,
        });
        if let Some(filter) = &opt.filters {
            body["filter"] = filter.clone();
        }
        let response: QueryResponse = serde_json::from_value(self.post("/query", &body).await?)?;

        Ok(response
            .matches
            .into_iter()
            .filter(|m| {
                opt.score_threshold
                    .is_none_or(|threshold| m.score >= f64::from(threshold))
            })
            .map(|m| {
                let mut metadata = m.metadata.unwrap_or_default();
                let page_content = match metadata.remove(&self.text_key) {
                    Some(Value::String(text)) => text,
                    Some(other) => other.to_string(),
                    None => String::new(),
                };
                Document {
                    page_content,
                    metadata: metadata.into_iter().collect(),
                    score: m.score,
                    id: Some(m.id).filter(|id| !id.is_empty()),
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_values_pinecone_cannot_store_are_converted_or_dropped() {
        for (input, expected) in [
            (json!("s"), Some(json!("s"))),
            (json!(2), Some(json!(2))),
            (json!(false), Some(json!(false))),
            (json!(["a", "b"]), Some(json!(["a", "b"]))),
            (json!([]), Some(json!([]))),
            (json!([1, 2]), Some(json!("[1,2]"))),
            (json!({"a": 1}), Some(json!("{\"a\":1}"))),
            (json!(null), None),
        ] {
            assert_eq!(pinecone_metadata_value(&input), expected, "{input}");
        }
    }
}
