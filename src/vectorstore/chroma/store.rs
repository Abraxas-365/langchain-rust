use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use reqwest::{Client, RequestBuilder};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{document_ids, VecStoreOptions, VectorStore},
};

/// Distance function of a new collection. Existing collections keep theirs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChromaDistance {
    /// Cosine distance, `1 - cosine similarity`. Score = `1 - distance`.
    #[default]
    Cosine,
    /// Squared euclidean distance. Score = `1 / (1 + distance)`.
    L2,
    /// Inner product, `1 - dot`. Score = `1 - distance`.
    InnerProduct,
}

impl ChromaDistance {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            ChromaDistance::Cosine => "cosine",
            ChromaDistance::L2 => "l2",
            ChromaDistance::InnerProduct => "ip",
        }
    }

    /// Converts a Chroma distance into a "higher is better" relevance score.
    fn score(self, distance: f64) -> f64 {
        match self {
            ChromaDistance::Cosine | ChromaDistance::InnerProduct => 1.0 - distance,
            ChromaDistance::L2 => 1.0 / (1.0 + distance),
        }
    }
}

/// [Chroma](https://www.trychroma.com/) vector store, over Chroma's v2 HTTP API (Chroma
/// 1.x server or Chroma Cloud). Build it with [`StoreBuilder`](super::StoreBuilder).
///
/// - Embeddings are computed here with the configured [`Embedder`]; Chroma only stores them.
/// - Options: `filters` is a Chroma `where` filter (e.g. `json!({"source": "a.txt"})` or
///   `{"$and": [...]}`); `name_space` scopes both writes and searches to a metadata value
///   (see [`StoreBuilder::namespace_key`](super::StoreBuilder::namespace_key));
///   `score_threshold` drops results below that score; `embedder` overrides the store's.
/// - Chroma metadata values must be scalars or arrays of one scalar type. Other values
///   (objects, mixed arrays) are stored as JSON strings.
pub struct Store {
    pub(super) client: Client,
    pub(super) collection_url: String,
    pub(super) api_key: Option<String>,
    pub(super) embedder: Arc<dyn Embedder>,
    pub(super) distance: ChromaDistance,
    pub(super) namespace_key: String,
}

pub type ChromaOptions = VecStoreOptions<Value>;

/// Chroma returns one row per query embedding; a missing cell is `null`.
type Rows<T> = Option<Vec<Vec<Option<T>>>>;

#[derive(Deserialize)]
struct QueryResponse {
    ids: Vec<Vec<String>>,
    #[serde(default)]
    documents: Rows<String>,
    #[serde(default)]
    metadatas: Rows<Map<String, Value>>,
    #[serde(default)]
    distances: Rows<f64>,
}

/// Sends a request and turns Chroma's `{"error", "message"}` bodies into errors.
pub(super) async fn send(
    request: RequestBuilder,
    api_key: &Option<String>,
) -> Result<Value, Box<dyn Error>> {
    let request = match api_key {
        Some(key) => request.header("x-chroma-token", key),
        None => request,
    };
    let response = request.send().await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        let message = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|v| v["message"].as_str().map(str::to_string))
            .unwrap_or(body);
        return Err(format!("Chroma error ({status}): {message}").into());
    }
    Ok(serde_json::from_str(&body).unwrap_or(Value::Null))
}

/// Chroma accepts scalars and arrays of a single scalar type as metadata values; anything
/// else is stored as its JSON text so writes don't fail on arbitrary metadata.
fn chroma_metadata_value(value: &Value) -> Value {
    fn scalar_kind(v: &Value) -> Option<u8> {
        match v {
            Value::Bool(_) => Some(0),
            Value::Number(_) => Some(1),
            Value::String(_) => Some(2),
            _ => None,
        }
    }
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => value.clone(),
        Value::Array(items) => {
            let first = items.first().and_then(scalar_kind);
            if first.is_some() && items.iter().all(|i| scalar_kind(i) == first) {
                value.clone()
            } else {
                Value::String(value.to_string())
            }
        }
        Value::Object(_) => Value::String(value.to_string()),
    }
}

impl Store {
    fn namespace_filter(&self, opt: &ChromaOptions) -> Option<Value> {
        let namespace = opt
            .name_space
            .as_ref()
            .map(|ns| json!({ &self.namespace_key: ns }));
        match (namespace, opt.filters.clone()) {
            (Some(ns), Some(filter)) => Some(json!({ "$and": [ns, filter] })),
            (ns, filter) => ns.or(filter),
        }
    }

    fn metadatas(&self, docs: &[Document], opt: &ChromaOptions) -> Vec<Option<Map<String, Value>>> {
        docs.iter()
            .map(|doc| {
                let mut metadata: Map<String, Value> = doc
                    .metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), chroma_metadata_value(v)))
                    .collect();
                if let Some(ns) = &opt.name_space {
                    metadata.insert(self.namespace_key.clone(), json!(ns));
                }
                // Chroma rejects empty metadata objects on some versions; send null instead.
                (!metadata.is_empty()).then_some(metadata)
            })
            .collect()
    }
}

#[async_trait]
impl VectorStore for Store {
    type Options = ChromaOptions;

    /// Embeds and upserts the documents, returning their ids.
    ///
    /// Chroma's upsert merges metadata into an existing record, so documents with a
    /// [`Document::id`] are deleted first and then written: the new version fully
    /// replaces the old one. The two calls aren't atomic; if the upsert fails, those
    /// ids are gone until the write is retried.
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &ChromaOptions,
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

        let body = json!({
            "ids": ids,
            "embeddings": embeddings,
            "documents": texts,
            "metadatas": self.metadatas(docs, opt),
        });
        let caller_ids: Vec<&str> = docs.iter().filter_map(|d| d.id.as_deref()).collect();
        if !caller_ids.is_empty() {
            send(
                self.client
                    .post(format!("{}/delete", self.collection_url))
                    .json(&json!({ "ids": caller_ids })),
                &self.api_key,
            )
            .await?;
        }
        send(
            self.client
                .post(format!("{}/upsert", self.collection_url))
                .json(&body),
            &self.api_key,
        )
        .await?;
        Ok(ids)
    }

    /// Returns up to `limit` documents closest to `query`, best first, with `score` set
    /// (higher is more similar).
    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &ChromaOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let embedding = embedder.embed_query(query).await?;

        let mut body = json!({
            "query_embeddings": [embedding],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
        });
        if let Some(filter) = self.namespace_filter(opt) {
            body["where"] = filter;
        }
        let response = send(
            self.client
                .post(format!("{}/query", self.collection_url))
                .json(&body),
            &self.api_key,
        )
        .await?;
        let response: QueryResponse = serde_json::from_value(response)?;

        // One query embedding, so one row of results.
        let ids = response.ids.into_iter().next().unwrap_or_default();
        let mut documents = response
            .documents
            .and_then(|d| d.into_iter().next())
            .unwrap_or_default()
            .into_iter();
        let mut metadatas = response
            .metadatas
            .and_then(|m| m.into_iter().next())
            .unwrap_or_default()
            .into_iter();
        let mut distances = response
            .distances
            .and_then(|d| d.into_iter().next())
            .unwrap_or_default()
            .into_iter();

        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            let page_content = documents.next().flatten().unwrap_or_default();
            let metadata = metadatas
                .next()
                .flatten()
                .unwrap_or_default()
                .into_iter()
                .collect();
            let score = distances
                .next()
                .flatten()
                .map_or(0.0, |d| self.distance.score(d));
            if opt
                .score_threshold
                .is_some_and(|threshold| score < f64::from(threshold))
            {
                continue;
            }
            results.push(Document {
                page_content,
                metadata,
                score,
                id: Some(id),
            });
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_values_chroma_cannot_store_become_json_strings() {
        for (input, expected) in [
            (json!("s"), json!("s")),
            (json!(1.5), json!(1.5)),
            (json!(true), json!(true)),
            (json!(null), json!(null)),
            (json!([1, 2]), json!([1, 2])),
            (json!(["a", "b"]), json!(["a", "b"])),
            (json!([1, "a"]), json!("[1,\"a\"]")),
            (json!([]), json!("[]")),
            (json!([[1]]), json!("[[1]]")),
            (json!({"a": 1}), json!("{\"a\":1}")),
        ] {
            assert_eq!(chroma_metadata_value(&input), expected, "{input}");
        }
    }

    #[test]
    fn scores_from_distances() {
        assert_eq!(ChromaDistance::Cosine.score(0.25), 0.75);
        assert_eq!(ChromaDistance::InnerProduct.score(0.25), 0.75);
        assert_eq!(ChromaDistance::L2.score(1.0), 0.5);
    }
}
