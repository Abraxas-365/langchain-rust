use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::embedding::{Embedder, EmbedderError};
use crate::observability::{traced_embed_documents, traced_embed_query};

/// `provider` field of the `embed.*` spans.
const PROVIDER: &str = "voyage";

pub const VOYAGE_API_BASE: &str = "https://api.voyageai.com/v1";
pub const DEFAULT_VOYAGE_MODEL: &str = "voyage-3.5";
/// The API accepts up to 1,000 texts per request; smaller batches stay well under the
/// per-request token limits (120K-1M depending on the model).
pub const DEFAULT_VOYAGE_BATCH_SIZE: usize = 128;
const MAX_VOYAGE_BATCH_SIZE: usize = 1000;

/// Embeddings from the Voyage AI API.
///
/// Documents are sent with `input_type: "document"` and queries with
/// `input_type: "query"`, which is what Voyage recommends for retrieval.
#[derive(Debug, Clone)]
pub struct VoyageEmbedder {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    output_dimension: Option<u32>,
    truncation: Option<bool>,
    batch_size: usize,
}

impl Default for VoyageEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl VoyageEmbedder {
    /// Uses `VOYAGE_API_KEY` from the environment (empty if unset; see
    /// [`with_api_key`](Self::with_api_key)).
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: VOYAGE_API_BASE.to_string(),
            api_key: std::env::var("VOYAGE_API_KEY").unwrap_or_default(),
            model: DEFAULT_VOYAGE_MODEL.to_string(),
            output_dimension: None,
            truncation: None,
            batch_size: DEFAULT_VOYAGE_BATCH_SIZE,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// API base URL, without `/embeddings` (default `https://api.voyageai.com/v1`).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    /// `output_dimension` (2048, 1024, 512 or 256 on models that support it).
    pub fn with_output_dimension(mut self, dimension: u32) -> Self {
        self.output_dimension = Some(dimension);
        self
    }

    /// When `false`, over-length texts are an error instead of being truncated.
    pub fn with_truncation(mut self, truncation: bool) -> Self {
        self.truncation = Some(truncation);
        self
    }

    /// Texts per request, between 1 and 1,000 (default 128).
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.clamp(1, MAX_VOYAGE_BATCH_SIZE);
        self
    }

    async fn embed(
        &self,
        texts: &[String],
        input_type: &str,
    ) -> Result<Vec<Vec<f64>>, EmbedderError> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch_size) {
            let mut body = json!({ "input": batch, "model": self.model, "input_type": input_type });
            if let Some(dimension) = self.output_dimension {
                body["output_dimension"] = json!(dimension);
            }
            if let Some(truncation) = self.truncation {
                body["truncation"] = json!(truncation);
            }
            let response = self
                .client
                .post(format!("{}/embeddings", self.base_url))
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await?;
            let status = response.status();
            let text = response.text().await?;
            if !status.is_success() {
                return Err(EmbedderError::HttpError {
                    status_code: status,
                    error_message: error_message(&text),
                });
            }
            let mut parsed: EmbeddingsResponse =
                serde_json::from_str(&text).map_err(|e| EmbedderError::HttpError {
                    status_code: status,
                    error_message: format!("invalid Voyage response: {e}"),
                })?;
            if parsed.data.len() != batch.len() {
                return Err(EmbedderError::HttpError {
                    status_code: status,
                    error_message: format!(
                        "Voyage returned {} embeddings for {} texts",
                        parsed.data.len(),
                        batch.len()
                    ),
                });
            }
            parsed.data.sort_by_key(|item| item.index);
            embeddings.extend(parsed.data.into_iter().map(|item| item.embedding));
        }
        Ok(embeddings)
    }
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f64>,
    index: usize,
}

/// Voyage errors look like `{"detail": "..."}`; fall back to the raw body.
fn error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["detail"].as_str().map(str::to_string))
        .unwrap_or_else(|| body.to_string())
}

#[async_trait]
impl Embedder for VoyageEmbedder {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        traced_embed_documents(PROVIDER, &self.model, documents.len(), async {
            self.embed(documents, "document").await
        })
        .await
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        traced_embed_query(PROVIDER, &self.model, async {
            let mut vectors = self.embed(&[text.to_string()], "query").await?;
            Ok(vectors.remove(0))
        })
        .await
    }
}
