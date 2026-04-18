use crate::embedding::embedder_trait::Embedder;
use crate::embedding::EmbedderError;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// The internal request representation for embedding generation.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedRequest {
    /// The structural part wrapper accepted by the API.
    content: EmbedContent,
}

/// A wrapper denoting content payload.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContent {
    /// Ordered segments of text parts to process.
    parts: Vec<EmbedPart>,
}

/// Minimum part representation containing the raw text.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedPart {
    /// Text to embed.
    text: String,
}

/// The response payload received from the Gemini embeddings endpoint.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbedResponse {
    /// The generated multidimensional tensor structure.
    embedding: EmbedValues,
}

/// Values wrapper extracted from the Gemini representation.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbedValues {
    /// Floating point values representing the context in multi-dimensional space.
    values: Vec<f64>,
}

/// Provider client for generating embeddings directly using Google Gemini models.
///
/// Implements the standard [`Embedder`] trait used across Langchain-Rust vectorstores.
///
/// # Example
/// ```rust,no_run
/// use langchain_rust::llm::gemini::embeddings::GeminiEmbedder;
///
/// let embedder = GeminiEmbedder::builder()
///     .model("text-embedding-004")
///     .build()
///     .unwrap();
/// ```
#[derive(Clone)]
pub struct GeminiEmbedder {
    api_key: String,
    model: String,
    client: Client,
}

/// Extensible builder struct for correctly configuring a [`GeminiEmbedder`].
#[derive(Default)]
pub struct GeminiEmbedderBuilder {
    api_key: Option<String>,
    model: Option<String>,
    client: Option<Client>,
}

impl GeminiEmbedderBuilder {
    /// Initializes a new builder for standard Gemini embedder creation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the underlying generated Google API key required for authorization.
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Instructs the embedder algorithm to use a differently parameterized model.
    /// Defaults to `text-embedding-004`.
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets a preconfigured or reused HTTP client object reducing socket latency limits.
    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Concludes the configuration mapping returning a materialized `GeminiEmbedder`.
    ///
    /// # Errors
    /// Returns an error message if the `GEMINI_API_KEY` was neither passed directly nor found implicitly within the system environment setup.
    pub fn build(self) -> Result<GeminiEmbedder, String> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("GEMINI_API_KEY").ok())
            .ok_or_else(|| {
                "No API key found. Provide via builder or GEMINI_API_KEY env var".to_string()
            })?;

        let model = self
            .model
            .unwrap_or_else(|| "text-embedding-004".to_string());

        Ok(GeminiEmbedder {
            api_key,
            model,
            client: self.client.unwrap_or_default(),
        })
    }
}

impl GeminiEmbedder {
    /// Returns a new instance configured by the user builder template automatically matching standard.
    pub fn builder() -> GeminiEmbedderBuilder {
        GeminiEmbedderBuilder::new()
    }

    /// Simple instantiation overriding explicitly the keys needed to communicate without falling back.
    pub fn new<S: Into<String>>(api_key: S, model: S) -> Result<Self, String> {
        Self::builder().api_key(api_key).model(model).build()
    }
}

impl Default for GeminiEmbedder {
    fn default() -> Self {
        Self::builder().build().unwrap_or_else(|_| GeminiEmbedder {
            api_key: String::new(),
            model: "text-embedding-004".to_string(),
            client: Client::default(),
        })
    }
}

#[async_trait]
impl Embedder for GeminiEmbedder {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        let mut all_embeddings = Vec::with_capacity(documents.len());
        // Standard iterative fallback for endpoints primarily designed handling sequences independently
        for text in documents {
            let embedding = self.embed_query(text).await?;
            all_embeddings.push(embedding);
        }
        Ok(all_embeddings)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        let request_body = EmbedRequest {
            content: EmbedContent {
                parts: vec![EmbedPart {
                    text: text.to_string(),
                }],
            },
        };

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent?key={}",
            self.model, self.api_key
        );

        let res = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                EmbedderError::GeminiError(format!(
                    "Failed to securely route request over HTTP to Gemini embedder: {}",
                    e
                ))
            })?;

        if !res.status().is_success() {
            let status = res.status();
            let text = res.text().await.unwrap_or_else(|_| {
                "Hidden fallback HTTP stream failure format unavailable".to_string()
            });
            return Err(EmbedderError::GeminiError(format!(
                "Gemini Embedding generation rejected: {} | {}",
                status, text
            )));
        }

        let resp = res.json::<EmbedResponse>().await.map_err(|e| {
            EmbedderError::GeminiError(format!(
                "Structured JSON validation breakdown from generic HTTP layout: {}",
                e
            ))
        })?;

        Ok(resp.embedding.values)
    }
}
