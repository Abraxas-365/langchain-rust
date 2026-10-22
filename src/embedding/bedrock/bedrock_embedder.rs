use std::sync::Arc;

use async_trait::async_trait;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_smithy_types::{error::display::DisplayErrorContext, Blob};
use futures::{stream, StreamExt, TryStreamExt};
use serde_json::{json, Value};
use tokio::sync::OnceCell;

use crate::embedding::{Embedder, EmbedderError};
use crate::observability::{traced_embed_documents, traced_embed_query};

/// `provider` field of the `embed.*` spans.
const PROVIDER: &str = "bedrock";

/// Cohere Embed accepts at most 96 texts per request.
pub const COHERE_MAX_BATCH: usize = 96;

/// Embedding models on Bedrock. `Custom` takes any model id, inference profile or ARN;
/// ids containing `cohere.` use the Cohere request format, everything else Titan's.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum BedrockEmbeddingModel {
    TitanEmbedTextV1,
    #[default]
    TitanEmbedTextV2,
    CohereEmbedEnglishV3,
    CohereEmbedMultilingualV3,
    Custom(String),
}

impl BedrockEmbeddingModel {
    pub fn model_id(&self) -> &str {
        match self {
            Self::TitanEmbedTextV1 => "amazon.titan-embed-text-v1",
            Self::TitanEmbedTextV2 => "amazon.titan-embed-text-v2:0",
            Self::CohereEmbedEnglishV3 => "cohere.embed-english-v3",
            Self::CohereEmbedMultilingualV3 => "cohere.embed-multilingual-v3",
            Self::Custom(id) => id,
        }
    }

    fn is_cohere(&self) -> bool {
        self.model_id().contains("cohere.")
    }
}

/// Embeddings from Amazon Bedrock.
///
/// Titan embeds one text per request, so documents are sent concurrently (see
/// [`with_concurrency`](Self::with_concurrency)). Cohere takes batches of up to
/// [`COHERE_MAX_BATCH`] texts and is told whether it's embedding documents
/// (`search_document`) or a query (`search_query`).
///
/// The SDK client is built lazily on first use and shared across clones.
#[derive(Clone)]
pub struct BedrockEmbedder {
    client: Arc<OnceCell<BedrockClient>>,
    region: Option<String>,
    model: BedrockEmbeddingModel,
    dimensions: Option<u32>,
    normalize: Option<bool>,
    truncate: Option<String>,
    concurrency: usize,
}

impl Default for BedrockEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl BedrockEmbedder {
    pub fn new() -> Self {
        Self {
            client: Arc::new(OnceCell::new()),
            region: None,
            model: BedrockEmbeddingModel::default(),
            dimensions: None,
            normalize: None,
            truncate: None,
            concurrency: 5,
        }
    }

    /// Use an already-constructed SDK client instead of the default config chain.
    pub fn with_client(mut self, client: BedrockClient) -> Self {
        self.client = Arc::new(OnceCell::new_with(Some(client)));
        self
    }

    /// Override the AWS region. Resets any lazily-built client.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self.client = Arc::new(OnceCell::new());
        self
    }

    pub fn with_model(mut self, model: BedrockEmbeddingModel) -> Self {
        self.model = model;
        self
    }

    /// Output size for Titan Text Embeddings V2 (256, 512 or 1024).
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Whether Titan Text Embeddings V2 returns unit vectors (its default is `true`).
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = Some(normalize);
        self
    }

    /// Cohere's `truncate` setting (`NONE`, `START` or `END`) for texts over its limit.
    pub fn with_truncate(mut self, truncate: impl Into<String>) -> Self {
        self.truncate = Some(truncate.into());
        self
    }

    /// Maximum number of Titan requests in flight while embedding documents (default 5).
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    async fn client(&self) -> &BedrockClient {
        self.client
            .get_or_init(|| async {
                let mut loader = aws_config::defaults(BehaviorVersion::latest());
                if let Some(region) = &self.region {
                    loader = loader.region(Region::new(region.clone()));
                }
                BedrockClient::new(&loader.load().await)
            })
            .await
    }

    async fn invoke(&self, body: Value) -> Result<Value, EmbedderError> {
        let output = self
            .client()
            .await
            .invoke_model()
            .model_id(self.model.model_id())
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body.to_string()))
            .send()
            .await
            .map_err(|e| EmbedderError::BedrockError(DisplayErrorContext(e).to_string()))?;
        serde_json::from_slice(output.body().as_ref())
            .map_err(|e| EmbedderError::BedrockError(format!("invalid InvokeModel response: {e}")))
    }

    async fn embed_titan(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        let mut body = json!({ "inputText": text });
        if let Some(dimensions) = self.dimensions {
            body["dimensions"] = json!(dimensions);
        }
        if let Some(normalize) = self.normalize {
            body["normalize"] = json!(normalize);
        }
        let response = self.invoke(body).await?;
        vector(&response["embedding"]).ok_or_else(|| unexpected("embedding", &response))
    }

    async fn embed_cohere(
        &self,
        texts: &[String],
        input_type: &str,
    ) -> Result<Vec<Vec<f64>>, EmbedderError> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(COHERE_MAX_BATCH) {
            let mut body = json!({ "texts": batch, "input_type": input_type });
            if let Some(truncate) = &self.truncate {
                body["truncate"] = json!(truncate);
            }
            let response = self.invoke(body).await?;
            // `{"embeddings": [[..]]}`, or `{"embeddings": {"float": [[..]]}}` when the
            // model answers with typed embeddings (Embed v4).
            let list = match &response["embeddings"] {
                Value::Object(typed) => &typed["float"],
                other => other,
            };
            let vectors = list
                .as_array()
                .and_then(|items| items.iter().map(vector).collect::<Option<Vec<_>>>())
                .ok_or_else(|| unexpected("embeddings", &response))?;
            if vectors.len() != batch.len() {
                return Err(EmbedderError::BedrockError(format!(
                    "Cohere returned {} embeddings for {} texts",
                    vectors.len(),
                    batch.len()
                )));
            }
            embeddings.extend(vectors);
        }
        Ok(embeddings)
    }
}

fn vector(value: &Value) -> Option<Vec<f64>> {
    value.as_array()?.iter().map(Value::as_f64).collect()
}

pub(super) fn unexpected(field: &str, response: &Value) -> EmbedderError {
    let mut body = response.to_string();
    if body.len() > 200 {
        let cut = (0..=200)
            .rev()
            .find(|&i| body.is_char_boundary(i))
            .unwrap_or(0);
        body.truncate(cut);
        body.push_str("...");
    }
    EmbedderError::BedrockError(format!("InvokeModel response has no `{field}`: {body}"))
}

#[async_trait]
impl Embedder for BedrockEmbedder {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        traced_embed_documents(PROVIDER, self.model.model_id(), documents.len(), async {
            if self.model.is_cohere() {
                return self.embed_cohere(documents, "search_document").await;
            }
            stream::iter(documents.iter().cloned())
                .map(|text| async move { self.embed_titan(&text).await })
                .buffered(self.concurrency)
                .try_collect()
                .await
        })
        .await
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        traced_embed_query(PROVIDER, self.model.model_id(), async {
            if self.model.is_cohere() {
                let mut vectors = self
                    .embed_cohere(&[text.to_string()], "search_query")
                    .await?;
                return Ok(vectors.remove(0));
            }
            self.embed_titan(text).await
        })
        .await
    }
}
