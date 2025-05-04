use std::sync::Arc;

use crate::embedding::{embedder_trait::Embedder, EmbedderError};
use async_trait::async_trait;
use ollama_rs::{
    generation::{
        embeddings::request::{EmbeddingsInput, GenerateEmbeddingsRequest},
    },
    Ollama as OllamaClient,
};
use ollama_rs::models::ModelOptions;

#[derive(Debug)]
pub struct OllamaEmbedder {
    pub(crate) client: Arc<OllamaClient>,
    pub(crate) model: String,
    pub(crate) options: Option<ModelOptions>,
}

/// [nomic-embed-text](https://ollama.com/library/nomic-embed-text) is a 137M parameters, 274MB model.
const DEFAULT_MODEL: &str = "nomic-embed-text";

impl OllamaEmbedder {
    pub fn new<S: Into<String>>(
        client: Arc<OllamaClient>,
        model: S,
        options: Option<ModelOptions>,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            options,
        }
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_options(mut self, options: ModelOptions) -> Self {
        self.options = Some(options);
        self
    }
}

impl Default for OllamaEmbedder {
    fn default() -> Self {
        let client = Arc::new(OllamaClient::default());
        Self::new(client, String::from(DEFAULT_MODEL), None)
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        log::debug!("Embedding documents: {:?}", documents);

        let response = self
            .client
            .generate_embeddings(GenerateEmbeddingsRequest::new(
                self.model.clone(),
                EmbeddingsInput::Multiple(documents.to_vec()),
            ))
            .await?;

        let embeddings = response
            .embeddings
            .into_iter()
            .map(|embedding| embedding.into_iter().map(f64::from).collect())
            .collect();

        Ok(embeddings)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        log::debug!("Embedding query: {:?}", text);

        let response = self
            .client
            .generate_embeddings(GenerateEmbeddingsRequest::new(
                self.model.clone(),
                EmbeddingsInput::Single(text.into()),
            ))
            .await?;

        let embeddings = response
            .embeddings
            .into_iter()
            .next()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect();

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_ollama_embed() {
        let ollama = OllamaEmbedder::default()
            .with_model("nomic-embed-text")
            .with_options(ModelOptions::default().temperature(0.5));

        let response = ollama.embed_query("Why is the sky blue?").await.unwrap();

        assert_eq!(response.len(), 768);
    }
}
