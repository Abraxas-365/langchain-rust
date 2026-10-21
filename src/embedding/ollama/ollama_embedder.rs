use std::sync::Arc;

use crate::embedding::{embedder_trait::Embedder, EmbedderError};
use async_trait::async_trait;
use ollama_rs::{
    error::OllamaError,
    generation::embeddings::request::{EmbeddingsInput, GenerateEmbeddingsRequest},
    models::ModelOptions as GenerationOptions,
    Ollama as OllamaClient,
};

#[derive(Debug)]
pub struct OllamaEmbedder {
    pub(crate) client: Arc<OllamaClient>,
    pub(crate) model: String,
    pub(crate) options: Option<GenerationOptions>,
}

/// [nomic-embed-text](https://ollama.com/library/nomic-embed-text) is a 137M parameters, 274MB model.
const DEFAULT_MODEL: &str = "nomic-embed-text";

impl OllamaEmbedder {
    pub fn new<S: Into<String>>(
        client: Arc<OllamaClient>,
        model: S,
        options: Option<GenerationOptions>,
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

    pub fn with_options(mut self, options: GenerationOptions) -> Self {
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
            .generate_embeddings(self.request(EmbeddingsInput::Multiple(documents.to_vec())))
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
            .generate_embeddings(self.request(EmbeddingsInput::Single(text.into())))
            .await?;

        let embeddings = response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| OllamaError::Other("Ollama returned no embeddings".to_string()))?
            .into_iter()
            .map(f64::from)
            .collect();

        Ok(embeddings)
    }
}

impl OllamaEmbedder {
    fn request(&self, input: EmbeddingsInput) -> GenerateEmbeddingsRequest {
        let request = GenerateEmbeddingsRequest::new(self.model.clone(), input);
        match &self.options {
            Some(options) => request.options(options.clone()),
            None => request,
        }
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
            .with_options(GenerationOptions::default().temperature(0.5));

        let response = ollama.embed_query("Why is the sky blue?").await.unwrap();

        assert_eq!(response.len(), 768);
    }

    fn embedder_for(server: &mockito::ServerGuard) -> OllamaEmbedder {
        let client = Arc::new(OllamaClient::try_new(server.url()).unwrap());
        OllamaEmbedder::new(client, "emb", None)
    }

    #[tokio::test]
    async fn embed_documents_against_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/embed")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "model": "emb",
                "input": ["a", "b"],
                "options": {"temperature": 0.5}
            })))
            .with_header("content-type", "application/json")
            .with_body(r#"{"embeddings": [[0.5, 1.0], [0.25, 0.0]]}"#)
            .create_async()
            .await;

        let out = embedder_for(&server)
            .with_options(GenerationOptions::default().temperature(0.5))
            .embed_documents(&["a".into(), "b".into()])
            .await
            .unwrap();
        mock.assert_async().await;
        assert_eq!(out, vec![vec![0.5, 1.0], vec![0.25, 0.0]]);
    }

    #[tokio::test]
    async fn embed_query_against_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/embed")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"input": "hi"}),
            ))
            .with_header("content-type", "application/json")
            .with_body(r#"{"embeddings": [[0.5]]}"#)
            .create_async()
            .await;

        let out = embedder_for(&server).embed_query("hi").await.unwrap();
        mock.assert_async().await;
        assert_eq!(out, vec![0.5]);
    }

    #[tokio::test]
    async fn embed_query_empty_response_is_an_error() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/api/embed")
            .with_header("content-type", "application/json")
            .with_body(r#"{"embeddings": []}"#)
            .create_async()
            .await;

        assert!(embedder_for(&server).embed_query("hi").await.is_err());
    }
}
