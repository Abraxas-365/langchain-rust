#![allow(dead_code)]

use crate::embedding::{embedder_trait::Embedder, EmbedderError};
pub use async_openai::config::{AzureConfig, Config, OpenAIConfig};
use async_openai::{
    types::embeddings::{CreateEmbeddingRequestArgs, EmbeddingInput},
    Client,
};
use async_trait::async_trait;

#[derive(Debug)]
pub struct OpenAiEmbedder<C: Config> {
    config: C,
    model: String,
}

impl<C: Config + Clone + Send + Sync + 'static> Into<Box<dyn Embedder>> for OpenAiEmbedder<C> {
    fn into(self) -> Box<dyn Embedder> {
        Box::new(self)
    }
}

impl<C: Config> OpenAiEmbedder<C> {
    pub fn new(config: C) -> Self {
        OpenAiEmbedder {
            config,
            model: String::from("text-embedding-ada-002"),
        }
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_config(mut self, config: C) -> Self {
        self.config = config;
        self
    }
}

impl Default for OpenAiEmbedder<OpenAIConfig> {
    fn default() -> Self {
        OpenAiEmbedder::new(OpenAIConfig::default())
    }
}

#[async_trait]
impl<C: Config + Clone + Send + Sync> Embedder for OpenAiEmbedder<C> {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        let client = Client::with_config(self.config.clone());

        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(EmbeddingInput::StringArray(documents.into()))
            .build()?;

        let response = client.embeddings().create(request).await?;

        let embeddings = response
            .data
            .into_iter()
            .map(|item| item.embedding)
            .map(|embedding| {
                embedding
                    .into_iter()
                    .map(|x| x as f64)
                    .collect::<Vec<f64>>()
            })
            .collect();

        Ok(embeddings)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        let client = Client::with_config(self.config.clone());

        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(text)
            .build()?;

        let response = client.embeddings().create(request).await?;

        let item = response.data.into_iter().next().ok_or_else(|| {
            async_openai::error::OpenAIError::InvalidArgument(
                "OpenAI returned no embeddings".into(),
            )
        })?;

        Ok(item
            .embedding
            .into_iter()
            .map(|x| x as f64)
            .collect::<Vec<f64>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn embedder_for(server: &mockito::ServerGuard) -> OpenAiEmbedder<OpenAIConfig> {
        OpenAiEmbedder::new(
            OpenAIConfig::new()
                .with_api_base(server.url())
                .with_api_key("test-key"),
        )
        .with_model("emb-model")
    }

    fn body(vectors: &[Vec<f32>]) -> String {
        let data: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| json!({"index": i, "object": "embedding", "embedding": v}))
            .collect();
        json!({
            "object": "list",
            "model": "emb-model",
            "data": data,
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
        })
        .to_string()
    }

    #[tokio::test]
    async fn embed_documents_sends_batch_and_converts_vectors() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/embeddings")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::PartialJson(json!({
                "model": "emb-model",
                "input": ["a", "b"]
            })))
            .with_header("content-type", "application/json")
            .with_body(body(&[vec![0.5, 1.0], vec![0.25, 0.0]]))
            .create_async()
            .await;

        let out = embedder_for(&server)
            .embed_documents(&["a".to_string(), "b".to_string()])
            .await
            .unwrap();
        mock.assert_async().await;
        assert_eq!(out, vec![vec![0.5, 1.0], vec![0.25, 0.0]]);
    }

    #[tokio::test]
    async fn embed_query_sends_single_string() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/embeddings")
            .match_body(mockito::Matcher::PartialJson(json!({"input": "hello"})))
            .with_header("content-type", "application/json")
            .with_body(body(&[vec![0.5]]))
            .create_async()
            .await;

        let out = embedder_for(&server).embed_query("hello").await.unwrap();
        mock.assert_async().await;
        assert_eq!(out, vec![0.5]);
    }

    #[tokio::test]
    async fn embed_query_with_empty_response_is_an_error() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/embeddings")
            .with_header("content-type", "application/json")
            .with_body(body(&[]))
            .create_async()
            .await;

        let err = embedder_for(&server)
            .embed_query("hello")
            .await
            .unwrap_err();
        assert!(matches!(err, EmbedderError::OpenAIError(_)), "{err:?}");
    }
}
