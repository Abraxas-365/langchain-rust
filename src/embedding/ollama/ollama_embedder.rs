#![allow(dead_code)]

use crate::embedding::{embedder_trait::Embedder, EmbedderError};
use async_trait::async_trait;
use reqwest::{Client, Url};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f64>,
}

#[derive(Debug)]
pub struct OllamaEmbedder {
    pub(crate) model: String,
    pub(crate) base_url: String,
}

impl OllamaEmbedder {
    pub fn new<S: Into<String>>(model: S, base_url: S) -> Self {
        OllamaEmbedder {
            model: model.into(),
            base_url: base_url.into(),
        }
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_api_base<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into();
        self
    }
}

impl Default for OllamaEmbedder {
    fn default() -> Self {
        let model = String::from("nomic-embed-text");
        let base_url = String::from("http://localhost:11434");
        OllamaEmbedder::new(model, base_url)
    }
}

#[async_trait]
impl Embedder for OllamaEmbedder {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        log::debug!("Embedding documents: {:?}", documents);
        let client = Client::new();
        let url = Url::parse(&format!("{}{}", self.base_url, "/api/embeddings"))?;

        let mut embeddings = Vec::with_capacity(documents.len());

        for doc in documents {
            let res = client
                .post(url.clone())
                .json(&json!({
                    "prompt": doc,
                    "model": &self.model,
                }))
                .send()
                .await?;
            if res.status() != 200 {
                log::error!("Error from OLLAMA: {}", &res.status());
                return Err(EmbedderError::HttpError {
                    status_code: res.status(),
                    error_message: format!("Received non-200 response: {}", res.status()),
                });
            }
            let data: EmbeddingResponse = res.json().await?;
            embeddings.push(data.embedding);
        }

        Ok(embeddings)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        log::debug!("Embedding query: {:?}", text);
        let client = Client::new();
        let url = Url::parse(&format!("{}{}", self.base_url, "/api/embeddings"))?;

        let res = client
            .post(url)
            .json(&json!({
                "prompt": text,
                "model": &self.model,
            }))
            .send()
            .await?;

        if res.status() != 200 {
            log::error!("Error from OLLAMA: {}", &res.status());
            return Err(EmbedderError::HttpError {
                status_code: res.status(),
                error_message: format!("Received non-200 response: {}", res.status()),
            });
        }
        let data: EmbeddingResponse = res.json().await?;
        Ok(data.embedding)
    }
}
