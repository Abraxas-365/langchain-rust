#![allow(dead_code)]
use std::{env, error::Error};

use crate::embedding::embedder_trait::Embedder;
use async_trait::async_trait;
use reqwest::{Client, Url};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingData>,
    model: String,
    usage: UsageData,
}
impl EmbeddingResponse {
    fn extract_embedding(&self) -> Vec<f64> {
        self.data[0].embedding.clone()
    }
    fn extract_all_embeddings(&self) -> Vec<Vec<f64>> {
        self.data.iter().map(|d| d.embedding.clone()).collect()
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    object: String,
    embedding: Vec<f64>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct UsageData {
    prompt_tokens: usize,
    total_tokens: usize,
}
#[derive(Debug)]
pub struct OpenAiEmbedder {
    pub(crate) model: String,
    pub(crate) openai_key: String,
    pub(crate) base_url: String,
}
impl OpenAiEmbedder {
    pub fn new<S: Into<String>>(openai_key: S, model: S, base_url: S) -> Self {
        OpenAiEmbedder {
            model: model.into(),
            openai_key: openai_key.into(),
            base_url: base_url.into(),
        }
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_api_key<S: Into<String>>(mut self, openai_key: S) -> Self {
        self.openai_key = openai_key.into();
        self
    }

    pub fn with_api_base<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into();
        self
    }
}

impl Default for OpenAiEmbedder {
    fn default() -> Self {
        let model = String::from("text-embedding-ada-002");
        let openai_key = env::var("OPENAI_API_KEY").unwrap_or(String::new());
        let base_url = String::from("https://api.openai.com/v1/embeddings");
        OpenAiEmbedder::new(openai_key, model, base_url)
    }
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
        log::debug!("Embedding documents: {:?}", documents);
        let client = Client::new();
        let url = Url::parse(&self.base_url)?;
        let res = client
            .post(url)
            .bearer_auth(&self.openai_key)
            .json(&json!({
                "input": documents,
                "model": &self.model,
            }))
            .send()
            .await?;

        if res.status() != 200 {
            let error_message: String = res
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error message".into());
            log::error!("Error from OpenAI: {}", &error_message);
            return Err(error_message.into());
        }

        let data: EmbeddingResponse = res.json().await?;
        Ok(data.extract_all_embeddings())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, Box<dyn Error>> {
        log::debug!("Embedding query: {:?}", text);
        let client = Client::new();
        let url = Url::parse("https://api.openai.com/v1/embeddings")?;

        let res = client
            .post(url)
            .bearer_auth(&self.openai_key)
            .json(&json!({
                "input": text,
                "model": &self.model,
            }))
            .send()
            .await?;

        if res.status() != 200 {
            let error_message: String = res
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read error message".into());
            log::error!("Error from OpenAI: {}", &error_message);
            return Err(error_message.into());
        }
        let data: EmbeddingResponse = res.json().await?;
        Ok(data.extract_embedding())
    }
}
