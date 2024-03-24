#![allow(dead_code)]
use std::error::Error;

use crate::embedding::embedder_trait::Embedder;
pub use async_openai::config::{AzureConfig, Config, OpenAIConfig};
use async_openai::{
    types::{CreateEmbeddingRequestArgs, EmbeddingInput},
    Client,
};
use async_trait::async_trait;

#[derive(Debug)]
pub struct OpenAiEmbedder<C: Config> {
    config: C,
    model: String,
}

impl<C: Config + Send + Sync + 'static> Into<Box<dyn Embedder>> for OpenAiEmbedder<C> {
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
impl<C: Config + Send + Sync> Embedder for OpenAiEmbedder<C> {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
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

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, Box<dyn Error>> {
        let client = Client::with_config(self.config.clone());

        let request = CreateEmbeddingRequestArgs::default()
            .model(&self.model)
            .input(text)
            .build()?;

        let mut response = client.embeddings().create(request).await?;

        let item = response.data.swap_remove(0);

        Ok(item
            .embedding
            .into_iter()
            .map(|x| x as f64)
            .collect::<Vec<f64>>())
    }
}
