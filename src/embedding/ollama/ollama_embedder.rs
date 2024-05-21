use std::sync::Arc;

use crate::embedding::{embedder_trait::Embedder, EmbedderError};
use async_trait::async_trait;
use ollama_rs::{generation::options::GenerationOptions, Ollama as OllamaClient};

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

        let mut embeddings = Vec::with_capacity(documents.len());

        for doc in documents {
            let res = self
                .client
                .generate_embeddings(self.model.clone(), doc.clone(), self.options.clone())
                .await?;

            embeddings.push(res.embeddings);
        }

        Ok(embeddings)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        log::debug!("Embedding query: {:?}", text);

        let res = self
            .client
            .generate_embeddings(self.model.clone(), text.to_string(), self.options.clone())
            .await?;

        Ok(res.embeddings)
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
}
