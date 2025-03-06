use std::sync::Arc;

use crate::embedding::{embedder_trait::Embedder, EmbedderError};
use async_trait::async_trait;
use mistralai_client::v1::{client::Client, constants::EmbedModel};

pub struct MistralAIEmbedder {
    client: Arc<Client>,
    model: EmbedModel,
}

impl MistralAIEmbedder {
    pub fn try_new() -> Result<Self, EmbedderError> {
        Ok(Self {
            client: Arc::new(
                Client::new(None, None, None, None).map_err(EmbedderError::MistralAIClientError)?,
            ),
            model: EmbedModel::MistralEmbed,
        })
    }
}

#[async_trait]
impl Embedder for MistralAIEmbedder {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        let response = self
            .client
            .embeddings_async(self.model.clone(), documents.into(), None)
            .await
            .map_err(EmbedderError::MistralAIApiError)?;

        Ok(response
            .data
            .into_iter()
            .map(|item| item.embedding.into_iter().map(|x| x as f64).collect())
            .collect::<Vec<Vec<f64>>>())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        let response = self
            .client
            .embeddings_async(self.model.clone(), vec![text.to_string()], None)
            .await
            .map_err(EmbedderError::MistralAIApiError)?;

        Ok(response.data[0]
            .embedding
            .iter()
            .map(|x| *x as f64)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_mistralai_embed_query() {
        let mistralai = MistralAIEmbedder::try_new().unwrap();
        let embeddings = mistralai.embed_query("Why is the sky blue?").await.unwrap();
        assert_eq!(embeddings.len(), 1024);
    }

    #[tokio::test]
    #[ignore]
    async fn test_mistralai_embed_documents() {
        let mistralai = MistralAIEmbedder::try_new().unwrap();
        let embeddings = mistralai
            .embed_documents(&["hello world".to_string(), "foo bar".to_string()])
            .await
            .unwrap();
        assert_eq!(embeddings.len(), 2);
    }
}
