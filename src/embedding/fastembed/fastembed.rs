use async_trait::async_trait;

use crate::embedding::{Embedder, EmbedderError};
use fastembed::TextEmbedding;

pub struct FastEmbed {
    model: TextEmbedding,
    batch_size: Option<usize>,
}

impl FastEmbed {
    pub fn try_new() -> Result<Self, EmbedderError> {
        Ok(Self {
            model: TextEmbedding::try_new(Default::default())
                .map_err(|e| EmbedderError::FastEmbedError(e.to_string()))?,
            batch_size: None,
        })
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }
}

impl From<TextEmbedding> for FastEmbed {
    fn from(model: TextEmbedding) -> Self {
        Self {
            model,
            batch_size: None,
        }
    }
}

#[async_trait]
impl Embedder for FastEmbed {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        let embeddings = self
            .model
            .embed(documents.to_vec(), self.batch_size)
            .map_err(|e| EmbedderError::FastEmbedError(e.to_string()))?;

        Ok(embeddings
            .into_iter()
            .map(|inner_vec| {
                inner_vec
                    .into_iter()
                    .map(|x| x as f64)
                    .collect::<Vec<f64>>()
            })
            .collect::<Vec<Vec<f64>>>())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        let embedding = self
            .model
            .embed(vec![text], self.batch_size)
            .map_err(|e| EmbedderError::FastEmbedError(e.to_string()))?;

        Ok(embedding[0].iter().map(|x| *x as f64).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_fastembed() {
        let fastembed = FastEmbed::try_new().unwrap();
        let embeddings = fastembed
            .embed_documents(&["hello world".to_string(), "foo bar".to_string()])
            .await
            .unwrap();
        assert_eq!(embeddings.len(), 2);
    }
}
