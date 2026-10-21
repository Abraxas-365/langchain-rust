use std::sync::{Mutex, OnceLock};

use async_trait::async_trait;

use crate::embedding::{Embedder, EmbedderError};
use fastembed::{TextEmbedding, TextInitOptions};
use ort::environment::Environment;

/// The process-wide ONNX Runtime environment.
///
/// Since fastembed 7, ONNX Runtime allows a single environment per process, so every model
/// after the first must be created with it. Pass it to your own options with
/// `TextInitOptions::new(model).with_environment(onnx_environment()?)`.
pub fn onnx_environment() -> Result<Environment, EmbedderError> {
    static ENVIRONMENT: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENVIRONMENT
        .get_or_init(|| ort::init().build().map_err(|e| e.to_string()))
        .clone()
        .map_err(EmbedderError::FastEmbedError)
}

pub struct FastEmbed {
    // fastembed's `embed` takes `&mut self`; the lock is never held across an await.
    model: Mutex<TextEmbedding>,
    batch_size: Option<usize>,
}

impl FastEmbed {
    pub fn try_new() -> Result<Self, EmbedderError> {
        Ok(Self {
            model: Mutex::new(
                TextEmbedding::try_new(
                    TextInitOptions::default().with_environment(onnx_environment()?),
                )
                .map_err(|e| EmbedderError::FastEmbedError(e.to_string()))?,
            ),
            batch_size: None,
        })
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    fn embed<S: AsRef<str> + Send + Sync>(
        &self,
        texts: Vec<S>,
    ) -> Result<Vec<Vec<f32>>, EmbedderError> {
        self.model
            .lock()
            .map_err(|e| EmbedderError::FastEmbedError(e.to_string()))?
            .embed(texts, self.batch_size)
            .map_err(|e| EmbedderError::FastEmbedError(e.to_string()))
    }
}

impl From<TextEmbedding> for FastEmbed {
    fn from(model: TextEmbedding) -> Self {
        Self {
            model: Mutex::new(model),
            batch_size: None,
        }
    }
}

#[async_trait]
impl Embedder for FastEmbed {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        let embeddings = self.embed(documents.to_vec())?;

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
        let embedding = self.embed(vec![text])?;

        embedding
            .into_iter()
            .next()
            .map(|v| v.into_iter().map(f64::from).collect())
            .ok_or_else(|| EmbedderError::FastEmbedError("no embedding returned".into()))
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

    /// fastembed 7 (ort rc.15) allows one ONNX environment per process; a second model
    /// created after the first is dropped must still work.
    #[tokio::test]
    async fn fastembed_can_be_created_twice() {
        let first = FastEmbed::try_new().unwrap();
        drop(first);
        let second = FastEmbed::try_new().unwrap();
        let both_alive = FastEmbed::try_new().unwrap();
        assert_eq!(second.embed_query("hi").await.unwrap().len(), 384);
        assert_eq!(both_alive.embed_query("hi").await.unwrap().len(), 384);
    }
}
