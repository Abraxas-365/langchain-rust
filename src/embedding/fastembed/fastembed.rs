use std::sync::{Arc, Mutex, OnceLock};

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
    // fastembed's `embed` takes `&mut self`. Inference runs on the blocking pool, which
    // needs an owned handle, hence the Arc.
    model: Arc<Mutex<TextEmbedding>>,
    batch_size: Option<usize>,
}

impl FastEmbed {
    pub fn try_new() -> Result<Self, EmbedderError> {
        Ok(Self {
            model: Arc::new(Mutex::new(
                TextEmbedding::try_new(
                    TextInitOptions::default().with_environment(onnx_environment()?),
                )
                .map_err(|e| EmbedderError::FastEmbedError(e.to_string()))?,
            )),
            batch_size: None,
        })
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Runs the model on Tokio's blocking pool: ONNX inference is CPU-bound and would
    /// otherwise stall every other task on this worker thread.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedderError> {
        let model = self.model.clone();
        let batch_size = self.batch_size;
        crate::blocking::run_blocking(move || {
            model
                .lock()
                .map_err(|e| EmbedderError::FastEmbedError(e.to_string()))?
                .embed(texts, batch_size)
                .map_err(|e| EmbedderError::FastEmbedError(e.to_string()))
        })
        .await
    }
}

impl From<TextEmbedding> for FastEmbed {
    fn from(model: TextEmbedding) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            batch_size: None,
        }
    }
}

#[async_trait]
impl Embedder for FastEmbed {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        let embeddings = self.embed(documents.to_vec()).await?;

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
        let embedding = self.embed(vec![text.to_string()]).await?;

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

    /// Creates a model, one test at a time: with an empty `.fastembed_cache` (fresh CI
    /// checkout) parallel tests would download the same model files concurrently and
    /// trip over each other's partial downloads ("Failed to retrieve model file").
    fn new_model() -> FastEmbed {
        static CREATE: Mutex<()> = Mutex::new(());
        let _guard = CREATE.lock().unwrap_or_else(|e| e.into_inner());
        FastEmbed::try_new().unwrap()
    }

    #[tokio::test]
    async fn test_fastembed() {
        let fastembed = new_model();
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
        let first = new_model();
        drop(first);
        let second = new_model();
        let both_alive = new_model();
        assert_eq!(second.embed_query("hi").await.unwrap().len(), 384);
        assert_eq!(both_alive.embed_query("hi").await.unwrap().len(), 384);
    }

    /// Inference runs on the blocking pool: other tasks on the same (single) runtime thread
    /// keep running while the model works (#142).
    #[tokio::test]
    async fn fastembed_does_not_block_the_runtime() {
        let fastembed = new_model();
        let texts: Vec<String> = (0..64)
            .map(|i| format!("document number {i} about rust and async runtimes"))
            .collect();
        let (embeddings, ticks) =
            crate::blocking::testing::ticks_while(fastembed.embed_documents(&texts)).await;
        assert_eq!(embeddings.unwrap().len(), 64);
        assert!(ticks > 0, "embedding blocked the runtime thread");
    }
}
