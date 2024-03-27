use async_trait::async_trait;

use super::EmbedderError;

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError>;
    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError>;
}
