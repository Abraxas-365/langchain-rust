use std::error::Error;

use async_trait::async_trait;

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f32>>, Box<dyn Error>>;
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>>;
}
