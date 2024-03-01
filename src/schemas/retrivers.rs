use std::error::Error;

use async_trait::async_trait;

use super::Document;

#[async_trait]
pub trait Retriever: Sync + Send {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>, Box<dyn Error>>;
}
