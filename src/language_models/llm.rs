use std::error::Error;

use async_trait::async_trait;

use super::GenerateResult;

#[async_trait]
pub trait LLM: Sync + Send {
    async fn generate(&self, prompt: &str) -> Result<GenerateResult, Box<dyn Error>>;
}
