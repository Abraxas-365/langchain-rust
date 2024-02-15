use std::error::Error;

use async_trait::async_trait;

use crate::schemas::messages::Message;

use super::GenerateResult;

#[async_trait]
pub trait LLMChat: Sync + Send {
    async fn generate(&self, prompt: &[Message]) -> Result<GenerateResult, Box<dyn Error>>;
}
