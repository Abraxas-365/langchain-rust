use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::schemas::messages::Message;

use super::{options::CallOptions, GenerateResult, LLMError};

#[async_trait]
pub trait LLM: Sync + Send {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError>;
    async fn invoke(&self, prompt: &str) -> Result<String, LLMError>;
    async fn stream(
        &self,
        _messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<serde_json::Value, LLMError>> + Send>>, LLMError>;

    /// This is usefull when you want to create a chain and override
    /// LLM options
    fn add_options(&mut self, _options: CallOptions) {
        // No action taken
    }
    //This is usefull when using non chat models
    fn messages_to_string(&self, messages: &[Message]) -> String {
        messages
            .iter()
            .map(|m| format!("{:?}: {}", m.message_type, m.content))
            .collect::<Vec<String>>()
            .join("\n")
    }
}
