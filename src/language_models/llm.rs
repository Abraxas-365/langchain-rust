use std::error::Error;

use async_trait::async_trait;

use crate::schemas::messages::Message;

use super::{options::CallOptions, GenerateResult};

#[async_trait]
pub trait LLM: Sync + Send {
    async fn generate(&self, messgaes: &[Message]) -> Result<GenerateResult, Box<dyn Error>>;
    async fn invoke(&self, prompt: &str) -> Result<String, Box<dyn Error>>;
    fn with_options(&mut self, _options: CallOptions) {
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
