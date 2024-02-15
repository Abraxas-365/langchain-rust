use serde::{Deserialize, Serialize};

pub mod chat_model;
pub mod llm;

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateResult {
    pub tokens: Option<TokenUsage>,
    pub generation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
impl Default for GenerateResult {
    fn default() -> Self {
        Self {
            tokens: Default::default(),
            generation: Default::default(),
        }
    }
}
