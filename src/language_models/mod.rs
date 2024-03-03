use serde::{Deserialize, Serialize};

pub mod llm;
pub mod options;

//TODO: check if its this should have a data:serde::Value to save all other things, like OpenAI
//function responses
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GenerateResult {
    pub tokens: Option<TokenUsage>,
    pub generation: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Default for TokenUsage {
    fn default() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }
    }
}

impl TokenUsage {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

impl Default for GenerateResult {
    fn default() -> Self {
        Self {
            tokens: Default::default(),
            generation: Default::default(),
        }
    }
}
