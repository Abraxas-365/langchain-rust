use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub mod llm;
pub mod options;

mod error;
pub use error::*;

//TODO: check if its this should have a data:serde::Value to save all other things, like OpenAI
//function responses
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GenerateResult {
    pub tokens: Option<TokenUsage>,
    pub generation: String,
}

impl GenerateResult {
    pub fn to_hashmap(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();

        // Insert the 'generation' field into the hashmap
        map.insert("generation".to_string(), self.generation.clone());

        // Check if 'tokens' is Some and insert its fields into the hashmap
        if let Some(ref tokens) = self.tokens {
            map.insert(
                "prompt_tokens".to_string(),
                tokens.prompt_tokens.to_string(),
            );
            map.insert(
                "completion_tokens".to_string(),
                tokens.completion_tokens.to_string(),
            );
            map.insert("total_tokens".to_string(), tokens.total_tokens.to_string());
        }

        map
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    pub fn sum(&self, other: &TokenUsage) -> TokenUsage {
        TokenUsage {
            prompt_tokens: self.prompt_tokens + other.prompt_tokens,
            completion_tokens: self.completion_tokens + other.completion_tokens,
            total_tokens: self.total_tokens + other.total_tokens,
        }
    }

    pub fn add(&mut self, other: &TokenUsage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_new_sum_add() {
        let a = TokenUsage::new(3, 4);
        assert_eq!(a.total_tokens, 7);
        let b = TokenUsage::new(1, 2);
        let s = a.sum(&b);
        assert_eq!(
            (s.prompt_tokens, s.completion_tokens, s.total_tokens),
            (4, 6, 10)
        );
        let mut c = a.clone();
        c.add(&b);
        assert_eq!(c.total_tokens, 10);
        // sum does not mutate
        assert_eq!(a.total_tokens, 7);
    }

    #[test]
    fn generate_result_to_hashmap() {
        let without = GenerateResult {
            generation: "hi".into(),
            tokens: None,
        };
        let m = without.to_hashmap();
        assert_eq!(m.len(), 1);
        assert_eq!(m["generation"], "hi");

        let with = GenerateResult {
            generation: "hi".into(),
            tokens: Some(TokenUsage::new(1, 2)),
        };
        let m = with.to_hashmap();
        assert_eq!(m["prompt_tokens"], "1");
        assert_eq!(m["completion_tokens"], "2");
        assert_eq!(m["total_tokens"], "3");
    }
}
