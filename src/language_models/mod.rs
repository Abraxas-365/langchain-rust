use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::schemas::ToolCall;

pub mod llm;
pub mod options;

mod error;
pub use error::*;

/// The result of an LLM call.
///
/// When the model asks for tools, `tool_calls` holds them (same type for every provider)
/// and `generation` holds the same calls as an OpenAI-style JSON array, so code that
/// reads `generation` keeps working. Use [`GenerateResult::with_tool_calls`] to build one.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GenerateResult {
    pub tokens: Option<TokenUsage>,
    pub generation: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl GenerateResult {
    /// Sets `tool_calls` and mirrors them into `generation` as JSON. Empty `calls`
    /// leave the result untouched.
    pub fn with_tool_calls(mut self, calls: Vec<ToolCall>) -> Self {
        if !calls.is_empty() {
            self.generation = ToolCall::list_to_string(&calls);
            self.tool_calls = calls;
        }
        self
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

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
            ..Default::default()
        };
        let m = without.to_hashmap();
        assert_eq!(m.len(), 1);
        assert_eq!(m["generation"], "hi");

        let with = GenerateResult {
            generation: "hi".into(),
            tokens: Some(TokenUsage::new(1, 2)),
            ..Default::default()
        };
        let m = with.to_hashmap();
        assert_eq!(m["prompt_tokens"], "1");
        assert_eq!(m["completion_tokens"], "2");
        assert_eq!(m["total_tokens"], "3");
    }

    #[test]
    fn with_tool_calls_mirrors_into_generation() {
        let calls = vec![ToolCall::new("c1", "weather", r#"{"city":"Lima"}"#)];
        let r = GenerateResult {
            generation: "let me check".into(),
            ..Default::default()
        }
        .with_tool_calls(calls.clone());
        assert!(r.has_tool_calls());
        assert_eq!(r.tool_calls, calls);
        assert_eq!(ToolCall::parse_list(&r.generation).unwrap(), calls);

        let plain = GenerateResult {
            generation: "hi".into(),
            ..Default::default()
        }
        .with_tool_calls(vec![]);
        assert!(!plain.has_tool_calls());
        assert_eq!(plain.generation, "hi");
    }

    #[test]
    fn generate_result_serde_is_backward_compatible() {
        // Old payloads without `tool_calls` still deserialize.
        let old: GenerateResult =
            serde_json::from_str(r#"{"tokens":null,"generation":"hi"}"#).unwrap();
        assert!(old.tool_calls.is_empty());
        // No tool calls -> field omitted.
        let v = serde_json::to_value(&old).unwrap();
        assert!(v.get("tool_calls").is_none());

        let with = GenerateResult::default().with_tool_calls(vec![ToolCall::new("a", "f", "{}")]);
        let v = serde_json::to_value(&with).unwrap();
        assert_eq!(v["tool_calls"][0]["function"]["name"], "f");
        let back: GenerateResult = serde_json::from_value(v).unwrap();
        assert_eq!(back.tool_calls, with.tool_calls);
    }
}
