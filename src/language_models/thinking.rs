//! Reasoning ("thinking") settings and the reasoning a model returns.
//!
//! Set a [`ThinkingConfig`] with [`CallOptions::with_thinking`](super::options::CallOptions::with_thinking);
//! each provider maps it to its own parameter (Anthropic `thinking` / `output_config.effort`,
//! OpenAI `reasoning_effort`, Ollama `think`). What the model thought comes back in
//! [`GenerateResult::thinking`](super::GenerateResult::thinking) and, when streaming, as
//! [`LLMEvent::Thinking`](super::LLMEvent::Thinking).

use serde::{Deserialize, Serialize};

/// How hard a model should think, for providers that take an effort level. Providers
/// with a shorter scale use their closest level (Ollama has no `XHigh` or `Max`, so
/// both become `high`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ThinkingEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl ThinkingEffort {
    pub fn as_str(&self) -> &'static str {
        match self {
            ThinkingEffort::Low => "low",
            ThinkingEffort::Medium => "medium",
            ThinkingEffort::High => "high",
            ThinkingEffort::XHigh => "xhigh",
            ThinkingEffort::Max => "max",
        }
    }
}

/// How a model that summarizes its thinking returns it (Anthropic adaptive thinking).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ThinkingDisplay {
    /// A summary of the thinking.
    Summarized,
    /// No thinking text; the model still thinks.
    Omitted,
}

impl ThinkingDisplay {
    pub fn as_str(&self) -> &'static str {
        match self {
            ThinkingDisplay::Summarized => "summarized",
            ThinkingDisplay::Omitted => "omitted",
        }
    }
}

/// Reasoning settings for a call.
///
/// ```rust
/// use langchain_rust::language_models::{options::CallOptions, ThinkingConfig, ThinkingEffort};
///
/// let options = CallOptions::new()
///     .with_thinking(ThinkingConfig::enabled().with_effort(ThinkingEffort::High));
/// ```
///
/// Providers ignore what they don't support: a token budget means nothing to OpenAI,
/// and an effort level means nothing to Claude models that only take a budget.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ThinkingConfig {
    /// Whether the model should think. `false` turns thinking off where the provider has
    /// a switch for it (Ollama, Claude); OpenAI reasoning models always reason.
    pub enabled: bool,
    /// Tokens the model may spend thinking (Claude budget-based models; minimum 1024).
    pub budget_tokens: Option<u32>,
    /// Effort level (OpenAI `reasoning_effort`, Anthropic `output_config.effort`, Ollama `think`).
    pub effort: Option<ThinkingEffort>,
    /// How summarized thinking is returned (Anthropic adaptive thinking).
    pub display: Option<ThinkingDisplay>,
}

impl ThinkingConfig {
    /// Thinking on, with the provider's defaults.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Thinking off.
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn with_budget_tokens(mut self, budget_tokens: u32) -> Self {
        self.budget_tokens = Some(budget_tokens);
        self
    }

    pub fn with_effort(mut self, effort: ThinkingEffort) -> Self {
        self.effort = Some(effort);
        self
    }

    pub fn with_display(mut self, display: ThinkingDisplay) -> Self {
        self.display = Some(display);
        self
    }
}

/// A piece of model reasoning, kept as the provider returned it so it can be sent back
/// in later turns. Claude requires its thinking blocks, signature included, to come
/// back unchanged in a tool-use loop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ThinkingBlock {
    /// Reasoning text. `signature` is set by providers that sign their reasoning (Anthropic).
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Reasoning the provider encrypted (Anthropic `redacted_thinking`). Opaque; only
    /// useful to send back.
    Redacted { data: String },
}

impl ThinkingBlock {
    /// Reasoning text without a signature.
    pub fn text<S: Into<String>>(thinking: S) -> Self {
        ThinkingBlock::Thinking {
            thinking: thinking.into(),
            signature: None,
        }
    }

    /// The readable reasoning, empty for redacted blocks.
    pub fn thinking_text(&self) -> &str {
        match self {
            ThinkingBlock::Thinking { thinking, .. } => thinking,
            ThinkingBlock::Redacted { .. } => "",
        }
    }
}

/// The readable text of `blocks`, joined by blank lines (redacted blocks are skipped).
pub(crate) fn join_thinking(blocks: &[ThinkingBlock]) -> String {
    blocks
        .iter()
        .map(ThinkingBlock::thinking_text)
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn config_builders() {
        let config = ThinkingConfig::enabled()
            .with_budget_tokens(2048)
            .with_effort(ThinkingEffort::XHigh)
            .with_display(ThinkingDisplay::Omitted);
        assert!(config.enabled);
        assert_eq!(config.budget_tokens, Some(2048));
        assert_eq!(config.effort, Some(ThinkingEffort::XHigh));
        assert_eq!(config.display, Some(ThinkingDisplay::Omitted));
        assert!(!ThinkingConfig::disabled().enabled);
        assert_eq!(ThinkingConfig::disabled(), ThinkingConfig::default());
    }

    #[test]
    fn effort_and_display_strings_match_the_wire_values() {
        let efforts = [
            (ThinkingEffort::Low, "low"),
            (ThinkingEffort::Medium, "medium"),
            (ThinkingEffort::High, "high"),
            (ThinkingEffort::XHigh, "xhigh"),
            (ThinkingEffort::Max, "max"),
        ];
        for (effort, s) in efforts {
            assert_eq!(effort.as_str(), s);
            assert_eq!(serde_json::to_value(effort).unwrap(), json!(s));
        }
        assert_eq!(ThinkingDisplay::Summarized.as_str(), "summarized");
        assert_eq!(ThinkingDisplay::Omitted.as_str(), "omitted");
        assert_eq!(
            serde_json::to_value(ThinkingDisplay::Omitted).unwrap(),
            json!("omitted")
        );
    }

    #[test]
    fn blocks_serialize_with_a_type_tag() {
        let signed = ThinkingBlock::Thinking {
            thinking: "hmm".into(),
            signature: Some("sig".into()),
        };
        assert_eq!(
            serde_json::to_value(&signed).unwrap(),
            json!({"type": "thinking", "thinking": "hmm", "signature": "sig"})
        );
        assert_eq!(
            serde_json::to_value(ThinkingBlock::text("a")).unwrap(),
            json!({"type": "thinking", "thinking": "a"})
        );
        let redacted = ThinkingBlock::Redacted { data: "x".into() };
        let value = serde_json::to_value(&redacted).unwrap();
        assert_eq!(value, json!({"type": "redacted", "data": "x"}));
        assert_eq!(
            serde_json::from_value::<ThinkingBlock>(value).unwrap(),
            redacted
        );
    }

    #[test]
    fn join_thinking_skips_redacted_and_empty_blocks() {
        let blocks = [
            ThinkingBlock::text("one"),
            ThinkingBlock::Redacted { data: "x".into() },
            ThinkingBlock::text(""),
            ThinkingBlock::text("two"),
        ];
        assert_eq!(join_thinking(&blocks), "one\n\ntwo");
        assert_eq!(join_thinking(&[]), "");
    }
}
