use std::{collections::HashMap, pin::Pin};

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::schemas::ToolCall;

pub mod llm;
pub mod options;

mod error;
pub use error::*;

pub mod thinking;
pub use thinking::{ThinkingBlock, ThinkingConfig, ThinkingDisplay, ThinkingEffort};

/// The result of an LLM call.
///
/// When the model asks for tools, `tool_calls` holds them (same type for every provider)
/// and `generation` holds the same calls as an OpenAI-style JSON array, so code that
/// reads `generation` keeps working. Use [`GenerateResult::with_tool_calls`] to build one.
///
/// `thinking` holds the model's reasoning when the provider returns it (see
/// [`ThinkingConfig`]); `generation` never includes it.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GenerateResult {
    pub tokens: Option<TokenUsage>,
    pub generation: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub thinking: Vec<ThinkingBlock>,
}

impl GenerateResult {
    /// Sets the model's reasoning blocks.
    pub fn with_thinking(mut self, thinking: Vec<ThinkingBlock>) -> Self {
        self.thinking = thinking;
        self
    }

    /// The readable reasoning, blocks joined by blank lines (empty when there is none).
    pub fn thinking_text(&self) -> String {
        thinking::join_thinking(&self.thinking)
    }

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

/// An event from [`llm::LLM::generate_stream`] (and `Chain::call_stream`).
///
/// Zero or more `Text` events arrive as the model writes, then exactly one `Done` with
/// the complete result: full text, typed `tool_calls`, reasoning and token usage when
/// the provider reports them. Models that reason send `Thinking` chunks too, usually
/// before the text.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum LLMEvent {
    /// A chunk of text, as the model produces it.
    Text(String),
    /// A chunk of the model's reasoning, as it produces it. Never part of `Text`.
    Thinking(String),
    /// The complete response. Always the last event.
    Done(GenerateResult),
}

/// Stream returned by [`llm::LLM::generate_stream`].
pub type LLMEventStream<'a> = Pin<Box<dyn Stream<Item = Result<LLMEvent, LLMError>> + Send + 'a>>;

/// Boxes a stream of LLM events; also pins down the error type for `try_stream!` blocks.
pub(crate) fn box_llm_events<'a, S>(stream: S) -> LLMEventStream<'a>
where
    S: Stream<Item = Result<LLMEvent, LLMError>> + Send + 'a,
{
    Box::pin(stream)
}

/// The events for a result that was not streamed: its reasoning and its text, each in
/// one piece (no text when the model asked for tools, as `generation` is then just the
/// JSON mirror), then `Done`.
pub(crate) fn result_events(result: GenerateResult) -> Vec<LLMEvent> {
    let mut events = Vec::with_capacity(3);
    let thinking = result.thinking_text();
    if !thinking.is_empty() {
        events.push(LLMEvent::Thinking(thinking));
    }
    if !result.has_tool_calls() && !result.generation.is_empty() {
        events.push(LLMEvent::Text(result.generation.clone()));
    }
    events.push(LLMEvent::Done(result));
    events
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq, Eq)]
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

    #[test]
    fn result_events_skip_text_for_tool_calls_and_empty_output() {
        let text = |events: &[LLMEvent]| {
            events
                .iter()
                .filter_map(|e| match e {
                    LLMEvent::Text(t) => Some(t.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let plain = result_events(GenerateResult {
            generation: "hi".into(),
            ..Default::default()
        });
        assert_eq!(text(&plain), vec!["hi"]);
        assert!(matches!(plain.last(), Some(LLMEvent::Done(r)) if r.generation == "hi"));

        let tools = result_events(
            GenerateResult::default().with_tool_calls(vec![ToolCall::new("a", "f", "{}")]),
        );
        assert!(text(&tools).is_empty(), "tool-call JSON is not answer text");
        assert!(matches!(tools.as_slice(), [LLMEvent::Done(r)] if r.has_tool_calls()));

        assert!(matches!(
            result_events(GenerateResult::default()).as_slice(),
            [LLMEvent::Done(_)]
        ));
    }

    #[test]
    fn result_events_send_thinking_before_text() {
        let result = GenerateResult {
            generation: "4".into(),
            ..Default::default()
        }
        .with_thinking(vec![
            ThinkingBlock::text("2+2"),
            ThinkingBlock::text("is 4"),
        ]);
        let events = result_events(result);
        assert!(
            matches!(events.as_slice(), [LLMEvent::Thinking(t), LLMEvent::Text(x), LLMEvent::Done(r)]
                if t == "2+2\n\nis 4" && x == "4" && r.thinking.len() == 2),
            "{events:?}"
        );

        // Thinking with tool calls: thinking, then Done (no JSON text).
        let tools = result_events(
            GenerateResult::default()
                .with_tool_calls(vec![ToolCall::new("a", "f", "{}")])
                .with_thinking(vec![ThinkingBlock::text("use f")]),
        );
        assert!(matches!(
            tools.as_slice(),
            [LLMEvent::Thinking(_), LLMEvent::Done(_)]
        ));
    }

    #[test]
    fn generate_result_thinking_serde() {
        let old: GenerateResult =
            serde_json::from_str(r#"{"tokens":null,"generation":"hi"}"#).unwrap();
        assert!(old.thinking.is_empty());
        assert!(serde_json::to_value(&old)
            .unwrap()
            .get("thinking")
            .is_none());

        let with = GenerateResult::default().with_thinking(vec![ThinkingBlock::Thinking {
            thinking: "t".into(),
            signature: Some("s".into()),
        }]);
        let v = serde_json::to_value(&with).unwrap();
        assert_eq!(v["thinking"][0]["signature"], "s");
        let back: GenerateResult = serde_json::from_value(v).unwrap();
        assert_eq!(back.thinking, with.thinking);
        assert_eq!(back.thinking_text(), "t");
    }

    /// An LLM that only implements `generate` still streams through the default method.
    #[derive(Clone)]
    struct OnlyGenerate(Result<&'static str, &'static str>);

    #[async_trait::async_trait]
    impl llm::LLM for OnlyGenerate {
        async fn generate(
            &self,
            _: &[crate::schemas::Message],
        ) -> Result<GenerateResult, LLMError> {
            match self.0 {
                Ok(s) => Ok(GenerateResult {
                    generation: s.into(),
                    ..Default::default()
                }),
                Err(e) => Err(LLMError::OtherError(e.into())),
            }
        }
        async fn stream(
            &self,
            _: &[crate::schemas::Message],
        ) -> Result<
            Pin<Box<dyn Stream<Item = Result<crate::schemas::StreamData, LLMError>> + Send>>,
            LLMError,
        > {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn default_generate_stream_wraps_generate() {
        use futures::StreamExt;
        use llm::LLM;

        let events: Vec<_> = OnlyGenerate(Ok("hello"))
            .generate_stream(&[])
            .map(Result::unwrap)
            .collect()
            .await;
        assert!(matches!(&events[0], LLMEvent::Text(t) if t == "hello"));
        assert!(matches!(&events[1], LLMEvent::Done(r) if r.generation == "hello"));
        assert_eq!(events.len(), 2);

        let events: Vec<_> = OnlyGenerate(Err("down"))
            .generate_stream(&[])
            .collect()
            .await;
        assert_eq!(events.len(), 1);
        assert!(events[0].as_ref().unwrap_err().to_string().contains("down"));
    }
}
