use crate::{
    language_models::{
        box_llm_events, llm::LLM, options::CallOptions, GenerateResult, LLMError, LLMEvent,
        LLMEventStream, TokenUsage,
    },
    llm::AnthropicError,
    observability::{traced_event_stream, traced_generate},
    schemas::{Message, MessageType, StreamData},
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde_json::Value;
use std::pin::Pin;

use super::models::{
    result_from_blocks, thinking_params, to_claude_messages, tool_choice, ApiResponse, ClaudeTool,
    Content, Payload,
};
use crate::llm::sse::sse_json_events;

/// Claude model IDs. Any other ID can be passed to [`Claude::with_model`] as a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaudeModel {
    ClaudeFable5_1,
    ClaudeOpus5_5,
    ClaudeSonnet5_5,
    ClaudeHaiku5_5,
    #[deprecated(note = "retired by Anthropic; use ClaudeModel::ClaudeOpus5_5")]
    Claude3pus20240229,
    #[deprecated(note = "retired by Anthropic; use ClaudeModel::ClaudeSonnet5_5")]
    Claude3sonnet20240229,
    #[deprecated(note = "retired by Anthropic; use ClaudeModel::ClaudeHaiku5_5")]
    Claude3haiku20240307,
    #[deprecated(note = "retired by Anthropic; use ClaudeModel::ClaudeSonnet5_5")]
    Claude3_5sonnet20240620,
}

impl ClaudeModel {
    #[allow(deprecated)]
    pub fn as_str(&self) -> &'static str {
        match self {
            ClaudeModel::ClaudeFable5_1 => "claude-fable-5-1",
            ClaudeModel::ClaudeOpus5_5 => "claude-opus-5-5",
            ClaudeModel::ClaudeSonnet5_5 => "claude-sonnet-5-5",
            ClaudeModel::ClaudeHaiku5_5 => "claude-haiku-5-5",
            ClaudeModel::Claude3pus20240229 => "claude-3-opus-20240229",
            ClaudeModel::Claude3sonnet20240229 => "claude-3-sonnet-20240229",
            ClaudeModel::Claude3haiku20240307 => "claude-3-haiku-20240307",
            ClaudeModel::Claude3_5sonnet20240620 => "claude-3-5-sonnet-20240620",
        }
    }
}

impl std::fmt::Display for ClaudeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<ClaudeModel> for String {
    fn from(model: ClaudeModel) -> Self {
        model.as_str().to_string()
    }
}

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_MODEL: ClaudeModel = ClaudeModel::ClaudeSonnet5_5;
/// `max_tokens` when unset. With a thinking budget it is added on top of the budget.
const DEFAULT_MAX_TOKENS: u32 = 1024;
/// `max_tokens` when unset and the model picks its own thinking budget (adaptive), so
/// thinking doesn't use up the whole answer.
const DEFAULT_ADAPTIVE_MAX_TOKENS: u32 = 16_000;
/// `provider` field of the `llm.generate` span.
const PROVIDER: &str = "anthropic";

#[derive(Clone)]
pub struct Claude {
    model: String,
    options: CallOptions,
    api_key: String,
    anthropic_version: String,
    base_url: String,
}

impl Default for Claude {
    fn default() -> Self {
        Self::new()
    }
}

impl Claude {
    pub fn new() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            options: CallOptions::default(),
            api_key: std::env::var("CLAUDE_API_KEY").unwrap_or_default(),
            anthropic_version: "2023-06-01".to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_options(mut self, options: CallOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = api_key.into();
        self
    }

    pub fn with_anthropic_version<S: Into<String>>(mut self, version: S) -> Self {
        self.anthropic_version = version.into();
        self
    }

    /// Overrides the API base URL (default `https://api.anthropic.com`), e.g. for a proxy.
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    async fn send(&self, payload: &Payload) -> Result<reqwest::Response, LLMError> {
        let res = Client::new()
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .header("content-type", "application/json; charset=utf-8")
            .json(payload)
            .send()
            .await?;
        if res.status().is_success() {
            return Ok(res);
        }
        let status = res.status().as_u16();
        let body = res.text().await.unwrap_or_default();
        Err(status_error(status, &body))
    }

    async fn generate_blocking(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let payload = self.build_payload(messages, false);
        let res: ApiResponse = self.send(&payload).await?.json().await?;

        let tokens = Some(TokenUsage {
            prompt_tokens: res.usage.input_tokens,
            completion_tokens: res.usage.output_tokens,
            total_tokens: res.usage.input_tokens + res.usage.output_tokens,
        });

        Ok(result_from_blocks(&res.content, tokens))
    }

    /// Streams the response, forwarding text deltas to `streaming_func` while rebuilding the
    /// full content blocks (tool calls arrive as `input_json_delta` fragments).
    async fn generate_streaming(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let func = self
            .options
            .streaming_func
            .clone()
            .ok_or_else(|| LLMError::OtherError("streaming_func not set".into()))?;
        let payload = self.build_payload(messages, true);
        let res = self.send(&payload).await?;
        let events = sse_json_events(res.bytes_stream());
        futures::pin_mut!(events);

        let mut acc = StreamAccumulator::default();
        while let Some(event) = events.next().await {
            let event = check_event(event?)?;
            if let Some(Delta::Text(text)) = acc.push(&event)? {
                let mut func = func.lock().await;
                let _ = func(text).await;
            }
        }
        Ok(acc.finish())
    }

    fn build_payload(&self, messages: &[Message], stream: bool) -> Payload {
        let (system_messages, other_messages): (Vec<_>, Vec<_>) = messages
            .iter()
            .partition(|m| m.message_type == MessageType::SystemMessage);
        let system = (!system_messages.is_empty()).then(|| {
            system_messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n")
        });
        let tools = self
            .options
            .functions
            .as_ref()
            .filter(|f| !f.is_empty())
            .map(|f| f.iter().map(ClaudeTool::from).collect());
        let (thinking, output_config, budget) =
            thinking_params(&self.model, self.options.thinking.as_ref());
        // Anthropic rejects custom sampling while thinking.
        let sampling = thinking.is_none();
        let max_tokens = self
            .options
            .max_tokens
            .unwrap_or(match (&thinking, budget) {
                (_, Some(budget)) => budget + DEFAULT_MAX_TOKENS,
                (Some(_), None) => DEFAULT_ADAPTIVE_MAX_TOKENS,
                (None, None) => DEFAULT_MAX_TOKENS,
            });
        Payload {
            model: self.model.clone(),
            system,
            messages: to_claude_messages(&other_messages),
            max_tokens,
            stream: stream.then_some(true),
            stop_sequences: self.options.stop_words.clone(),
            temperature: self.options.temperature.filter(|_| sampling),
            top_p: self.options.top_p.filter(|_| sampling),
            top_k: self.options.top_k.filter(|_| sampling),
            tool_choice: tools
                .as_ref()
                .and(self.options.function_call_behavior.as_ref())
                .map(tool_choice),
            tools,
            thinking,
            output_config,
        }
    }
}

#[async_trait]
impl LLM for Claude {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        traced_generate(PROVIDER, &self.model, async {
            match &self.options.streaming_func {
                Some(_) => self.generate_streaming(messages).await,
                None => self.generate_blocking(messages).await,
            }
        })
        .await
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let payload = self.build_payload(messages, true);
        let res = self.send(&payload).await?;
        let stream = sse_json_events(res.bytes_stream()).map(|event| {
            let event = check_event(event?)?;
            let text = text_delta(&event).unwrap_or_default().to_string();
            let tokens = event
                .get("usage")
                .and_then(|u| u.get("output_tokens"))
                .and_then(Value::as_u64)
                .map(|out| TokenUsage::new(0, out as u32));
            Ok(StreamData::new(event, tokens, text))
        });
        Ok(Box::pin(stream))
    }

    fn generate_stream<'a>(&'a self, messages: &'a [Message]) -> LLMEventStream<'a> {
        let events = box_llm_events(async_stream::try_stream! {
            let payload = self.build_payload(messages, true);
            let res = self.send(&payload).await?;
            let events = sse_json_events(res.bytes_stream());
            futures::pin_mut!(events);

            let mut acc = StreamAccumulator::default();
            while let Some(event) = events.next().await {
                let event = check_event(event?)?;
                match acc.push(&event)? {
                    Some(Delta::Text(text)) => yield LLMEvent::Text(text),
                    Some(Delta::Thinking(thinking)) => yield LLMEvent::Thinking(thinking),
                    None => {}
                }
            }
            yield LLMEvent::Done(acc.finish());
        });
        traced_event_stream(PROVIDER, &self.model, events)
    }

    fn add_options(&mut self, options: CallOptions) {
        self.options.merge_options(options)
    }
}

fn text_delta(event: &Value) -> Option<&str> {
    (event["type"] == "content_block_delta" && event["delta"]["type"] == "text_delta")
        .then(|| event["delta"]["text"].as_str())
        .flatten()
}

/// What a stream event adds that callers see as it arrives.
#[derive(Debug, PartialEq)]
enum Delta {
    Text(String),
    Thinking(String),
}

/// Rebuilds the response content blocks from Anthropic stream events.
#[derive(Default)]
struct StreamAccumulator {
    blocks: Vec<Content>,
    partial_json: Vec<String>,
    usage: TokenUsage,
}

impl StreamAccumulator {
    /// Applies one event; returns the text or thinking delta to forward, if any.
    fn push(&mut self, event: &Value) -> Result<Option<Delta>, LLMError> {
        match event["type"].as_str().unwrap_or_default() {
            "message_start" => {
                let usage = &event["message"]["usage"];
                self.usage.prompt_tokens = usage["input_tokens"].as_u64().unwrap_or(0) as u32;
                self.usage.completion_tokens = usage["output_tokens"].as_u64().unwrap_or(0) as u32;
            }
            "content_block_start" => {
                let index = block_index(event)?;
                let block: Content = serde_json::from_value(event["content_block"].clone())?;
                if self.blocks.len() <= index {
                    self.blocks.resize_with(index + 1, Content::default);
                    self.partial_json.resize_with(index + 1, String::new);
                }
                self.blocks[index] = block;
            }
            "content_block_delta" => {
                let index = block_index(event)?;
                let block = self.blocks.get_mut(index).ok_or_else(|| {
                    LLMError::ParsingError(format!("delta for unknown content block {index}"))
                })?;
                let delta = &event["delta"];
                match delta["type"].as_str().unwrap_or_default() {
                    "text_delta" => {
                        let text = delta["text"].as_str().unwrap_or_default();
                        block.text.get_or_insert_with(String::new).push_str(text);
                        return Ok(Some(Delta::Text(text.to_string())));
                    }
                    "thinking_delta" => {
                        let thinking = delta["thinking"].as_str().unwrap_or_default();
                        block
                            .thinking
                            .get_or_insert_with(String::new)
                            .push_str(thinking);
                        return Ok(Some(Delta::Thinking(thinking.to_string())));
                    }
                    "signature_delta" => {
                        let signature = delta["signature"].as_str().unwrap_or_default();
                        block
                            .signature
                            .get_or_insert_with(String::new)
                            .push_str(signature);
                    }
                    "input_json_delta" => {
                        let part = delta["partial_json"].as_str().unwrap_or_default();
                        self.partial_json[index].push_str(part);
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = block_index(event)?;
                if let (Some(block), Some(json)) =
                    (self.blocks.get_mut(index), self.partial_json.get(index))
                {
                    if block.content_type == "tool_use" && !json.is_empty() {
                        block.input = Some(serde_json::from_str(json).map_err(|e| {
                            LLMError::ParsingError(format!("invalid tool input JSON: {e}"))
                        })?);
                    }
                }
            }
            "message_delta" => {
                if let Some(out) = event["usage"]["output_tokens"].as_u64() {
                    self.usage.completion_tokens = out as u32;
                }
            }
            _ => {}
        }
        Ok(None)
    }

    fn finish(self) -> GenerateResult {
        let usage = self.usage;
        result_from_blocks(
            &self.blocks,
            Some(TokenUsage::new(
                usage.prompt_tokens,
                usage.completion_tokens,
            )),
        )
    }
}

fn block_index(event: &Value) -> Result<usize, LLMError> {
    event["index"]
        .as_u64()
        .map(|i| i as usize)
        .ok_or_else(|| LLMError::ParsingError("stream event without index".into()))
}

/// Turns an in-stream `error` event into an error.
fn check_event(event: Value) -> Result<Value, LLMError> {
    if event["type"] == "error" {
        return Err(parse_error(&event));
    }
    Ok(event)
}

/// Maps a non-2xx response to an error, using the Anthropic error body when there is one.
fn status_error(status: u16, body: &str) -> LLMError {
    if let Ok(json) = serde_json::from_str::<Value>(body) {
        if json.get("error").is_some() {
            return parse_error(&json);
        }
    }
    let message = if body.is_empty() {
        format!("HTTP {status}")
    } else {
        body.to_string()
    };
    match status {
        400 => AnthropicError::InvalidRequestError(message).into(),
        401 => AnthropicError::AuthenticationError(message).into(),
        403 => AnthropicError::PermissionError(message).into(),
        404 => AnthropicError::NotFoundError(message).into(),
        429 => AnthropicError::RateLimitError(message).into(),
        529 | 503 => AnthropicError::OverloadedError(message).into(),
        _ => AnthropicError::ApiError(message).into(),
    }
}

fn parse_error(json: &Value) -> LLMError {
    let error_type = json["error"]["type"].as_str().unwrap_or("");
    let message = json["error"]["message"].as_str().unwrap_or("").to_string();
    match error_type {
        "invalid_request_error" => AnthropicError::InvalidRequestError(message).into(),
        "authentication_error" => AnthropicError::AuthenticationError(message).into(),
        "permission_error" => AnthropicError::PermissionError(message).into(),
        "not_found_error" => AnthropicError::NotFoundError(message).into(),
        "rate_limit_error" => AnthropicError::RateLimitError(message).into(),
        "api_error" => AnthropicError::ApiError(message).into(),
        "overloaded_error" => AnthropicError::OverloadedError(message).into(),
        _ => LLMError::OtherError(format!("Unknown Anthropic error: {json}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    #[ignore]
    async fn test_cloudia_generate() {
        let cloudia = Claude::new();

        let res = cloudia
            .generate(&[Message::new_human_message("Hi, how are you doing")])
            .await
            .unwrap();

        println!("{:?}", res)
    }

    #[test]
    #[ignore]
    async fn test_cloudia_stream() {
        let cloudia = Claude::new();
        let mut stream = cloudia
            .stream(&[Message::new_human_message("Hi, how are you doing")])
            .await
            .unwrap();
        while let Some(data) = stream.next().await {
            match data {
                Ok(value) => value.to_stdout().unwrap(),
                Err(e) => panic!("Error invoking LLMChain: {:?}", e),
            }
        }
    }
}

#[cfg(test)]
mod offline_tests {
    use super::*;
    use crate::language_models::{ThinkingBlock, ThinkingConfig, ThinkingDisplay, ThinkingEffort};
    use crate::schemas::{FunctionCallBehavior, FunctionDefinition};
    use serde_json::json;

    #[test]
    fn claude_payload_splits_system_message() {
        let claude = Claude::new()
            .with_model("claude-x")
            .with_options(CallOptions::new().with_stop_words(vec!["END".into()]));
        let payload = claude.build_payload(
            &[
                Message::new_system_message("sys"),
                Message::new_human_message("hi"),
                Message::new_ai_message("hello"),
            ],
            true,
        );
        let v = serde_json::to_value(&payload).unwrap();
        assert_eq!(v["model"], "claude-x");
        assert_eq!(v["system"], "sys");
        assert_eq!(v["max_tokens"], 1024);
        assert_eq!(v["stream"], true);
        assert_eq!(v["stop_sequences"], json!(["END"]));
        assert_eq!(
            v["messages"],
            json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        );
        assert!(v.get("tools").is_none());
        assert!(v.get("tool_choice").is_none());
    }

    #[test]
    fn claude_default_model_is_current() {
        let payload = Claude::new().build_payload(&[Message::new_human_message("hi")], false);
        assert_eq!(payload.model, "claude-sonnet-5-5");
        assert_eq!(ClaudeModel::ClaudeOpus5_5.to_string(), "claude-opus-5-5");
        let id: String = ClaudeModel::ClaudeHaiku5_5.into();
        assert_eq!(id, "claude-haiku-5-5");
        assert_eq!(
            Claude::new()
                .with_model(ClaudeModel::ClaudeFable5_1)
                .build_payload(&[], false)
                .model,
            "claude-fable-5-1"
        );
    }

    #[test]
    fn claude_payload_joins_multiple_system_messages() {
        let payload = Claude::new().build_payload(
            &[
                Message::new_system_message("a"),
                Message::new_system_message("b"),
                Message::new_human_message("hi"),
            ],
            false,
        );
        let v = serde_json::to_value(&payload).unwrap();
        assert_eq!(v["system"], "a\n\nb");
        assert!(v.get("stream").is_none());
    }

    #[test]
    fn claude_payload_sends_tools_and_tool_choice() {
        let schema = json!({"type": "object", "properties": {"city": {"type": "string"}}});
        let claude = Claude::new().with_options(
            CallOptions::new()
                .with_functions(vec![FunctionDefinition::new(
                    "weather",
                    "Get the weather",
                    schema.clone(),
                )])
                .with_function_call_behavior(FunctionCallBehavior::Named("weather".into())),
        );
        let v =
            serde_json::to_value(claude.build_payload(&[Message::new_human_message("hi")], false))
                .unwrap();
        assert_eq!(
            v["tools"],
            json!([{"name": "weather", "description": "Get the weather", "input_schema": schema}])
        );
        assert_eq!(v["tool_choice"], json!({"type": "tool", "name": "weather"}));
    }

    #[test]
    fn claude_payload_maps_tool_round_trip() {
        // The scratchpad the tools agent builds: AI message with OpenAI-like tool calls,
        // then one tool message per call.
        let calls = json!([
            {"id": "toolu_1", "type": "function",
             "function": {"name": "weather", "arguments": "{\"city\":\"Lima\"}"}},
            {"id": "toolu_2", "type": "function",
             "function": {"name": "time", "arguments": "now"}},
        ]);
        let payload = Claude::new().build_payload(
            &[
                Message::new_human_message("weather and time?"),
                Message::new_ai_message("").with_tool_calls(calls),
                Message::new_tool_message("sunny", "toolu_1"),
                Message::new_tool_message("noon", "toolu_2"),
            ],
            false,
        );
        let v = serde_json::to_value(&payload).unwrap();
        assert_eq!(
            v["messages"],
            json!([
                {"role": "user", "content": "weather and time?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "weather",
                     "input": {"city": "Lima"}},
                    {"type": "tool_use", "id": "toolu_2", "name": "time",
                     "input": {"input": "now"}},
                ]},
                // Both results in one user turn, no "tool" role.
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "noon"},
                ]},
            ])
        );
    }

    #[test]
    fn claude_payload_keeps_assistant_text_before_tool_use() {
        let calls = json!([{"id": "t", "type": "function",
                            "function": {"name": "f", "arguments": ""}}]);
        let payload = Claude::new().build_payload(
            &[Message::new_ai_message("Let me check").with_tool_calls(calls)],
            false,
        );
        let v = serde_json::to_value(&payload).unwrap();
        assert_eq!(
            v["messages"][0]["content"],
            json!([
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "id": "t", "name": "f", "input": {}},
            ])
        );
    }

    #[test]
    fn stream_accumulator_rebuilds_text_and_tool_calls() {
        let events = [
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 7, "output_tokens": 1}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me "}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "check"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": "}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"Lima\"}"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}}),
            json!({"type": "message_stop"}),
        ];
        let mut acc = StreamAccumulator::default();
        let mut forwarded = String::new();
        for e in &events {
            if let Some(Delta::Text(t)) = acc.push(e).unwrap() {
                forwarded.push_str(&t);
            }
        }
        assert_eq!(forwarded, "Let me check");
        let result = acc.finish();
        let tokens = result.tokens.unwrap();
        assert_eq!((tokens.prompt_tokens, tokens.completion_tokens), (7, 12));
        let calls: Value = serde_json::from_str(&result.generation).unwrap();
        assert_eq!(
            result.tool_calls,
            vec![crate::schemas::ToolCall::new(
                "toolu_1",
                "weather",
                r#"{"city":"Lima"}"#
            )]
        );
        assert_eq!(
            calls,
            json!([{"id": "toolu_1", "type": "function",
                    "function": {"name": "weather", "arguments": "{\"city\":\"Lima\"}"}}])
        );
    }

    #[test]
    fn stream_accumulator_text_only() {
        let mut acc = StreamAccumulator::default();
        for e in [
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
            json!({"type": "ping"}),
        ] {
            acc.push(&e).unwrap();
        }
        assert_eq!(acc.finish().generation, "Hi");
    }

    #[test]
    fn stream_accumulator_rebuilds_thinking_with_signature() {
        let events = [
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Lima is "}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "in Peru."}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "abc"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "redacted_thinking", "data": "ENC"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "Peru"}}),
            json!({"type": "content_block_stop", "index": 2}),
        ];
        let mut acc = StreamAccumulator::default();
        let deltas: Vec<Delta> = events.iter().filter_map(|e| acc.push(e).unwrap()).collect();
        assert_eq!(
            deltas,
            [
                Delta::Thinking("Lima is ".into()),
                Delta::Thinking("in Peru.".into()),
                Delta::Text("Peru".into()),
            ]
        );
        let result = acc.finish();
        assert_eq!(result.generation, "Peru", "thinking stays out of the text");
        assert_eq!(
            result.thinking,
            vec![
                ThinkingBlock::Thinking {
                    thinking: "Lima is in Peru.".into(),
                    signature: Some("sig-abc".into()),
                },
                ThinkingBlock::Redacted { data: "ENC".into() },
            ]
        );
    }

    fn payload_json(model: &str, options: CallOptions, messages: &[Message]) -> Value {
        serde_json::to_value(
            Claude::new()
                .with_model(model)
                .with_options(options)
                .build_payload(messages, false),
        )
        .unwrap()
    }

    fn sampling() -> CallOptions {
        CallOptions::new()
            .with_temperature(0.3)
            .with_top_p(0.9)
            .with_top_k(5)
    }

    #[test]
    fn thinking_budget_model_sends_budget_and_drops_sampling() {
        let hi = [Message::new_human_message("hi")];
        let v = payload_json(
            "claude-sonnet-4-5",
            sampling().with_thinking(ThinkingConfig::enabled().with_budget_tokens(4000)),
            &hi,
        );
        assert_eq!(
            v["thinking"],
            json!({"type": "enabled", "budget_tokens": 4000})
        );
        assert_eq!(v["max_tokens"], 4000 + 1024, "room for the answer");
        for key in ["temperature", "top_p", "top_k", "output_config"] {
            assert!(v.get(key).is_none(), "{key} must not be sent: {v}");
        }

        // Default and minimum budget is 1024, maximum 128k; explicit max_tokens wins.
        let v = payload_json(
            "claude-3-7-sonnet-latest",
            CallOptions::new().with_thinking(ThinkingConfig::enabled()),
            &hi,
        );
        assert_eq!(v["thinking"]["budget_tokens"], 1024);
        assert_eq!(v["max_tokens"], 2048);
        let v = payload_json(
            "claude-opus-4-1",
            CallOptions::new()
                .with_max_tokens(500)
                .with_thinking(ThinkingConfig::enabled().with_budget_tokens(10)),
            &hi,
        );
        assert_eq!(v["thinking"]["budget_tokens"], 1024);
        assert_eq!(v["max_tokens"], 500);
        let v = payload_json(
            "claude-opus-4-1",
            CallOptions::new().with_thinking(ThinkingConfig::enabled().with_budget_tokens(999_999)),
            &hi,
        );
        assert_eq!(v["thinking"]["budget_tokens"], 128_000);
    }

    #[test]
    fn thinking_adaptive_model_sends_type_display_and_effort() {
        let hi = [Message::new_human_message("hi")];
        let config = ThinkingConfig::enabled()
            .with_budget_tokens(5000)
            .with_effort(ThinkingEffort::High)
            .with_display(ThinkingDisplay::Summarized);
        for model in [
            "claude-opus-4-7",
            "claude-fable-5-1",
            "claude-sonnet-5-5",
            "claude-opus-4-9",
        ] {
            let v = payload_json(model, sampling().with_thinking(config.clone()), &hi);
            assert_eq!(
                v["thinking"],
                json!({"type": "adaptive", "display": "summarized"}),
                "{model}"
            );
            assert_eq!(v["output_config"], json!({"effort": "high"}), "{model}");
            assert_eq!(v["max_tokens"], 16_000, "{model}");
            assert!(v.get("temperature").is_none(), "{model}");
        }
        let v = payload_json(
            "claude-opus-4-7",
            CallOptions::new().with_thinking(ThinkingConfig::enabled()),
            &hi,
        );
        assert_eq!(v["thinking"], json!({"type": "adaptive"}));
        assert!(v.get("output_config").is_none());
    }

    #[test]
    fn thinking_effort_only_where_supported() {
        let hi = [Message::new_human_message("hi")];
        let config = ThinkingConfig::enabled().with_effort(ThinkingEffort::Low);
        let v = payload_json(
            "claude-sonnet-4-6",
            CallOptions::new().with_thinking(config.clone()),
            &hi,
        );
        assert_eq!(v["output_config"], json!({"effort": "low"}));
        assert_eq!(v["thinking"]["type"], "enabled");
        let v = payload_json(
            "claude-sonnet-4-5",
            CallOptions::new().with_thinking(config),
            &hi,
        );
        assert!(v.get("output_config").is_none());
    }

    #[test]
    fn thinking_disabled_or_unsupported_sends_nothing() {
        let hi = [Message::new_human_message("hi")];
        for (model, config) in [
            ("claude-fable-5-1", ThinkingConfig::disabled()),
            ("claude-sonnet-4-5", ThinkingConfig::disabled()),
            // Off means off, even with a budget left over.
            (
                "claude-sonnet-4-5",
                ThinkingConfig::disabled().with_budget_tokens(4000),
            ),
            (
                "claude-3-5-haiku-latest",
                ThinkingConfig::enabled().with_budget_tokens(2000),
            ),
        ] {
            let v = payload_json(model, sampling().with_thinking(config), &hi);
            assert!(v.get("thinking").is_none(), "{model}: {v}");
            assert_eq!(v["max_tokens"], 1024, "{model}");
            assert_eq!(v["top_k"], 5, "sampling kept without thinking ({model})");
        }
        // Disabled still carries effort on models that take it.
        let v = payload_json(
            "claude-fable-5-1",
            CallOptions::new()
                .with_thinking(ThinkingConfig::disabled().with_effort(ThinkingEffort::Low)),
            &hi,
        );
        assert!(v.get("thinking").is_none());
        assert_eq!(v["output_config"], json!({"effort": "low"}));
    }

    #[test]
    fn model_capabilities_follow_langchaingo() {
        use super::super::models::model_capabilities;
        let caps = |m: &str| {
            let c = model_capabilities(m);
            (c.thinking, c.adaptive, c.effort)
        };
        assert_eq!(caps("claude-3-5-sonnet-20240620"), (false, false, false));
        assert_eq!(caps("claude-3-7-sonnet-20250219"), (true, false, false));
        assert_eq!(caps("claude-sonnet-4-20250514"), (true, false, false));
        assert_eq!(caps("claude-opus-4-6"), (true, false, true));
        assert_eq!(caps("claude-opus-4-8"), (true, true, true));
        assert_eq!(caps("claude-haiku-4-5"), (true, false, false));
        assert_eq!(
            caps("claude-haiku-4-6"),
            (true, true, true),
            "past the last listed minor"
        );
        assert_eq!(caps("claude-opus-4"), (true, false, false));
        assert_eq!(caps("my-gateway-alias"), (true, false, false));
        assert_eq!(caps("Claude-Sonnet-5-5"), (true, true, true));
    }

    #[test]
    fn assistant_thinking_goes_back_first_and_signed_only() {
        let calls = json!([{"id": "t", "type": "function",
                            "function": {"name": "f", "arguments": ""}}]);
        let message = Message::new_ai_message("ok")
            .with_tool_calls(calls)
            .with_thinking(vec![
                ThinkingBlock::Thinking {
                    thinking: "plan".into(),
                    signature: Some("sig".into()),
                },
                ThinkingBlock::text("unsigned, from another provider"),
                ThinkingBlock::Redacted { data: "ENC".into() },
            ]);
        let v = payload_json("claude-sonnet-4-5", CallOptions::new(), &[message]);
        assert_eq!(
            v["messages"][0]["content"],
            json!([
                {"type": "thinking", "thinking": "plan", "signature": "sig"},
                {"type": "redacted_thinking", "data": "ENC"},
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "t", "name": "f", "input": {}},
            ])
        );

        // Text-only replies with signed thinking also become blocks.
        let message =
            Message::new_ai_message("answer").with_thinking(vec![ThinkingBlock::Thinking {
                thinking: "hm".into(),
                signature: Some("s".into()),
            }]);
        let v = payload_json("claude-sonnet-4-5", CallOptions::new(), &[message]);
        assert_eq!(
            v["messages"][0]["content"],
            json!([
                {"type": "thinking", "thinking": "hm", "signature": "s"},
                {"type": "text", "text": "answer"},
            ])
        );
        // Only unsigned thinking: plain string as before.
        let message =
            Message::new_ai_message("answer").with_thinking(vec![ThinkingBlock::text("x")]);
        let v = payload_json("claude-sonnet-4-5", CallOptions::new(), &[message]);
        assert_eq!(v["messages"][0]["content"], "answer");
    }

    #[test]
    fn claude_error_mapping() {
        let err = check_event(
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            LLMError::AnthropicError(AnthropicError::OverloadedError(m)) if m == "busy"
        ));

        let err = status_error(
            400,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad role"}}"#,
        );
        assert!(matches!(
            err,
            LLMError::AnthropicError(AnthropicError::InvalidRequestError(m)) if m == "bad role"
        ));

        assert!(matches!(
            status_error(401, ""),
            LLMError::AnthropicError(AnthropicError::AuthenticationError(_))
        ));
        assert!(matches!(
            status_error(529, "<html>"),
            LLMError::AnthropicError(AnthropicError::OverloadedError(_))
        ));
    }
}
