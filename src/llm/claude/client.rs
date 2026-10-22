use crate::{
    language_models::{llm::LLM, options::CallOptions, GenerateResult, LLMError, TokenUsage},
    llm::AnthropicError,
    schemas::{Message, MessageType, StreamData},
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde_json::Value;
use std::pin::Pin;

use super::models::{
    generation_from_blocks, to_claude_messages, tool_choice, ApiResponse, ClaudeTool, Content,
    Payload,
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

        Ok(GenerateResult {
            tokens,
            generation: generation_from_blocks(&res.content),
        })
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
            if let Some(text) = acc.push(&event)? {
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
        Payload {
            model: self.model.clone(),
            system,
            messages: to_claude_messages(&other_messages),
            max_tokens: self.options.max_tokens.unwrap_or(1024),
            stream: stream.then_some(true),
            stop_sequences: self.options.stop_words.clone(),
            temperature: self.options.temperature,
            top_p: self.options.top_p,
            top_k: self.options.top_k,
            tool_choice: tools
                .as_ref()
                .and(self.options.function_call_behavior.as_ref())
                .map(tool_choice),
            tools,
        }
    }
}

#[async_trait]
impl LLM for Claude {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        match &self.options.streaming_func {
            Some(_) => self.generate_streaming(messages).await,
            None => self.generate_blocking(messages).await,
        }
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

    fn add_options(&mut self, options: CallOptions) {
        self.options.merge_options(options)
    }
}

fn text_delta(event: &Value) -> Option<&str> {
    (event["type"] == "content_block_delta" && event["delta"]["type"] == "text_delta")
        .then(|| event["delta"]["text"].as_str())
        .flatten()
}

/// Rebuilds the response content blocks from Anthropic stream events.
#[derive(Default)]
struct StreamAccumulator {
    blocks: Vec<Content>,
    partial_json: Vec<String>,
    usage: TokenUsage,
}

impl StreamAccumulator {
    /// Applies one event; returns the text delta to forward, if any.
    fn push(&mut self, event: &Value) -> Result<Option<String>, LLMError> {
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
                        return Ok(Some(text.to_string()));
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
        GenerateResult {
            tokens: Some(TokenUsage::new(
                usage.prompt_tokens,
                usage.completion_tokens,
            )),
            generation: generation_from_blocks(&self.blocks),
        }
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
            if let Some(t) = acc.push(e).unwrap() {
                forwarded.push_str(&t);
            }
        }
        assert_eq!(forwarded, "Let me check");
        let result = acc.finish();
        let tokens = result.tokens.unwrap();
        assert_eq!((tokens.prompt_tokens, tokens.completion_tokens), (7, 12));
        let calls: Value = serde_json::from_str(&result.generation).unwrap();
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
