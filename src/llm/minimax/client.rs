use crate::{
    language_models::{llm::LLM, options::CallOptions, GenerateResult, LLMError, TokenUsage},
    llm::MiniMaxError,
    schemas::{Message, StreamData},
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde_json::Value;
use std::{pin::Pin, str};

use super::models::{ApiResponse, MiniMaxMessage, Payload, ResponseFormat};

pub enum MiniMaxModel {
    MiniMaxM27,
    MiniMaxM27Highspeed,
    MiniMaxM25,
    MiniMaxM25Highspeed,
}

impl ToString for MiniMaxModel {
    fn to_string(&self) -> String {
        match self {
            MiniMaxModel::MiniMaxM27 => "MiniMax-M2.7".to_string(),
            MiniMaxModel::MiniMaxM27Highspeed => "MiniMax-M2.7-highspeed".to_string(),
            MiniMaxModel::MiniMaxM25 => "MiniMax-M2.5".to_string(),
            MiniMaxModel::MiniMaxM25Highspeed => "MiniMax-M2.5-highspeed".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct MiniMax {
    model: String,
    options: CallOptions,
    api_key: String,
    base_url: String,
    json_mode: bool,
}

impl Default for MiniMax {
    fn default() -> Self {
        Self::new()
    }
}

impl MiniMax {
    pub fn new() -> Self {
        Self {
            model: MiniMaxModel::MiniMaxM27.to_string(),
            options: CallOptions::default(),
            api_key: std::env::var("MINIMAX_API_KEY").unwrap_or_default(),
            base_url: "https://api.minimax.io".to_string(),
            json_mode: false,
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

    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_json_mode(mut self, json_mode: bool) -> Self {
        self.json_mode = json_mode;
        self
    }

    /// Clamp temperature to MiniMax's accepted range [0.0, 1.0].
    fn clamp_temperature(temp: Option<f32>) -> Option<f32> {
        temp.map(|t| t.clamp(0.0, 1.0))
    }

    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let client = Client::new();
        let is_stream = self.options.streaming_func.is_some();

        let payload = self.build_payload(messages, is_stream);
        let res = client
            .post(&format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        let status = res.status().as_u16();

        let res = match status {
            400 => Err(LLMError::MiniMaxError(MiniMaxError::InvalidFormatError(
                "Invalid request format".to_string(),
            ))),
            401 => Err(LLMError::MiniMaxError(MiniMaxError::AuthenticationError(
                "Invalid API Key".to_string(),
            ))),
            402 => Err(LLMError::MiniMaxError(
                MiniMaxError::InsufficientBalanceError("Insufficient balance".to_string()),
            )),
            422 => Err(LLMError::MiniMaxError(
                MiniMaxError::InvalidParametersError("Invalid parameters".to_string()),
            )),
            429 => Err(LLMError::MiniMaxError(MiniMaxError::RateLimitError(
                "Rate limit reached".to_string(),
            ))),
            500 => Err(LLMError::MiniMaxError(MiniMaxError::ServerError(
                "Server error".to_string(),
            ))),
            503 => Err(LLMError::MiniMaxError(
                MiniMaxError::ServerOverloadedError("Server overloaded".to_string()),
            )),
            _ => Ok(res.json::<ApiResponse>().await?),
        }?;

        let choice = res.choices.first();

        let generation = choice
            .map(|c| Self::strip_think_tags(&c.message.content))
            .unwrap_or_default();

        let tokens = Some(TokenUsage {
            prompt_tokens: res.usage.prompt_tokens,
            completion_tokens: res.usage.completion_tokens,
            total_tokens: res.usage.total_tokens,
        });

        Ok(GenerateResult { tokens, generation })
    }

    fn build_payload(&self, messages: &[Message], stream: bool) -> Payload {
        let mut response_format = None;
        if self.json_mode {
            response_format = Some(ResponseFormat {
                format_type: "json_object".to_string(),
            });
        }

        let mut payload = Payload {
            model: self.model.clone(),
            messages: messages
                .iter()
                .map(MiniMaxMessage::from_message)
                .collect::<Vec<_>>(),
            max_tokens: self.options.max_tokens,
            stream: None,
            temperature: Self::clamp_temperature(self.options.temperature),
            top_p: self.options.top_p,
            frequency_penalty: None,
            presence_penalty: None,
            stop: self.options.stop_words.clone(),
            response_format,
        };

        if stream {
            payload.stream = Some(true);
        }

        if let Some(fp) = self.options.frequency_penalty {
            if fp >= -2.0 && fp <= 2.0 {
                payload.frequency_penalty = Some(fp);
            }
        }

        if let Some(pp) = self.options.presence_penalty {
            if pp >= -2.0 && pp <= 2.0 {
                payload.presence_penalty = Some(pp);
            }
        }

        payload
    }

    /// Strip `<think>...</think>` tags from MiniMax M2.5+ reasoning output.
    fn strip_think_tags(content: &str) -> String {
        let mut result = content.to_string();
        while let Some(start) = result.find("<think>") {
            if let Some(end) = result.find("</think>") {
                result = format!("{}{}", &result[..start], &result[end + 8..]);
            } else {
                break;
            }
        }
        result.trim().to_string()
    }

    fn parse_sse_chunk(chunk: &[u8]) -> Result<Vec<Value>, LLMError> {
        let text = str::from_utf8(chunk).map_err(|e| LLMError::ParsingError(e.to_string()))?;
        let mut values = Vec::new();

        for line in text.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..];
                if data == "[DONE]" {
                    continue;
                }
                let value: Value = serde_json::from_str(data).map_err(|e| {
                    LLMError::ParsingError(format!("Failed to parse SSE data: {}", e))
                })?;
                values.push(value);
            }
        }

        Ok(values)
    }
}

#[async_trait]
impl LLM for MiniMax {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        match &self.options.streaming_func {
            Some(func) => {
                let mut complete_response = String::new();
                let mut stream = self.stream(messages).await?;
                while let Some(data) = stream.next().await {
                    match data {
                        Ok(value) => {
                            let mut func = func.lock().await;
                            complete_response.push_str(&value.content);
                            let _ = func(value.content).await;
                        }
                        Err(e) => return Err(e),
                    }
                }
                let mut generate_result = GenerateResult::default();
                generate_result.generation = complete_response;
                Ok(generate_result)
            }
            None => self.generate(messages).await,
        }
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let client = Client::new();
        let payload = self.build_payload(messages, true);
        let request = client
            .post(&format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .build()?;

        let stream = client.execute(request).await?;
        let stream = stream.bytes_stream();

        let processed_stream = stream
            .then(move |result| async move {
                match result {
                    Ok(bytes) => {
                        let chunks = Self::parse_sse_chunk(&bytes)?;

                        for chunk in chunks {
                            if let Some(choices) =
                                chunk.get("choices").and_then(|c| c.as_array())
                            {
                                if let Some(choice) = choices.first() {
                                    if let Some(delta) = choice.get("delta") {
                                        if let Some(content) =
                                            delta.get("content").and_then(|c| c.as_str())
                                        {
                                            if !content.is_empty() {
                                                let usage =
                                                    if let Some(usage) = chunk.get("usage") {
                                                        Some(TokenUsage {
                                                            prompt_tokens: usage
                                                                .get("prompt_tokens")
                                                                .and_then(|t| t.as_u64())
                                                                .unwrap_or(0)
                                                                as u32,
                                                            completion_tokens: usage
                                                                .get("completion_tokens")
                                                                .and_then(|t| t.as_u64())
                                                                .unwrap_or(0)
                                                                as u32,
                                                            total_tokens: usage
                                                                .get("total_tokens")
                                                                .and_then(|t| t.as_u64())
                                                                .unwrap_or(0)
                                                                as u32,
                                                        })
                                                    } else {
                                                        None
                                                    };

                                                return Ok(StreamData::new(
                                                    chunk.clone(),
                                                    usage,
                                                    content,
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        Ok(StreamData::new(Value::Null, None, ""))
                    }
                    Err(e) => Err(LLMError::OtherError(e.to_string())),
                }
            })
            .filter_map(|result| async move {
                match result {
                    Ok(data) if !data.content.is_empty() => Some(Ok(data)),
                    Ok(_) => None,
                    Err(e) => Some(Err(e)),
                }
            });

        Ok(Box::pin(processed_stream))
    }

    fn add_options(&mut self, options: CallOptions) {
        self.options = options;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas::{Message, MessageType};

    #[test]
    fn test_minimax_model_names() {
        assert_eq!(MiniMaxModel::MiniMaxM27.to_string(), "MiniMax-M2.7");
        assert_eq!(
            MiniMaxModel::MiniMaxM27Highspeed.to_string(),
            "MiniMax-M2.7-highspeed"
        );
        assert_eq!(MiniMaxModel::MiniMaxM25.to_string(), "MiniMax-M2.5");
        assert_eq!(
            MiniMaxModel::MiniMaxM25Highspeed.to_string(),
            "MiniMax-M2.5-highspeed"
        );
    }

    #[test]
    fn test_minimax_default() {
        let client = MiniMax::new();
        assert_eq!(client.model, "MiniMax-M2.7");
        assert_eq!(client.base_url, "https://api.minimax.io");
        assert!(!client.json_mode);
    }

    #[test]
    fn test_minimax_builder() {
        let client = MiniMax::new()
            .with_model("MiniMax-M2.5")
            .with_api_key("test-key")
            .with_base_url("https://custom.api.com")
            .with_json_mode(true);

        assert_eq!(client.model, "MiniMax-M2.5");
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.base_url, "https://custom.api.com");
        assert!(client.json_mode);
    }

    #[test]
    fn test_minimax_with_options() {
        let options = CallOptions::default()
            .with_max_tokens(1000)
            .with_temperature(0.7);
        let client = MiniMax::new().with_options(options);
        assert_eq!(client.options.max_tokens, Some(1000));
        assert_eq!(client.options.temperature, Some(0.7));
    }

    #[test]
    fn test_clamp_temperature() {
        assert_eq!(MiniMax::clamp_temperature(Some(0.5)), Some(0.5));
        assert_eq!(MiniMax::clamp_temperature(Some(1.5)), Some(1.0));
        assert_eq!(MiniMax::clamp_temperature(Some(-0.5)), Some(0.0));
        assert_eq!(MiniMax::clamp_temperature(None), None);
    }

    #[test]
    fn test_strip_think_tags() {
        assert_eq!(
            MiniMax::strip_think_tags("<think>reasoning</think>Answer here"),
            "Answer here"
        );
        assert_eq!(MiniMax::strip_think_tags("No tags here"), "No tags here");
        assert_eq!(
            MiniMax::strip_think_tags("<think>step 1</think>Part 1<think>step 2</think>Part 2"),
            "Part 1Part 2"
        );
    }

    #[test]
    fn test_minimax_message_from_message() {
        let msg = Message {
            content: "Hello".to_string(),
            message_type: MessageType::HumanMessage,
            id: None,
            images: None,
            tool_calls: None,
        };
        let mm_msg = MiniMaxMessage::from_message(&msg);
        assert_eq!(mm_msg.role, "user");
        assert_eq!(mm_msg.content, "Hello");

        let msg = Message {
            content: "System prompt".to_string(),
            message_type: MessageType::SystemMessage,
            id: None,
            images: None,
            tool_calls: None,
        };
        let mm_msg = MiniMaxMessage::from_message(&msg);
        assert_eq!(mm_msg.role, "system");

        let msg = Message {
            content: "AI response".to_string(),
            message_type: MessageType::AIMessage,
            id: None,
            images: None,
            tool_calls: None,
        };
        let mm_msg = MiniMaxMessage::from_message(&msg);
        assert_eq!(mm_msg.role, "assistant");

        let msg = Message {
            content: "Tool output".to_string(),
            message_type: MessageType::ToolMessage,
            id: Some("tool-1".to_string()),
            images: None,
            tool_calls: None,
        };
        let mm_msg = MiniMaxMessage::from_message(&msg);
        assert_eq!(mm_msg.role, "tool");
    }

    #[test]
    fn test_build_payload_basic() {
        let client = MiniMax::new().with_model("MiniMax-M2.7");
        let messages = vec![Message {
            content: "Hello".to_string(),
            message_type: MessageType::HumanMessage,
            id: None,
            images: None,
            tool_calls: None,
        }];
        let payload = client.build_payload(&messages, false);
        assert_eq!(payload.model, "MiniMax-M2.7");
        assert_eq!(payload.messages.len(), 1);
        assert!(payload.stream.is_none());
        assert!(payload.response_format.is_none());
    }

    #[test]
    fn test_build_payload_stream() {
        let client = MiniMax::new();
        let messages = vec![Message {
            content: "Hello".to_string(),
            message_type: MessageType::HumanMessage,
            id: None,
            images: None,
            tool_calls: None,
        }];
        let payload = client.build_payload(&messages, true);
        assert_eq!(payload.stream, Some(true));
    }

    #[test]
    fn test_build_payload_json_mode() {
        let client = MiniMax::new().with_json_mode(true);
        let messages = vec![Message {
            content: "Hello".to_string(),
            message_type: MessageType::HumanMessage,
            id: None,
            images: None,
            tool_calls: None,
        }];
        let payload = client.build_payload(&messages, false);
        assert!(payload.response_format.is_some());
        assert_eq!(
            payload.response_format.unwrap().format_type,
            "json_object"
        );
    }

    #[test]
    fn test_build_payload_temperature_clamping() {
        let options = CallOptions::default().with_temperature(1.5);
        let client = MiniMax::new().with_options(options);
        let messages = vec![Message {
            content: "Hello".to_string(),
            message_type: MessageType::HumanMessage,
            id: None,
            images: None,
            tool_calls: None,
        }];
        let payload = client.build_payload(&messages, false);
        assert_eq!(payload.temperature, Some(1.0));
    }

    #[test]
    fn test_build_payload_with_stop_words() {
        let options = CallOptions::default()
            .with_stop_words(vec!["stop1".to_string(), "stop2".to_string()]);
        let client = MiniMax::new().with_options(options);
        let messages = vec![Message {
            content: "Hello".to_string(),
            message_type: MessageType::HumanMessage,
            id: None,
            images: None,
            tool_calls: None,
        }];
        let payload = client.build_payload(&messages, false);
        assert_eq!(
            payload.stop,
            Some(vec!["stop1".to_string(), "stop2".to_string()])
        );
    }

    #[test]
    fn test_build_payload_with_penalties() {
        let options = CallOptions::default()
            .with_frequency_penalty(0.5)
            .with_presence_penalty(-0.5);
        let client = MiniMax::new().with_options(options);
        let messages = vec![Message {
            content: "Hello".to_string(),
            message_type: MessageType::HumanMessage,
            id: None,
            images: None,
            tool_calls: None,
        }];
        let payload = client.build_payload(&messages, false);
        assert_eq!(payload.frequency_penalty, Some(0.5));
        assert_eq!(payload.presence_penalty, Some(-0.5));
    }

    #[test]
    fn test_parse_sse_chunk_done() {
        let chunk = b"data: [DONE]\n";
        let values = MiniMax::parse_sse_chunk(chunk).unwrap();
        assert!(values.is_empty());
    }

    #[test]
    fn test_parse_sse_chunk_valid() {
        let chunk = b"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n";
        let values = MiniMax::parse_sse_chunk(chunk).unwrap();
        assert_eq!(values.len(), 1);
    }

    #[test]
    fn test_parse_sse_chunk_empty_lines() {
        let chunk = b"\n\ndata: {\"test\":true}\n\n";
        let values = MiniMax::parse_sse_chunk(chunk).unwrap();
        assert_eq!(values.len(), 1);
    }

    #[test]
    fn test_minimax_model_enum() {
        let model = MiniMaxModel::MiniMaxM27;
        let client = MiniMax::new().with_model(model.to_string());
        assert_eq!(client.model, "MiniMax-M2.7");
    }

    #[tokio::test]
    #[ignore]
    async fn test_minimax_generate() {
        let messages = vec![Message {
            content: "Hello".to_string(),
            message_type: MessageType::HumanMessage,
            id: Some("test_id".to_string()),
            images: None,
            tool_calls: None,
        }];

        let client = MiniMax::new();
        let res = client.generate(&messages).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    #[ignore]
    async fn test_minimax_stream() {
        let messages = vec![Message {
            content: "Hello".to_string(),
            message_type: MessageType::HumanMessage,
            id: Some("test_id".to_string()),
            images: None,
            tool_calls: None,
        }];

        let client = MiniMax::new();
        let res = client.stream(&messages).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    #[ignore]
    async fn test_minimax_m25_highspeed() {
        let messages = vec![Message {
            content: "What is 2+2?".to_string(),
            message_type: MessageType::HumanMessage,
            id: Some("test_id".to_string()),
            images: None,
            tool_calls: None,
        }];

        let client = MiniMax::new()
            .with_model(MiniMaxModel::MiniMaxM25Highspeed.to_string());

        let res = client.generate(&messages).await;
        assert!(res.is_ok());
    }
}
