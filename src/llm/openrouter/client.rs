
//! OpenRouter LLM client.
//!
//! Implements the LLM trait for OpenRouter API integration.

use super::models::OpenRouterModel;
use crate::language_models::llm::LLM;
use crate::language_models::options::CallOptions;
use crate::language_models::{GenerateResult, LLMError};
use crate::schemas::{Message, MessageType, StreamData};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

/// OpenRouter LLM client.
#[derive(Clone)]
pub struct OpenRouter {
    /// API key for authentication.
    api_key: String,
    /// Model to use.
    model: OpenRouterModel,
}

impl OpenRouter {
    /// Creates a new OpenRouter client with the given API key and model.
    pub fn new<S: Into<String>>(api_key: S, model: OpenRouterModel) -> Self {
        Self {
            api_key: api_key.into(),
            model,
        }
    }
}

#[derive(Serialize, Debug)]
struct OpenRouterMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize, Debug)]
struct OpenRouterCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Deserialize, Debug)]
struct AssistantMessage {
    role: String,
    content: String,
}

fn message_type_to_role(mt: &MessageType) -> &'static str {
    match mt {
        MessageType::SystemMessage => "system",
        MessageType::HumanMessage => "user",
        MessageType::AIMessage => "assistant",
        // OpenRouter does not support tool messages, default to "user"
        MessageType::ToolMessage => "user",
    }
}

fn map_messages(messages: &[Message]) -> Vec<OpenRouterMessage> {
    messages
        .iter()
        .map(|m| OpenRouterMessage {
            role: message_type_to_role(&m.message_type),
            content: &m.content,
        })
        .collect()
}

#[async_trait::async_trait]
impl LLM for OpenRouter {
    /// Generates a completion using OpenRouter API (non-streaming).
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        // Build API request body
        let mapped_messages = map_messages(messages);
        let mut body = json!({
            "model": self.model.as_str(),
            "messages": mapped_messages,
        });

        let client = Client::new();
        let resp = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LLMError::OtherError(format!("OpenRouter request failed: {}", e)))?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| LLMError::OtherError(format!("OpenRouter request failed: {}", e)))?;

        if !status.is_success() {
            return Err(LLMError::OtherError(format!("OpenRouter API returned error: HTTP {}: {}", status, text)));
        }

        let resp_json: OpenRouterCompletionResponse = serde_json::from_str(&text)
            .map_err(|e| LLMError::OtherError(format!("OpenRouter: Invalid JSON: {}", e)))?;

        let reply = resp_json
            .choices
            .get(0)
            .map(|c| c.message.content.clone())
            .ok_or_else(|| LLMError::OtherError("OpenRouter: No assistant reply found".to_string()))?;

        Ok(GenerateResult {
            generation: reply,
            tokens: None,
        })
    }

    /// Streams a completion using OpenRouter API (SSE).
    ///
    /// This method POSTs to the OpenRouter endpoint with "stream": true in the body,
    /// parses the SSE response, and yields content chunks as StreamData.
    ///
    /// # Arguments
    /// - messages: The conversation history to send.
    ///
    /// # Returns
    /// A Stream yielding StreamData for each incoming content chunk.
    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let mapped_messages = map_messages(messages);
        let body = json!({
            "model": self.model.as_str(),
            "messages": mapped_messages,
            "stream": true,
        });

        let client = Client::new();
        let req = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body);

        // Send the request and get the streaming response
        let resp = req.send().await.map_err(|e| {
            LLMError::OtherError(format!("OpenRouter request failed: {}", e))
        })?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_else(|_| "".to_string());
            return Err(LLMError::OtherError(format!("OpenRouter API returned error: HTTP {}: {}", status, text)));
        }

        let stream = resp.bytes_stream();

        // If a streaming_func callback is provided in CallOptions, we need to use it.
        // But the LLM trait stream() signature does not take options, so for now we cannot support it directly.
        // If needed, adapt trait to take CallOptions.

        // We'll parse the SSE lines and yield StreamData for each assistant content delta.
        let s = sse_stream_to_streamdata(stream);

        Ok(Box::pin(s))
    }

    fn add_options(&mut self, _options: CallOptions) {
        // Stub implementation.
    }
}

/// Parses the Server-Sent Events (SSE) byte stream into a stream of StreamData.
/// Only yields assistant content chunks (delta).
fn sse_stream_to_streamdata<S>(mut stream: S) -> impl Stream<Item = Result<StreamData, LLMError>> + Send
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin + Send + 'static,
{
    use futures::stream;
    use serde_json::Value;
    use crate::schemas::StreamData;

    // We'll buffer incoming bytes and split by newline.
    let mut buffer = Vec::new();

    stream::unfold((stream, buffer), |(mut stream, mut buffer)| async move {
        loop {
            match stream.next().await {
                Some(Ok(chunk)) => {
                    buffer.extend_from_slice(&chunk);
                    // Process lines
                    while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                        let mut line = buffer.drain(..=pos).collect::<Vec<u8>>();
                        // Remove trailing newline
                        if let Some(b'\n') = line.last() {
                            line.pop();
                        }
                        let line_str = String::from_utf8_lossy(&line).trim().to_string();
                        if line_str.is_empty() { continue; }
                        if line_str.starts_with("data: ") {
                            let data = &line_str[6..];
                            if data == "[DONE]" {
                                return None;
                            }
                            // Parse JSON
                            let v: Value = match serde_json::from_str(data) {
                                Ok(v) => v,
                                Err(e) => {
                                    return Some((Err(LLMError::OtherError(format!("OpenRouter SSE JSON error: {}", e))), (stream, buffer)));
                                }
                            };
                            // Try to extract assistant delta content
                            // OpenRouter/ChatCompletionChunk: { "choices": [{ "delta": { "content": ... } }] }
                            let content = v.get("choices")
                                .and_then(|choices| choices.get(0))
                                .and_then(|c| c.get("delta"))
                                .and_then(|d| d.get("content"))
                                .and_then(|c| c.as_str())
                                .unwrap_or("");
                            if !content.is_empty() {
                                let stream_data = StreamData {
                                    value: v.clone(),
                                    tokens: None,
                                    content: content.to_string(),
                                };
                                return Some((Ok(stream_data), (stream, buffer)));
                            }
                        }
                    }
                }
                Some(Err(e)) => {
                    return Some((Err(LLMError::OtherError(format!("OpenRouter SSE stream error: {}", e))), (stream, buffer)));
                }
                None => {
                    // End of stream
                    return None;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas::{Message, MessageType};

    #[tokio::test]
    async fn test_openrouter_message_mapping() {
        let m1 = Message {
            content: "You are a helpful assistant.".to_string(),
            message_type: MessageType::SystemMessage,
            id: None,
            tool_calls: None,
            images: None,
        };
        let m2 = Message {
            content: "Hello!".to_string(),
            message_type: MessageType::HumanMessage,
            id: None,
            tool_calls: None,
            images: None,
        };
        let binding = [m1, m2];
        let mapped = super::map_messages(&binding);
        assert_eq!(mapped[0].role, "system");
        assert_eq!(mapped[1].role, "user");
    }

    /// Integration test for OpenRouter streaming.
    ///
    /// This test is ignored by default because it requires a real API key and network.
    #[tokio::test]
    #[ignore]
    async fn test_openrouter_stream_real_network() {
        // This test will only work if you set a valid API key and model.
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
        let model = OpenRouterModel::Gpt4o;
        let client = OpenRouter::new(api_key, model);
        let messages = vec![
            Message {
                content: "You are a helpful assistant.".to_string(),
                message_type: MessageType::SystemMessage,
                id: None,
                tool_calls: None,
                images: None,
            },
            Message {
                content: "Hello!".to_string(),
                message_type: MessageType::HumanMessage,
                id: None,
                tool_calls: None,
                images: None,
            },
        ];
        let mut stream = client.stream(&messages).await.expect("Failed to start stream");
        let mut got_content = false;
        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("Stream error");
            if !chunk.content.is_empty() {
                got_content = true;
            }
        }
        assert!(got_content, "Did not receive any streamed content");
    }
}
