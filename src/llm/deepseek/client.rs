use crate::{
    language_models::{llm::LLM, options::CallOptions, GenerateResult, LLMError, TokenUsage},
    llm::DeepseekError,
    schemas::{Message, StreamData},
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde_json::Value;
use std::{pin::Pin, str};

use super::models::{ApiResponse, DeepseekMessage, Payload, ResponseFormat};

pub enum DeepseekModel {
    DeepseekChat,
    DeepseekReasoner,
}

impl ToString for DeepseekModel {
    fn to_string(&self) -> String {
        match self {
            DeepseekModel::DeepseekChat => "deepseek-chat".to_string(),
            DeepseekModel::DeepseekReasoner => "deepseek-reasoner".to_string(),
        }
    }
}

#[derive(Clone)]
pub struct Deepseek {
    model: String,
    options: CallOptions,
    api_key: String,
    base_url: String,
    json_mode: bool,
    include_reasoning: bool,
}

impl Default for Deepseek {
    fn default() -> Self {
        Self::new()
    }
}

impl Deepseek {
    pub fn new() -> Self {
        Self {
            model: DeepseekModel::DeepseekChat.to_string(),
            options: CallOptions::default(),
            api_key: std::env::var("DEEPSEEK_API_KEY").unwrap_or_default(),
            base_url: "https://api.deepseek.com".to_string(),
            json_mode: false,
            include_reasoning: false,
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

    pub fn with_include_reasoning(mut self, include_reasoning: bool) -> Self {
        self.include_reasoning = include_reasoning;
        self
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
            400 => Err(LLMError::DeepseekError(DeepseekError::InvalidFormatError(
                "Invalid request format".to_string(),
            ))),
            401 => Err(LLMError::DeepseekError(DeepseekError::AuthenticationError(
                "Invalid API Key".to_string(),
            ))),
            402 => Err(LLMError::DeepseekError(
                DeepseekError::InsufficientBalanceError("Insufficient balance".to_string()),
            )),
            422 => Err(LLMError::DeepseekError(
                DeepseekError::InvalidParametersError("Invalid parameters".to_string()),
            )),
            429 => Err(LLMError::DeepseekError(DeepseekError::RateLimitError(
                "Rate limit reached".to_string(),
            ))),
            500 => Err(LLMError::DeepseekError(DeepseekError::ServerError(
                "Server error".to_string(),
            ))),
            503 => Err(LLMError::DeepseekError(
                DeepseekError::ServerOverloadedError("Server overloaded".to_string()),
            )),
            _ => Ok(res.json::<ApiResponse>().await?),
        }?;

        let choice = res.choices.first();

        let mut generation = choice
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        // If include_reasoning is enabled and the model is deepseek-reasoner,
        // append the reasoning content to the generation if available
        if self.include_reasoning && self.model == DeepseekModel::DeepseekReasoner.to_string() {
            if let Some(reasoning) = choice.and_then(|c| c.message.reasoning_content.clone()) {
                generation = format!("Reasoning:\n{}\n\nAnswer:\n{}", reasoning, generation);
            }
        }

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
                .map(DeepseekMessage::from_message)
                .collect::<Vec<_>>(),
            max_tokens: self.options.max_tokens,
            stream: None,
            temperature: self.options.temperature,
            top_p: self.options.top_p,
            frequency_penalty: None,
            presence_penalty: None,
            stop: self.options.stop_words.clone(),
            response_format,
        };

        if stream {
            payload.stream = Some(true);
        }

        // Apply frequency_penalty if it's in the options range
        if let Some(fp) = self.options.frequency_penalty {
            if fp >= -2.0 && fp <= 2.0 {
                payload.frequency_penalty = Some(fp);
            }
        }

        // Apply presence_penalty if it's in the options range
        if let Some(pp) = self.options.presence_penalty {
            if pp >= -2.0 && pp <= 2.0 {
                payload.presence_penalty = Some(pp);
            }
        }

        payload
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
impl LLM for Deepseek {
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

        let include_reasoning = self.include_reasoning;
        let is_reasoner = self.model == DeepseekModel::DeepseekReasoner.to_string();

        let processed_stream = stream
            .then(move |result| {
                async move {
                    match result {
                        Ok(bytes) => {
                            let chunks = Self::parse_sse_chunk(&bytes)?;

                            for chunk in chunks {
                                if let Some(choices) =
                                    chunk.get("choices").and_then(|c| c.as_array())
                                {
                                    if let Some(choice) = choices.first() {
                                        if let Some(delta) = choice.get("delta") {
                                            // Handle reasoning_content if it exists
                                            if include_reasoning && is_reasoner {
                                                if let Some(reasoning) = delta
                                                    .get("reasoning_content")
                                                    .and_then(|c| c.as_str())
                                                {
                                                    if !reasoning.is_empty() {
                                                        let usage = if let Some(usage) =
                                                            chunk.get("usage")
                                                        {
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
                                                            format!("Reasoning: {}", reasoning),
                                                        ));
                                                    }
                                                }
                                            }

                                            // Handle content as before
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

                            // If we didn't return within the loop, return an empty stream data
                            Ok(StreamData::new(Value::Null, None, ""))
                        }
                        Err(e) => Err(LLMError::OtherError(e.to_string())),
                    }
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

    #[tokio::test]
    #[ignore]
    async fn test_deepseek_generate() {
        let messages = vec![Message {
            content: "Hello".to_string(),
            message_type: MessageType::HumanMessage,
            id: Some("test_id".to_string()),
            images: None,
            tool_calls: None,
        }];

        let client = Deepseek::new();
        let res = client.generate(&messages).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    #[ignore]
    async fn test_deepseek_stream() {
        let messages = vec![Message {
            content: "Hello".to_string(),
            message_type: MessageType::HumanMessage,
            id: Some("test_id".to_string()),
            images: None,
            tool_calls: None,
        }];

        let client = Deepseek::new();
        let res = client.stream(&messages).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    #[ignore]
    async fn test_deepseek_reasoner() {
        let messages = vec![Message {
            content: "9.11 and 9.8, which is greater?".to_string(),
            message_type: MessageType::HumanMessage,
            id: Some("test_id".to_string()),
            images: None,
            tool_calls: None,
        }];

        // Create a client with the DeepseekReasoner model and enable reasoning content
        let client = Deepseek::new()
            .with_model(DeepseekModel::DeepseekReasoner.to_string())
            .with_include_reasoning(true);

        let res = client.generate(&messages).await;
        assert!(res.is_ok());

        // The response will contain both the reasoning and answer content
        if let Ok(result) = res {
            println!("Generation result: {}", result.generation);
        }
    }
}
