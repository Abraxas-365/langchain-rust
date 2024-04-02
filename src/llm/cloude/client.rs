use crate::{
    language_models::{llm::LLM, options::CallOptions, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, StreamData},
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde_json::{json, Value};
use std::{collections::HashMap, pin::Pin};

use super::models::{ApiResponse, CloudeMessage, Payload};

pub enum CloudeModel {
    Claude3pus20240229,
    Claude3sonnet20240229,
    Claude3haiku20240307,
}

impl ToString for CloudeModel {
    fn to_string(&self) -> String {
        match self {
            CloudeModel::Claude3pus20240229 => "claude-3-opus-20240229".to_string(),
            CloudeModel::Claude3sonnet20240229 => "claude-3-sonnet-20240229".to_string(),
            CloudeModel::Claude3haiku20240307 => "claude-3-haiku-20240307".to_string(),
        }
    }
}

pub struct Cloude {
    model: String,
    options: CallOptions,
    api_key: String,
    anthropic_version: String,
}

impl Default for Cloude {
    fn default() -> Self {
        Self::new()
    }
}

impl Cloude {
    pub fn new() -> Self {
        Self {
            model: CloudeModel::Claude3pus20240229.to_string(),
            options: CallOptions::default(),
            api_key: std::env::var("CLOUDE_API_KEY").unwrap_or_default(),
            anthropic_version: "2023-06-01".to_string(),
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

    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let client = Client::new();
        let payload = Payload {
            model: self.model.clone(),
            messages: messages
                .iter()
                .map(|m| CloudeMessage::from_message(m))
                .collect::<Vec<_>>(),
            max_tokens: self.options.max_tokens.unwrap_or(1024),
            stream: None,
        };

        let res = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", self.anthropic_version.clone())
            .header("content-type", "application/json")
            .json(&payload)
            .send()
            .await?
            .json::<ApiResponse>()
            .await?;

        let generation = res
            .content
            .get(0)
            .map(|c| c.text.clone())
            .unwrap_or_default();

        let tokens = Some(TokenUsage {
            prompt_tokens: res.usage.input_tokens,
            completion_tokens: res.usage.output_tokens,
            total_tokens: res.usage.input_tokens + res.usage.output_tokens,
        });

        Ok(GenerateResult { tokens, generation })
    }
}

#[async_trait]
impl LLM for Cloude {
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
        let payload = Payload {
            model: self.model.clone(),
            messages: messages
                .iter()
                .map(|m| CloudeMessage::from_message(m))
                .collect::<Vec<_>>(),
            max_tokens: self.options.max_tokens.unwrap_or(1024),
            stream: Some(true),
        };

        let request = client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.anthropic_version)
            .header("content-type", "application/json")
            .json(&payload)
            .build()?;

        // Instead of sending the request directly, return a stream wrapper
        let stream = client.execute(request).await?.bytes_stream();

        // Process each chunk as it arrives
        let processed_stream = stream.then(move |result| {
            async move {
                match result {
                    Ok(bytes) => {
                        let value: Value = parse_sse_to_json(&String::from_utf8_lossy(&bytes))
                            .unwrap_or_else(|_| {
                                log::error!(
                                    "Failed to parse SSE data to JSON {}",
                                    String::from_utf8_lossy(&bytes)
                                );
                                json!({})
                            });
                        if value["type"].as_str().unwrap_or("") == "content_block_delta" {
                            let content = value["delta"]["text"].clone();
                            // Return StreamData based on the parsed content
                            Ok(StreamData::new(value, content.as_str().unwrap_or("")))
                        } else {
                            Ok(StreamData::new(value, ""))
                        }
                    }
                    Err(e) => Err(LLMError::RequestError(e)),
                }
            }
        });

        Ok(Box::pin(processed_stream))
    }

    fn add_options(&mut self, options: CallOptions) {
        self.options.merge_options(options)
    }
}

fn parse_sse_to_json(sse_data: &str) -> Result<Value, serde_json::Error> {
    let lines: Vec<&str> = sse_data.trim().split('\n').collect();
    let mut event_data: HashMap<&str, String> = HashMap::new();

    for line in lines {
        if let Some((key, value)) = line.split_once(": ") {
            event_data.insert(key, value.to_string());
        }
    }

    if let Some(data) = event_data.get("data") {
        serde_json::from_str(data)
    } else {
        log::error!("No data field in the SSE event");
        Ok(json!({}))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    #[ignore]
    async fn test_cloudia_generate() {
        let cloudia = Cloude::new();

        let a = cloudia
            .generate(&[Message::new_human_message("Hi, how are you doing")])
            .await
            .unwrap();

        println!("{:?}", a)
    }

    #[test]
    #[ignore]
    async fn test_cloudia_stream() {
        let cloudia = Cloude::new();
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
