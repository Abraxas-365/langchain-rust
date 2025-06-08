use crate::{
    language_models::{llm::LLM, options::CallOptions, GenerateResult, LLMError, TokenUsage},
    llm::NebiusError,
    schemas::{Message, StreamData},
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde_json::Value;
use std::{collections::HashMap, pin::Pin};

use super::models::{NebiusMessage, NebiusPayload, NebiusResponse};

pub enum NebiusModel {
    DeepSeekR1_0528,
    Qwen3_235B_A22B,
    Qwen3_30B_A3B,
    Qwen3_32B,
    Qwen3_14B,
    Qwen3_4B_Fast,
    LlamaNemotronUltra253B,
    DeepSeekV3_0324,
    DeepSeekV3,
    DeepSeekR1,
    Llama3_3_70B,
    Llama3_1_70B,
    Llama3_1_8B,
    Llama3_1_405B,
    MistralNemo,
    Qwen2_5_Coder_7B,
    Qwen2_5_Coder_32B,
    Gemma2_2B,
    Gemma2_9B_Fast,
    Qwen2_5_32B,
    Qwen2_5_72B,
    LlamaOpenBioLLM70B,
    QwQ32B,
    Phi4,
    HermesLlama405B,
    DeepSeekR1DistillLlama70B,
    LlamaNemotronSuper49B,
}

impl ToString for NebiusModel {
    fn to_string(&self) -> String {
        match self {
            NebiusModel::DeepSeekR1_0528 => "deepseek-ai/DeepSeek-R1-0528".to_string(),
            NebiusModel::Qwen3_235B_A22B => "Qwen/Qwen3-235B-A22B".to_string(),
            NebiusModel::Qwen3_30B_A3B => "Qwen/Qwen3-30B-A3B".to_string(),
            NebiusModel::Qwen3_32B => "Qwen/Qwen3-32B".to_string(),
            NebiusModel::Qwen3_14B => "Qwen/Qwen3-14B".to_string(),
            NebiusModel::Qwen3_4B_Fast => "Qwen/Qwen3-4B-fast".to_string(),
            NebiusModel::LlamaNemotronUltra253B => {
                "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1".to_string()
            }
            NebiusModel::DeepSeekV3_0324 => "deepseek-ai/DeepSeek-V3-0324".to_string(),
            NebiusModel::DeepSeekV3 => "deepseek-ai/DeepSeek-V3".to_string(),
            NebiusModel::DeepSeekR1 => "deepseek-ai/DeepSeek-R1".to_string(),
            NebiusModel::Llama3_3_70B => "meta-llama/Llama-3.3-70B-Instruct".to_string(),
            NebiusModel::Llama3_1_70B => "meta-llama/Meta-Llama-3.1-70B-Instruct".to_string(),
            NebiusModel::Llama3_1_8B => "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string(),
            NebiusModel::Llama3_1_405B => "meta-llama/Meta-Llama-3.1-405B-Instruct".to_string(),
            NebiusModel::MistralNemo => "mistralai/Mistral-Nemo-Instruct-2407".to_string(),
            NebiusModel::Qwen2_5_Coder_7B => "Qwen/Qwen2.5-Coder-7B".to_string(),
            NebiusModel::Qwen2_5_Coder_32B => "Qwen/Qwen2.5-Coder-32B-Instruct".to_string(),
            NebiusModel::Gemma2_2B => "google/gemma-2-2b-it".to_string(),
            NebiusModel::Gemma2_9B_Fast => "google/gemma-2-9b-it-fast".to_string(),
            NebiusModel::Qwen2_5_32B => "Qwen/Qwen2.5-32B-Instruct".to_string(),
            NebiusModel::Qwen2_5_72B => "Qwen/Qwen2.5-72B-Instruct".to_string(),
            NebiusModel::LlamaOpenBioLLM70B => "aaditya/Llama3-OpenBioLLM-70B".to_string(),
            NebiusModel::QwQ32B => "Qwen/QwQ-32B".to_string(),
            NebiusModel::Phi4 => "microsoft/phi-4".to_string(),
            NebiusModel::HermesLlama405B => "NousResearch/Hermes-3-Llama-405B".to_string(),
            NebiusModel::DeepSeekR1DistillLlama70B => {
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B".to_string()
            }
            NebiusModel::LlamaNemotronSuper49B => {
                "nvidia/Llama-3_3-Nemotron-Super-49B-v1".to_string()
            }
        }
    }
}

impl Into<String> for NebiusModel {
    fn into(self) -> String {
        self.to_string()
    }
}

#[derive(Clone)]
pub struct Nebius {
    model: String,
    options: CallOptions,
    api_key: String,
}

impl Default for Nebius {
    fn default() -> Self {
        Self::new()
    }
}

impl Nebius {
    pub fn new() -> Self {
        Self {
            model: NebiusModel::DeepSeekV3.to_string(),
            options: CallOptions::default(),
            api_key: std::env::var("NEBIUS_API_KEY").unwrap_or_default(),
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

    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let client = Client::new();
        let is_stream = self.options.streaming_func.is_some();

        let payload = self.build_payload(messages, is_stream);
        let res = client
            .post("https://api.studio.nebius.ai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        let res = match res.status().as_u16() {
            401 => Err(LLMError::NebiusError(NebiusError::AuthenticationError(
                "Invalid API Key".to_string(),
            ))),
            403 => Err(LLMError::NebiusError(NebiusError::PermissionError(
                "Permission Denied".to_string(),
            ))),
            404 => Err(LLMError::NebiusError(NebiusError::NotFoundError(
                "Not Found".to_string(),
            ))),
            429 => Err(LLMError::NebiusError(NebiusError::RateLimitError(
                "Rate Limit Exceeded".to_string(),
            ))),
            503 => Err(LLMError::NebiusError(NebiusError::ServiceUnavailableError(
                "Service Unavailable".to_string(),
            ))),
            _ => {
                let response = res.json::<NebiusResponse>().await?;
                Ok(response)
            }
        }?;

        let generation = res
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let tokens = Some(TokenUsage {
            prompt_tokens: res.usage.prompt_tokens,
            completion_tokens: res.usage.completion_tokens,
            total_tokens: res.usage.total_tokens,
        });

        Ok(GenerateResult { tokens, generation })
    }

    fn build_payload(&self, messages: &[Message], stream: bool) -> NebiusPayload {
        let mut payload = NebiusPayload {
            model: self.model.clone(),
            messages: messages
                .iter()
                .map(NebiusMessage::from_message)
                .collect::<Vec<_>>(),
            max_tokens: self.options.max_tokens,
            stream: None,
            stop: self.options.stop_words.clone(),
            temperature: self.options.temperature,
            top_p: self.options.top_p,
            frequency_penalty: None,
            presence_penalty: None,
        };
        if stream {
            payload.stream = Some(true);
        }
        payload
    }
}

#[async_trait]
impl LLM for Nebius {
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
            .post("https://api.studio.nebius.ai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .build()?;

        let stream = client.execute(request).await?;
        let stream = stream.bytes_stream();

        let processed_stream = stream.then(move |result| async move {
            match result {
                Ok(bytes) => {
                    let value: Value = parse_sse_to_json(&String::from_utf8_lossy(&bytes))?;
                    if let Some(choices) = value["choices"].as_array() {
                        if let Some(choice) = choices.first() {
                            if let Some(delta) = choice["delta"].as_object() {
                                if let Some(content) = delta["content"].as_str() {
                                    return Ok(StreamData::new(value.clone(), None, content));
                                }
                            }
                        }
                    }
                    Ok(StreamData::new(value, None, ""))
                }
                Err(e) => Err(LLMError::RequestError(e)),
            }
        });

        Ok(Box::pin(processed_stream))
    }

    fn add_options(&mut self, options: CallOptions) {
        self.options.merge_options(options)
    }
}

fn parse_sse_to_json(sse_data: &str) -> Result<Value, LLMError> {
    if let Ok(json) = serde_json::from_str::<Value>(sse_data) {
        return parse_error(&json);
    }

    let lines: Vec<&str> = sse_data.trim().split('\n').collect();
    let mut event_data: HashMap<&str, String> = HashMap::new();

    for line in lines {
        if let Some((key, value)) = line.split_once(": ") {
            event_data.insert(key, value.to_string());
        }
    }

    if let Some(data) = event_data.get("data") {
        if data == "[DONE]" {
            return Ok(serde_json::json!({"done": true}));
        }
        let data: Value = serde_json::from_str(data)?;
        return match data["error"].as_object() {
            Some(_) => parse_error(&data),
            None => Ok(data),
        };
    }
    log::error!("No data field in the SSE event");
    Err(LLMError::ContentNotFound("data".to_string()))
}

fn parse_error(json: &Value) -> Result<Value, LLMError> {
    if let Some(error) = json["error"].as_object() {
        let error_type = error["type"].as_str().unwrap_or("");
        let message = error["message"].as_str().unwrap_or("").to_string();
        match error_type {
            "invalid_request_error" => Err(NebiusError::InvalidRequestError(message))?,
            "authentication_error" => Err(NebiusError::AuthenticationError(message))?,
            "permission_error" => Err(NebiusError::PermissionError(message))?,
            "not_found_error" => Err(NebiusError::NotFoundError(message))?,
            "rate_limit_error" => Err(NebiusError::RateLimitError(message))?,
            "api_error" => Err(NebiusError::ApiError(message))?,
            "service_unavailable" => Err(NebiusError::ServiceUnavailableError(message))?,
            _ => return Ok(json.clone()),
        }
    }
    Ok(json.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    #[ignore]
    async fn test_nebius_generate() {
        let nebius = Nebius::new();

        let res = nebius
            .generate(&[Message::new_human_message("Hi, how are you doing")])
            .await
            .unwrap();

        println!("{:?}", res)
    }

    #[test]
    #[ignore]
    async fn test_nebius_stream() {
        let nebius = Nebius::new();
        let mut stream = nebius
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
