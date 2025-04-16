use crate::{
    language_models::{llm::LLM, options::CallOptions, GenerateResult, LLMError, TokenUsage},
    llm::QwenError,
    schemas::{Message, StreamData},
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;
use serde_json::Value;
use std::{pin::Pin, str::from_utf8};

use super::models::{ApiResponse, ErrorResponse, Payload, QwenMessage, StreamResponse};

/// Parse error from JSON response and return appropriate QwenError
fn parse_error_response(code: &str, message: &str) -> LLMError {
    match code {
        // 400 errors
        "InvalidParameter" | "invalid_parameter_error" => 
            LLMError::QwenError(QwenError::InvalidParameterError(message.to_string())),
        "APIConnectionError" => 
            LLMError::QwenError(QwenError::APIConnectionError(message.to_string())),
        
        // 401 errors
        "InvalidApiKey" => 
            LLMError::QwenError(QwenError::InvalidApiKeyError(message.to_string())),
        
        // 429 errors
        "ModelServingError" => 
            LLMError::QwenError(QwenError::ModelServingError(message.to_string())),
        "PrepaidBillOverdue" => 
            LLMError::QwenError(QwenError::PrepaidBillOverdueError(message.to_string())),
        "PostpaidBillOverdue" => 
            LLMError::QwenError(QwenError::PostpaidBillOverdueError(message.to_string())),
        "CommodityNotPurchased" => 
            LLMError::QwenError(QwenError::CommodityNotPurchasedError(message.to_string())),
        
        // 500 errors
        "InternalError" | "internal_error" => 
            LLMError::QwenError(QwenError::InternalError(message.to_string())),
        "InternalError.Algo" => 
            LLMError::QwenError(QwenError::InternalAlgorithmError(message.to_string())),
        "InternalError.Timeout" => 
            LLMError::QwenError(QwenError::TimeoutError(message.to_string())),
        "RewriteFailed" => 
            LLMError::QwenError(QwenError::RewriteFailedError(message.to_string())),
        "RetrivalFailed" => 
            LLMError::QwenError(QwenError::RetrievalFailedError(message.to_string())),
        "AppProcessFailed" => 
            LLMError::QwenError(QwenError::AppProcessFailedError(message.to_string())),
        "ModelServiceFailed" => 
            LLMError::QwenError(QwenError::ModelServiceFailedError(message.to_string())),
        "InvokePluginFailed" => 
            LLMError::QwenError(QwenError::InvokePluginFailedError(message.to_string())),
        "SystemError" | "system_error" => 
            LLMError::QwenError(QwenError::SystemError(message.to_string())),
        
        // 503 errors
        "ModelUnavailable" => 
            LLMError::QwenError(QwenError::ModelUnavailableError(message.to_string())),
        
        // Other errors
        "mismatched_model" => 
            LLMError::QwenError(QwenError::MismatchedModelError(message.to_string())),
        "duplicate_custom_id" => 
            LLMError::QwenError(QwenError::DuplicateCustomIdError(message.to_string())),
        "model_not_found" => 
            LLMError::QwenError(QwenError::ModelNotFoundError(message.to_string())),
        
        // Default error
        _ => LLMError::QwenError(QwenError::SystemError(format!("Unknown error code: {}, message: {}", code, message))),
    }
}

/// Qwen model options
pub enum QwenModel {
    /// Qwen-Max
    QwenMax,
    /// Qwen-Turbo
    QwenTurbo,
    /// Qwen-Plus
    QwenPlus,
    /// Qwen-Long
    QwenLong,
    /// Qwen-72B-Chat (Open Source Version)
    Qwen1_72B_Chat,
    /// Qwen-14B-Chat (Open Source Version)
    Qwen1_14B_Chat,
    /// Qwen-7B-Chat (Open Source Version)
    Qwen1_7B_Chat,
    /// Qwen-1.8B-Chat (Open Source Version)
    Qwen1_1_8B_Chat,
    /// Qwen1.5-110B-Chat (Open Source Version)
    Qwen1_5_110B_Chat,
    /// Qwen1.5-72B-Chat (Open Source Version)
    Qwen1_5_72B_Chat,
    /// Qwen1.5-32B-Chat (Open Source Version)
    Qwen1_5_32B_Chat,
    /// Qwen1.5-14B-Chat (Open Source Version)
    Qwen1_5_14B_Chat,
    /// Qwen1.5-7B-Chat (Open Source Version)
    Qwen1_5_7B_Chat,
    /// Qwen1.5-1.8B-Chat (Open Source Version)
    Qwen1_5_1_8B_Chat,
    /// Qwen1.5-0.5B-Chat (Open Source Version)
    Qwen1_5_0_5B_Chat,
    /// Qwen2-72b-Instruct (Open Source Version)
    QWEN2_72B_INSTRUCT,
    /// Qwen2-57b-a14b-Instruct (Open Source Version)
    QWEN2_57B_A14B_INSTRUCT,
    /// Qwen2-7b-Instruct (Open Source Version)
    QWEN2_7B_INSTRUCT,
    /// Qwen2-1.5b-Instruct (Open Source Version)
    QWEN2_1_5B_INSTRUCT,
    /// Qwen2-0.5b-Instruct (Open Source Version)
    QWEN2_0_5B_INSTRUCT,
    /// Qwen2.5-14B-Instruct-1M (Open Source Version)
    Qwen2_5_14B_INSTRUCT_1M,
    /// Qwen2.5-7B-Instruct-1M (Open Source Version)
    Qwen2_5_7B_INSTRUCT_1M,
    /// Qwen2.5-72B-Instruct (Open Source Version)
    Qwen2_5_72B_INSTRUCT,
    /// Qwen2.5-32B-Instruct (Open Source Version)
    Qwen2_5_32B_INSTRUCT,
    /// Qwen2.5-14B-Instruct (Open Source Version)
    Qwen2_5_14B_INSTRUCT,
    /// Qwen2.5-7B-Instruct (Open Source Version)
    Qwen2_5_7B_INSTRUCT,
    /// Qwen2.5-3B-Instruct (Open Source Version)
    Qwen2_5_3B_INSTRUCT,
    /// Qwen2.5-1.5B-Instruct (Open Source Version)
    Qwen2_5_1_5B_INSTRUCT,
    /// Qwen2.5-0.5B-Instruct (Open Source Version)
    Qwen2_5_0_5B_INSTRUCT,
}

impl ToString for QwenModel {
    fn to_string(&self) -> String {
        match self {
            QwenModel::QwenMax => "qwen-max".to_string(),
            QwenModel::QwenTurbo => "qwen-turbo".to_string(),
            QwenModel::QwenPlus => "qwen-plus".to_string(),
            QwenModel::QwenLong => "qwen-long".to_string(),
            QwenModel::Qwen1_72B_Chat => "qwen-72b-chat".to_string(),
            QwenModel::Qwen1_14B_Chat => "qwen-14b-chat".to_string(),
            QwenModel::Qwen1_7B_Chat => "qwen-7b-chat".to_string(),
            QwenModel::Qwen1_1_8B_Chat => "qwen-1.8b-chat".to_string(),
            QwenModel::Qwen1_5_110B_Chat => "qwen1.5-110b-chat".to_string(),
            QwenModel::Qwen1_5_72B_Chat => "qwen-1.72b-chat".to_string(),
            QwenModel::Qwen1_5_32B_Chat => "qwen1.5-32b-chat".to_string(),
            QwenModel::Qwen1_5_14B_Chat => "qwen1.5-14b-chat".to_string(),
            QwenModel::Qwen1_5_7B_Chat => "qwen1.5-7b-chat".to_string(),
            QwenModel::Qwen1_5_1_8B_Chat => "qwen1.5-1.8b-chat".to_string(),
            QwenModel::Qwen1_5_0_5B_Chat => "qwen1.5-0.5b-chat".to_string(),
            QwenModel::QWEN2_72B_INSTRUCT => "qwen2-72b-instruct".to_string(),
            QwenModel::QWEN2_57B_A14B_INSTRUCT => "qwen2-57b-a14b-instruct".to_string(),
            QwenModel::QWEN2_7B_INSTRUCT => "qwen2-7b-instruct".to_string(),
            QwenModel::QWEN2_1_5B_INSTRUCT => "qwen2-1.5-b-instruct".to_string(),
            QwenModel::QWEN2_0_5B_INSTRUCT => "qwen2-0.5-b-instruct".to_string(),
            QwenModel::Qwen2_5_14B_INSTRUCT_1M => "qwen2.5-14b-instruct-1m".to_string(),
            QwenModel::Qwen2_5_7B_INSTRUCT_1M => "qwen2.5-7b-instruct-1m".to_string(),
            QwenModel::Qwen2_5_72B_INSTRUCT => "qwen2.5-72b-instruct".to_string(),
            QwenModel::Qwen2_5_32B_INSTRUCT => "qwen2.5-32b-instruct".to_string(),
            QwenModel::Qwen2_5_14B_INSTRUCT => "qwen2.5-14b-instruct".to_string(),
            QwenModel::Qwen2_5_7B_INSTRUCT => "qwen2.5-7b-instruct".to_string(),
            QwenModel::Qwen2_5_3B_INSTRUCT => "qwen2.5-3b-instruct".to_string(),
            QwenModel::Qwen2_5_1_5B_INSTRUCT => "qwen2.5-1.5b-instruct".to_string(),
            QwenModel::Qwen2_5_0_5B_INSTRUCT => "qwen2.5-0.5b-instruct".to_string(),
        }
    }
}

/// Qwen client
#[derive(Clone)]
pub struct Qwen {
    model: String,
    options: CallOptions,
    api_key: String,
    base_url: String,
}

impl Default for Qwen {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen {
    /// Create a new Qwen client with default settings
    pub fn new() -> Self {
        Self {
            model: QwenModel::QwenTurbo.to_string(), // Default to Turbo model
            options: CallOptions::default(),
            api_key: std::env::var("QWEN_API_KEY").unwrap_or_default(),
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions".to_string(),
        }
    }

    /// Set the model
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    /// Set call options
    pub fn with_options(mut self, options: CallOptions) -> Self {
        self.options = options;
        self
    }

    /// Set API key
    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Set the base URL
    pub fn with_base_url<S: Into<String>>(mut self, base_url: S) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Generates text using the Qwen API
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let client = Client::new();
        let is_stream = self.options.streaming_func.is_some();

        let payload = self.build_payload(messages, is_stream);
        let res = client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        match res.status().as_u16() {
            200 => {
                let api_response = res.json::<ApiResponse>().await?;
                
                // Extract the first choice content
                let generation = match api_response.choices.first() {
                    Some(choice) => choice.message.content.clone(),
                    None => return Err(LLMError::ContentNotFound("No content returned from API".to_string())),
                };

                let tokens = Some(TokenUsage {
                    prompt_tokens: api_response.usage.prompt_tokens,
                    completion_tokens: api_response.usage.completion_tokens,
                    total_tokens: api_response.usage.total_tokens,
                });

                Ok(GenerateResult { tokens, generation })
            }
            400 => {
                let error = res.json::<ErrorResponse>().await?;
                Err(parse_error_response(error.code.as_str(), &error.message))
            }
            401 => {
                let error = res.json::<ErrorResponse>().await?;
                Err(parse_error_response(error.code.as_str(), &error.message))
            }
            429 => {
                let error = res.json::<ErrorResponse>().await?;
                Err(parse_error_response(error.code.as_str(), &error.message))
            }
            500 => {
                let error = res.json::<ErrorResponse>().await?;
                Err(parse_error_response(error.code.as_str(), &error.message))
            }
            503 => {
                let error = res.json::<ErrorResponse>().await?;
                Err(parse_error_response(error.code.as_str(), &error.message))
            }
            _ => {
                let error = res.json::<ErrorResponse>().await?;
                Err(parse_error_response(error.code.as_str(), &error.message))
            }
        }
    }

    /// Builds the API payload from messages
    fn build_payload(&self, messages: &[Message], stream: bool) -> Payload {
        let mut payload = Payload {
            model: self.model.clone(),
            messages: messages
                .iter()
                .map(QwenMessage::from_message)
                .collect::<Vec<_>>(),
            max_tokens: self.options.max_tokens,
            stream: None,
            stop: self.options.stop_words.clone(),
            temperature: self.options.temperature,
            top_p: self.options.top_p,
            seed: None, // Optional
            result_format: None, // Optional
        };

        if stream {
            payload.stream = Some(true);
        }

        payload
    }
}

#[async_trait]
impl LLM for Qwen {
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
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", &self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&payload)
            .build()?;

        let stream = client.execute(request).await?;
        let stream = stream.bytes_stream();
        
        let processed_stream = stream.then(move |result| {
            async move {
                match result {
                    Ok(bytes) => {
                        // Handle Qwen streaming response
                        let bytes_str = from_utf8(&bytes).map_err(|e| LLMError::OtherError(e.to_string()))?;
                        
                        if bytes_str.trim().starts_with("{") {
                            let json_value: Value = serde_json::from_str(bytes_str)
                                .map_err(|e| LLMError::OtherError(format!("Failed to parse stream response: {}", e)))?;
                            
                            // Check if this is an error response
                            if json_value.get("code").is_some() && json_value.get("message").is_some() {
                                // This looks like an error response
                                let code = json_value["code"].as_str().unwrap_or("unknown");
                                let message = json_value["message"].as_str().unwrap_or("Unknown error");
                                return Err(parse_error_response(code, message));
                            }

                            // Try to parse as a stream response, but handle missing fields
                            // This handles both standard and non-standard response formats
                            let content = if let Some(choices) = json_value.get("choices") {
                                if let Some(choice) = choices.as_array().and_then(|arr| arr.first()) {
                                    if let Some(delta) = choice.get("delta") {
                                        delta.get("content").and_then(|c| c.as_str()).unwrap_or("")
                                    } else {
                                        // Try to get content directly from choice if delta isn't present
                                        choice.get("content").and_then(|c| c.as_str()).unwrap_or("")
                                    }
                                } else {
                                    ""
                                }
                            } else {
                                // If no choices field, check for direct content field
                                json_value.get("content").and_then(|c| c.as_str()).unwrap_or("")
                            };

                            return Ok(StreamData::new(
                                json_value.clone(),
                                None,
                                content,
                            ));
                        }
                        
                        // Handle simple text responses or other formats
                        Ok(StreamData::new(
                            Value::String(bytes_str.to_string()),
                            None,
                            bytes_str.trim(),
                        ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    #[ignore]
    async fn test_qwen_generate() {
        let qwen = Qwen::new();

        let res = qwen
            .generate(&[Message::new_human_message("Hello!")])
            .await
            .unwrap();

        println!("{:?}", res)
    }

    #[test]
    #[ignore]
    async fn test_qwen_stream() {
        let qwen = Qwen::new();
        let mut stream = qwen
            .stream(&[Message::new_human_message("Hello!")])
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