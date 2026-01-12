//! # AWS Bedrock LLM Integration
//!
//! This module provides integration with AWS Bedrock's foundation models.
//! AWS Bedrock is a fully managed service that makes foundation models from leading AI companies
//! available via an API.
//!
//! ## Supported Models
//!
//! - Anthropic Claude (claude-v2, claude-instant-v1, claude-3-*, claude-4-*)
//! - AI21 Labs Jurassic
//! - Amazon Titan
//! - Cohere Command
//! - Meta Llama 2
//!
//! ## Example
//!
//! ```rust,no_run
//! use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let bedrock = Bedrock::default()
//!         .with_model(BedrockModel::AnthropicClaudeV2)
//!         .with_region("us-east-1")
//!         .with_temperature(0.7);
//!
//!     let response = bedrock.invoke("What is the capital of France?").await?;
//!     println!("{}", response);
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use aws_config::meta::region::RegionProviderChain;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_sdk_bedrockruntime::types::{ContentBlock, ConversationRole, Message as BedrockMessage};
use serde_json::json;
use std::error::Error as StdError;
use std::fmt;

use crate::language_models::llm::LLM;
use crate::language_models::{GenerateResult, LLMError};
use crate::schemas::{Message, StreamData};

/// Errors that can occur when using the Bedrock LLM
#[derive(Debug)]
pub enum BedrockError {
    /// AWS SDK error
    AwsError(String),
    /// Invalid model configuration
    InvalidModel(String),
    /// Serialization/deserialization error
    SerdeError(serde_json::Error),
    /// Invalid region
    InvalidRegion(String),
    /// Model invocation error
    InvocationError(String),
}

impl fmt::Display for BedrockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BedrockError::AwsError(msg) => write!(f, "AWS Error: {}", msg),
            BedrockError::InvalidModel(msg) => write!(f, "Invalid Model: {}", msg),
            BedrockError::SerdeError(e) => write!(f, "Serialization Error: {}", e),
            BedrockError::InvalidRegion(msg) => write!(f, "Invalid Region: {}", msg),
            BedrockError::InvocationError(msg) => write!(f, "Invocation Error: {}", msg),
        }
    }
}

impl StdError for BedrockError {}

impl From<serde_json::Error> for BedrockError {
    fn from(err: serde_json::Error) -> Self {
        BedrockError::SerdeError(err)
    }
}

/// Supported Bedrock models
#[derive(Debug, Clone, PartialEq)]
pub enum BedrockModel {
    /// Anthropic Claude v2
    AnthropicClaudeV2,
    /// Anthropic Claude Instant v1
    AnthropicClaudeInstantV1,
    /// Anthropic Claude 3 Sonnet
    AnthropicClaude3Sonnet,
    /// Anthropic Claude 3 Haiku
    AnthropicClaude3Haiku,
    /// Anthropic Claude 3 Opus
    AnthropicClaude3Opus,
    /// Anthropic Claude 3.5 Haiku
    AnthropicClaude35Haiku,
    /// Anthropic Claude 4 Sonnet
    AnthropicClaude4Sonnet,
    /// Anthropic Claude 4.5 Haiku
    AnthropicClaude45Haiku,
    /// Anthropic Claude 4.1 Opus
    AnthropicClaude41Opus,
    /// Anthropic Claude 4.5 Opus
    AnthropicClaude45Opus,
    /// Anthropic Claude 4.5 Sonnet
    AnthropicClaude45Sonnet,
    /// AI21 Jurassic-2 Mid
    AI21Jurassic2Mid,
    /// AI21 Jurassic-2 Ultra
    AI21Jurassic2Ultra,
    /// Amazon Titan Text Express
    AmazonTitanTextExpress,
    /// Amazon Titan Text Lite
    AmazonTitanTextLite,
    /// Cohere Command
    CohereCommand,
    /// Cohere Command Light
    CohereCommandLight,
    /// Meta Llama 2 Chat 13B
    MetaLlama2Chat13B,
    /// Meta Llama 2 Chat 70B
    MetaLlama2Chat70B,
    /// Custom model ID
    Custom(String),
}

impl BedrockModel {
    /// Get the model ID string for AWS Bedrock
    pub fn model_id(&self) -> String {
        match self {
            BedrockModel::AnthropicClaudeV2 => "anthropic.claude-v2".to_string(),
            BedrockModel::AnthropicClaudeInstantV1 => "anthropic.claude-instant-v1".to_string(),
            BedrockModel::AnthropicClaude3Sonnet => {
                "anthropic.claude-3-sonnet-20240229-v1:0".to_string()
            }
            BedrockModel::AnthropicClaude3Haiku => {
                "anthropic.claude-3-haiku-20240307-v1:0".to_string()
            }
            BedrockModel::AnthropicClaude3Opus => {
                "anthropic.claude-3-opus-20240229-v1:0".to_string()
            }
            BedrockModel::AnthropicClaude35Haiku => {
                "anthropic.claude-3-5-haiku-20241022-v1:0".to_string()
            }
            BedrockModel::AnthropicClaude4Sonnet => {
                "anthropic.claude-sonnet-4-20250514-v1:0".to_string()
            }
            BedrockModel::AnthropicClaude45Haiku => {
                "anthropic.claude-haiku-4-5-20251001-v1:0".to_string()
            }
            BedrockModel::AnthropicClaude41Opus => {
                "anthropic.claude-opus-4-1-20250805-v1:0".to_string()
            }
            BedrockModel::AnthropicClaude45Opus => {
                "anthropic.claude-opus-4-5-20251101-v1:0".to_string()
            }
            BedrockModel::AnthropicClaude45Sonnet => {
                "anthropic.claude-sonnet-4-5-20250929-v1:0".to_string()
            }
            
            BedrockModel::AI21Jurassic2Mid => "ai21.j2-mid-v1".to_string(),
            BedrockModel::AI21Jurassic2Ultra => "ai21.j2-ultra-v1".to_string(),
            BedrockModel::AmazonTitanTextExpress => "amazon.titan-text-express-v1".to_string(),
            BedrockModel::AmazonTitanTextLite => "amazon.titan-text-lite-v1".to_string(),
            BedrockModel::CohereCommand => "cohere.command-text-v14".to_string(),
            BedrockModel::CohereCommandLight => "cohere.command-light-text-v14".to_string(),
            BedrockModel::MetaLlama2Chat13B => "meta.llama2-13b-chat-v1".to_string(),
            BedrockModel::MetaLlama2Chat70B => "meta.llama2-70b-chat-v1".to_string(),
            BedrockModel::Custom(id) => id.clone(),
        }
    }

    /// Get the model provider
    pub fn provider(&self) -> &str {
        match self {
            BedrockModel::AnthropicClaudeV2
            | BedrockModel::AnthropicClaudeInstantV1
            | BedrockModel::AnthropicClaude3Sonnet
            | BedrockModel::AnthropicClaude3Haiku
            | BedrockModel::AnthropicClaude3Opus
            | BedrockModel::AnthropicClaude35Haiku
            | BedrockModel::AnthropicClaude4Sonnet
            | BedrockModel::AnthropicClaude45Haiku
            | BedrockModel::AnthropicClaude41Opus
            | BedrockModel::AnthropicClaude45Opus
            | BedrockModel::AnthropicClaude45Sonnet => "anthropic",
            BedrockModel::AI21Jurassic2Mid | BedrockModel::AI21Jurassic2Ultra => "ai21",
            BedrockModel::AmazonTitanTextExpress | BedrockModel::AmazonTitanTextLite => "amazon",
            BedrockModel::CohereCommand | BedrockModel::CohereCommandLight => "cohere",
            BedrockModel::MetaLlama2Chat13B | BedrockModel::MetaLlama2Chat70B => "meta",
            BedrockModel::Custom(model_id) => {
                // Infer provider from model ID
                if model_id.starts_with("anthropic.") {
                    "anthropic"
                } else if model_id.starts_with("ai21.") {
                    "ai21"
                } else if model_id.starts_with("amazon.") {
                    "amazon"
                } else if model_id.starts_with("cohere.") {
                    "cohere"
                } else if model_id.starts_with("meta.") {
                    "meta"
                } else {
                    // Default to anthropic for unknown custom models
                    "anthropic"
                }
            }
        }
    }
}

impl Default for BedrockModel {
    fn default() -> Self {
        BedrockModel::AnthropicClaude3Sonnet
    }
}

/// Configuration for the Bedrock LLM
#[derive(Debug, Clone)]
pub struct BedrockConfig {
    /// AWS region
    pub region: Option<String>,
    /// Model to use
    pub model: BedrockModel,
    /// Temperature (0.0 to 1.0)
    pub temperature: Option<f32>,
    /// Maximum tokens to generate
    pub max_tokens: Option<i32>,
    /// Top P sampling parameter
    pub top_p: Option<f32>,
    /// Top K sampling parameter
    pub top_k: Option<i32>,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
    /// Additional model-specific parameters
    pub model_kwargs: serde_json::Value,
}

impl Default for BedrockConfig {
    fn default() -> Self {
        Self {
            region: Some("us-west-2".to_string()),
            model: BedrockModel::default(),
            temperature: Some(0.7),
            max_tokens: Some(512),
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            model_kwargs: json!({}),
        }
    }
}

/// AWS Bedrock LLM client
pub struct Bedrock {
    client: Option<BedrockClient>,
    config: BedrockConfig,
}

impl Bedrock {
    /// Create a new Bedrock instance with default configuration
    pub fn new() -> Self {
        Self {
            client: None,
            config: BedrockConfig::default(),
        }
    }

    /// Set the AWS region
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.config.region = Some(region.into());
        self.client = None; // Reset client to force reinitialization
        self
    }

    /// Set the model
    pub fn with_model(mut self, model: BedrockModel) -> Self {
        self.config.model = model;
        self
    }

    /// Set the temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    /// Set the maximum tokens
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Set top_p sampling parameter
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.config.top_p = Some(top_p);
        self
    }

    /// Set top_k sampling parameter
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.config.top_k = Some(top_k);
        self
    }

    /// Add a stop sequence
    pub fn with_stop_sequence(mut self, stop: impl Into<String>) -> Self {
        self.config.stop_sequences.push(stop.into());
        self
    }

    /// Set additional model parameters
    pub fn with_model_kwargs(mut self, kwargs: serde_json::Value) -> Self {
        self.config.model_kwargs = kwargs;
        self
    }

    /// Initialize the AWS Bedrock client
    async fn get_client(&mut self) -> Result<BedrockClient, BedrockError> {
        if self.client.is_none() {
            let region = self
                .config
                .region
                .clone()
                .unwrap_or_else(|| "us-east-1".to_string());

            // Use Box::leak to convert String to &'static str for region provider
            let region_ref: &'static str = Box::leak(region.into_boxed_str());
            let region_provider = RegionProviderChain::first_try(region_ref);

            let config = aws_config::defaults(BehaviorVersion::latest())
                .region(region_provider)
                .load()
                .await;

            self.client = Some(BedrockClient::new(&config));
        }

        Ok(self.client.as_ref().unwrap().clone())
    }

    /// Format the prompt according to the model's requirements
    fn format_prompt(&self, prompt: &str) -> String {
        match self.config.model.provider() {
            "anthropic" => {
                // Anthropic models require specific formatting
                if prompt.starts_with("Human:") || prompt.starts_with("\n\nHuman:") {
                    prompt.to_string()
                } else {
                    format!("\n\nHuman: {}\n\nAssistant:", prompt)
                }
            }
            _ => prompt.to_string(),
        }
    }

    /// Build the request body for the model
    fn build_request_body(&self, prompt: &str) -> Result<serde_json::Value, BedrockError> {
        let formatted_prompt = self.format_prompt(prompt);

        let body = match self.config.model.provider() {
            "anthropic" => {
                let mut request = json!({
                    "prompt": formatted_prompt,
                    "max_tokens_to_sample": self.config.max_tokens.unwrap_or(512),
                });

                if let Some(temp) = self.config.temperature {
                    request["temperature"] = json!(temp);
                }
                if let Some(top_p) = self.config.top_p {
                    request["top_p"] = json!(top_p);
                }
                if let Some(top_k) = self.config.top_k {
                    request["top_k"] = json!(top_k);
                }
                if !self.config.stop_sequences.is_empty() {
                    request["stop_sequences"] = json!(self.config.stop_sequences);
                }

                request
            }
            "ai21" => {
                json!({
                    "prompt": formatted_prompt,
                    "maxTokens": self.config.max_tokens.unwrap_or(512),
                    "temperature": self.config.temperature.unwrap_or(0.7),
                    "topP": self.config.top_p.unwrap_or(1.0),
                })
            }
            "amazon" => {
                json!({
                    "inputText": formatted_prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": self.config.max_tokens.unwrap_or(512),
                        "temperature": self.config.temperature.unwrap_or(0.7),
                        "topP": self.config.top_p.unwrap_or(1.0),
                        "stopSequences": self.config.stop_sequences,
                    }
                })
            }
            "cohere" => {
                json!({
                    "prompt": formatted_prompt,
                    "max_tokens": self.config.max_tokens.unwrap_or(512),
                    "temperature": self.config.temperature.unwrap_or(0.7),
                    "p": self.config.top_p.unwrap_or(0.9),
                    "k": self.config.top_k.unwrap_or(0),
                    "stop_sequences": self.config.stop_sequences,
                })
            }
            "meta" => {
                json!({
                    "prompt": formatted_prompt,
                    "max_gen_len": self.config.max_tokens.unwrap_or(512),
                    "temperature": self.config.temperature.unwrap_or(0.7),
                    "top_p": self.config.top_p.unwrap_or(0.9),
                })
            }
            _ => {
                return Err(BedrockError::InvalidModel(format!(
                    "Unsupported model provider: {}",
                    self.config.model.provider()
                )))
            }
        };

        Ok(body)
    }

    /// Parse the response from the model
    fn parse_response(&self, response_body: &[u8]) -> Result<String, BedrockError> {
        let response_json: serde_json::Value = serde_json::from_slice(response_body)?;

        let text = match self.config.model.provider() {
            "anthropic" => response_json["completion"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            "ai21" => response_json["completions"][0]["data"]["text"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            "amazon" => response_json["results"][0]["outputText"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            "cohere" => response_json["generations"][0]["text"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            "meta" => response_json["generation"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            _ => {
                return Err(BedrockError::InvalidModel(format!(
                    "Unsupported model provider: {}",
                    self.config.model.provider()
                )))
            }
        };

        Ok(text)
    }

    /// Check if the model requires the Converse API (Claude 3+, Claude 4+)
    fn requires_converse_api(&self) -> bool {
        match &self.config.model {
            BedrockModel::AnthropicClaude3Sonnet
            | BedrockModel::AnthropicClaude3Haiku
            | BedrockModel::AnthropicClaude3Opus
            | BedrockModel::AnthropicClaude35Haiku
            | BedrockModel::AnthropicClaude4Sonnet
            | BedrockModel::AnthropicClaude45Haiku
            | BedrockModel::AnthropicClaude41Opus
            | BedrockModel::AnthropicClaude45Opus
            | BedrockModel::AnthropicClaude45Sonnet => true,
            BedrockModel::Custom(model_id) => {
                // Check if custom model ID is Claude 3+ or Claude 4+
                model_id.contains("claude-3-")
                    || model_id.contains("claude-3-5-")
                    || model_id.contains("claude-4-")
            }
            _ => false,
        }
    }

    /// Convert langchain messages to Bedrock Converse API format
    fn messages_to_converse_format(&self, messages: &[Message]) -> (Option<String>, Vec<BedrockMessage>) {
        use crate::schemas::messages::MessageType;

        let mut system_prompt: Option<String> = None;
        let mut converse_messages: Vec<BedrockMessage> = Vec::new();

        for msg in messages {
            match &msg.message_type {
                MessageType::SystemMessage => {
                    // Bedrock Converse API takes system as a separate parameter
                    system_prompt = Some(msg.content.clone());
                }
                MessageType::HumanMessage => {
                    let content_block = ContentBlock::Text(msg.content.clone());
                    let bedrock_msg = BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(content_block)
                        .build()
                        .unwrap();
                    converse_messages.push(bedrock_msg);
                }
                MessageType::AIMessage => {
                    let content_block = ContentBlock::Text(msg.content.clone());
                    let bedrock_msg = BedrockMessage::builder()
                        .role(ConversationRole::Assistant)
                        .content(content_block)
                        .build()
                        .unwrap();
                    converse_messages.push(bedrock_msg);
                }
                MessageType::ToolMessage => {
                    // Default to user message for tool messages
                    let content_block = ContentBlock::Text(msg.content.clone());
                    let bedrock_msg = BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(content_block)
                        .build()
                        .unwrap();
                    converse_messages.push(bedrock_msg);
                }
            }
        }

        (system_prompt, converse_messages)
    }
}

impl Default for Bedrock {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LLM for Bedrock {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let mut bedrock = self.clone();
        let client = bedrock.get_client().await.map_err(|e| LLMError::OtherError(e.to_string()))?;

        // Use Converse API for Claude 3+ models
        if bedrock.requires_converse_api() {
            let (system_prompt, converse_messages) = bedrock.messages_to_converse_format(messages);

            let mut converse_request = client
                .converse()
                .model_id(bedrock.config.model.model_id());

            // Add system prompt if present
            if let Some(system) = system_prompt {
                use aws_sdk_bedrockruntime::types::SystemContentBlock;
                let system_block = SystemContentBlock::Text(system);
                converse_request = converse_request.system(system_block);
            }

            // Add messages
            for msg in converse_messages {
                converse_request = converse_request.messages(msg);
            }

            // Add inference configuration
            let mut inference_config = aws_sdk_bedrockruntime::types::InferenceConfiguration::builder();

            if let Some(max_tokens) = bedrock.config.max_tokens {
                inference_config = inference_config.max_tokens(max_tokens);
            }
            if let Some(temperature) = bedrock.config.temperature {
                inference_config = inference_config.temperature(temperature);
            }
            if let Some(top_p) = bedrock.config.top_p {
                inference_config = inference_config.top_p(top_p);
            }

            converse_request = converse_request.inference_config(inference_config.build());

            // Note: Bedrock Converse API handles stop sequences differently
            // They are model-specific and may not be supported via the top-level API

            let response = match converse_request.send().await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Bedrock SDK error (debug): {:?}", e);
                    return Err(LLMError::OtherError(format!("Bedrock invocation error: {}", e)));
                }
            };

            // Extract text from response
            let text = response
                .output()
                .and_then(|output| output.as_message().ok())
                .and_then(|msg| msg.content().first())
                .and_then(|content| content.as_text().ok())
                .map(|s| s.to_string())
                .unwrap_or_default();

            Ok(GenerateResult {
                generation: text,
                tokens: None,
            })
        } else {
            // Use legacy invoke_model for older models (Claude 2, Titan, etc.)
            let prompt = bedrock.messages_to_string(messages);
            let request_body = bedrock.build_request_body(&prompt).map_err(|e| LLMError::OtherError(e.to_string()))?;
            let body_bytes = serde_json::to_vec(&request_body).map_err(|e| LLMError::SerdeError(e))?;

            let send_result = client
                .invoke_model()
                .model_id(bedrock.config.model.model_id())
                .body(Blob::new(body_bytes))
                .send()
                .await;

            let response = match send_result {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Bedrock SDK error (debug): {:?}", e);
                    return Err(LLMError::OtherError(format!("Bedrock invocation error: {}", e)));
                }
            };

            let response_body = response.body().as_ref();
            let text = bedrock.parse_response(response_body).map_err(|e| LLMError::OtherError(e.to_string()))?;

            Ok(GenerateResult {
                generation: text,
                tokens: None,
            })
        }
    }

    async fn stream(
        &self,
        _messages: &[Message],
    ) -> Result<std::pin::Pin<Box<dyn futures::Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        Err(LLMError::OtherError("Streaming is not yet implemented for Bedrock".to_string()))
    }
}

// Clone implementation for Bedrock
impl Clone for Bedrock {
    fn clone(&self) -> Self {
        Self {
            client: None, // Client is not cloneable, will be reinitialized on use
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bedrock_model_ids() {
        assert_eq!(
            BedrockModel::AnthropicClaudeV2.model_id(),
            "anthropic.claude-v2"
        );
        assert_eq!(
            BedrockModel::AnthropicClaudeInstantV1.model_id(),
            "anthropic.claude-instant-v1"
        );
        assert_eq!(
            BedrockModel::AI21Jurassic2Mid.model_id(),
            "ai21.j2-mid-v1"
        );
        assert_eq!(
            BedrockModel::AmazonTitanTextExpress.model_id(),
            "amazon.titan-text-express-v1"
        );
    }

    #[test]
    fn test_bedrock_model_providers() {
        assert_eq!(BedrockModel::AnthropicClaudeV2.provider(), "anthropic");
        assert_eq!(BedrockModel::AI21Jurassic2Mid.provider(), "ai21");
        assert_eq!(BedrockModel::AmazonTitanTextExpress.provider(), "amazon");
        assert_eq!(BedrockModel::CohereCommand.provider(), "cohere");
        assert_eq!(BedrockModel::MetaLlama2Chat13B.provider(), "meta");
    }

    #[test]
    fn test_custom_model() {
        let custom = BedrockModel::Custom("my-custom-model".to_string());
        assert_eq!(custom.model_id(), "my-custom-model");
        assert_eq!(custom.provider(), "anthropic"); // Default provider for unknown custom models
    }

    #[test]
    fn test_bedrock_builder() {
        let bedrock = Bedrock::default()
            .with_model(BedrockModel::AnthropicClaudeV2)
            .with_region("us-west-2")
            .with_temperature(0.5)
            .with_max_tokens(1000)
            .with_stop_sequence("STOP");

        assert_eq!(
            bedrock.config.model,
            BedrockModel::AnthropicClaudeV2
        );
        assert_eq!(bedrock.config.region, Some("us-west-2".to_string()));
        assert_eq!(bedrock.config.temperature, Some(0.5));
        assert_eq!(bedrock.config.max_tokens, Some(1000));
        assert_eq!(bedrock.config.stop_sequences, vec!["STOP"]);
    }

    #[test]
    fn test_format_prompt_anthropic() {
        let bedrock = Bedrock::default().with_model(BedrockModel::AnthropicClaudeV2);

        let prompt = "What is the capital of France?";
        let formatted = bedrock.format_prompt(prompt);
        assert!(formatted.starts_with("\n\nHuman:"));
        assert!(formatted.ends_with("\n\nAssistant:"));

        let already_formatted = "\n\nHuman: Hello\n\nAssistant:";
        let formatted2 = bedrock.format_prompt(already_formatted);
        assert_eq!(formatted2, already_formatted);
    }

    #[test]
    fn test_format_prompt_other_providers() {
        let bedrock = Bedrock::default().with_model(BedrockModel::AmazonTitanTextExpress);

        let prompt = "What is the capital of France?";
        let formatted = bedrock.format_prompt(prompt);
        assert_eq!(formatted, prompt);
    }

    #[test]
    fn test_build_request_body_anthropic() {
        let bedrock = Bedrock::default()
            .with_model(BedrockModel::AnthropicClaudeV2)
            .with_temperature(0.8)
            .with_max_tokens(256)
            .with_stop_sequence("STOP");

        let body = bedrock.build_request_body("Test prompt").unwrap();

        assert!(body["prompt"].as_str().unwrap().contains("Test prompt"));
        assert_eq!(body["max_tokens_to_sample"].as_i64().unwrap(), 256);
        // Use approximate comparison for floating point
        let temp = body["temperature"].as_f64().unwrap();
        assert!((temp - 0.8).abs() < 0.01, "Temperature should be approximately 0.8, got {}", temp);
        assert_eq!(body["stop_sequences"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_build_request_body_amazon() {
        let bedrock = Bedrock::default()
            .with_model(BedrockModel::AmazonTitanTextExpress)
            .with_temperature(0.5)
            .with_max_tokens(512);

        let body = bedrock.build_request_body("Test prompt").unwrap();

        assert_eq!(body["inputText"].as_str().unwrap(), "Test prompt");
        assert_eq!(
            body["textGenerationConfig"]["maxTokenCount"]
                .as_i64()
                .unwrap(),
            512
        );
        // Use approximate comparison for floating point
        let temp = body["textGenerationConfig"]["temperature"]
            .as_f64()
            .unwrap();
        assert!((temp - 0.5).abs() < 0.01, "Temperature should be approximately 0.5, got {}", temp);
    }

    #[test]
    fn test_parse_response_anthropic() {
        let bedrock = Bedrock::default().with_model(BedrockModel::AnthropicClaudeV2);

        let response = json!({
            "completion": " Paris is the capital of France.",
            "stop_reason": "stop_sequence",
        });

        let response_bytes = serde_json::to_vec(&response).unwrap();
        let text = bedrock.parse_response(&response_bytes).unwrap();

        assert_eq!(text, " Paris is the capital of France.");
    }

    #[test]
    fn test_parse_response_amazon() {
        let bedrock = Bedrock::default().with_model(BedrockModel::AmazonTitanTextExpress);

        let response = json!({
            "results": [{
                "outputText": "Paris is the capital of France."
            }]
        });

        let response_bytes = serde_json::to_vec(&response).unwrap();
        let text = bedrock.parse_response(&response_bytes).unwrap();

        assert_eq!(text, "Paris is the capital of France.");
    }

    #[test]
    fn test_default_config() {
        let config = BedrockConfig::default();
        assert_eq!(config.region, Some("us-west-2".to_string()));
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(512));
        assert_eq!(config.model, BedrockModel::AnthropicClaude3Sonnet);
    }

    #[test]
    fn test_clone() {
        let bedrock1 = Bedrock::default()
            .with_model(BedrockModel::AnthropicClaudeV2)
            .with_temperature(0.9);

        let bedrock2 = bedrock1.clone();

        assert_eq!(bedrock2.config.model, BedrockModel::AnthropicClaudeV2);
        assert_eq!(bedrock2.config.temperature, Some(0.9));
    }
}