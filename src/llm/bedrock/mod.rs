//! AWS Bedrock LLM integration.
//!
//! [`Bedrock`] implements the [`LLM`] trait on top of the Bedrock
//! [Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html),
//! which gives one request/response shape across every Converse-capable model
//! (Anthropic Claude, Meta Llama, Mistral, Cohere Command, Amazon Titan, ...).
//!
//! Enable the `bedrock` feature to use it:
//!
//! ```toml
//! langchain-rust = { version = "...", features = ["bedrock"] }
//! ```
//!
//! Credentials and the default region are resolved by the AWS SDK's standard
//! chain (environment variables, `~/.aws/credentials`, instance profile, ...).
//! Calling [`Bedrock::with_region`] overrides the region.
//!
//! ```rust,no_run
//! use langchain_rust::language_models::llm::LLM;
//! use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let llm = Bedrock::default()
//!         .with_model(BedrockModel::AnthropicClaude35Haiku)
//!         .with_region("us-east-1")
//!         .with_temperature(0.7)
//!         .with_max_tokens(256);
//!
//!     let answer = llm.invoke("What is the capital of France?").await?;
//!     println!("{answer}");
//!     Ok(())
//! }
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::types::{
    ContentBlock, ConversationRole, InferenceConfiguration, Message as BedrockMessage,
    SystemContentBlock,
};
use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_smithy_types::{Document, Number};
use thiserror::Error;
use tokio::sync::OnceCell;

use crate::language_models::llm::LLM;
use crate::language_models::{GenerateResult, LLMError, TokenUsage};
use crate::schemas::messages::MessageType;
use crate::schemas::{Message, StreamData};

/// Errors specific to the Bedrock integration.
#[derive(Error, Debug)]
pub enum BedrockError {
    /// The request could not be built (e.g. a message builder rejected its input).
    #[error("Bedrock request error: {0}")]
    RequestError(String),

    /// The Bedrock API call failed.
    #[error("Bedrock invocation error: {0}")]
    InvocationError(String),

    /// The response contained no text content.
    #[error("Bedrock response contained no text output")]
    EmptyResponse,
}

/// Bedrock model identifiers.
///
/// Every variant here supports the Converse API. Use [`BedrockModel::Custom`]
/// for any other model or inference-profile ID (e.g. `us.anthropic.claude-...`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum BedrockModel {
    AnthropicClaudeV2,
    AnthropicClaudeInstantV1,
    AnthropicClaude3Sonnet,
    AnthropicClaude3Haiku,
    AnthropicClaude3Opus,
    #[default]
    AnthropicClaude35Haiku,
    AnthropicClaude4Sonnet,
    AnthropicClaude45Haiku,
    AnthropicClaude41Opus,
    AnthropicClaude45Opus,
    AnthropicClaude45Sonnet,
    AmazonTitanTextExpress,
    AmazonTitanTextLite,
    CohereCommand,
    CohereCommandLight,
    MetaLlama2Chat13B,
    MetaLlama2Chat70B,
    /// Any model or inference-profile ID accepted by Bedrock.
    Custom(String),
}

impl BedrockModel {
    /// The model ID string sent to Bedrock.
    pub fn model_id(&self) -> &str {
        match self {
            BedrockModel::AnthropicClaudeV2 => "anthropic.claude-v2",
            BedrockModel::AnthropicClaudeInstantV1 => "anthropic.claude-instant-v1",
            BedrockModel::AnthropicClaude3Sonnet => "anthropic.claude-3-sonnet-20240229-v1:0",
            BedrockModel::AnthropicClaude3Haiku => "anthropic.claude-3-haiku-20240307-v1:0",
            BedrockModel::AnthropicClaude3Opus => "anthropic.claude-3-opus-20240229-v1:0",
            BedrockModel::AnthropicClaude35Haiku => "anthropic.claude-3-5-haiku-20241022-v1:0",
            BedrockModel::AnthropicClaude4Sonnet => "anthropic.claude-sonnet-4-20250514-v1:0",
            BedrockModel::AnthropicClaude45Haiku => "anthropic.claude-haiku-4-5-20251001-v1:0",
            BedrockModel::AnthropicClaude41Opus => "anthropic.claude-opus-4-1-20250805-v1:0",
            BedrockModel::AnthropicClaude45Opus => "anthropic.claude-opus-4-5-20251101-v1:0",
            BedrockModel::AnthropicClaude45Sonnet => "anthropic.claude-sonnet-4-5-20250929-v1:0",
            BedrockModel::AmazonTitanTextExpress => "amazon.titan-text-express-v1",
            BedrockModel::AmazonTitanTextLite => "amazon.titan-text-lite-v1",
            BedrockModel::CohereCommand => "cohere.command-text-v14",
            BedrockModel::CohereCommandLight => "cohere.command-light-text-v14",
            BedrockModel::MetaLlama2Chat13B => "meta.llama2-13b-chat-v1",
            BedrockModel::MetaLlama2Chat70B => "meta.llama2-70b-chat-v1",
            BedrockModel::Custom(id) => id,
        }
    }
}

/// Generation settings for [`Bedrock`].
#[derive(Debug, Clone)]
pub struct BedrockConfig {
    /// AWS region. `None` defers to the SDK's default region chain.
    pub region: Option<String>,
    pub model: BedrockModel,
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Vec<String>,
    /// Extra model-specific parameters, sent as `additionalModelRequestFields`
    /// (for example `{"top_k": 40}` for Anthropic models).
    pub additional_model_request_fields: serde_json::Value,
}

impl Default for BedrockConfig {
    fn default() -> Self {
        Self {
            region: None,
            model: BedrockModel::default(),
            temperature: Some(0.7),
            max_tokens: Some(512),
            top_p: None,
            stop_sequences: Vec::new(),
            additional_model_request_fields: serde_json::Value::Null,
        }
    }
}

/// An [`LLM`] backed by the AWS Bedrock Converse API.
///
/// The underlying SDK client is built lazily on first use and shared across
/// clones, so a `Bedrock` can be cloned into chains and agents freely.
#[derive(Clone)]
pub struct Bedrock {
    client: Arc<OnceCell<BedrockClient>>,
    config: BedrockConfig,
}

impl Bedrock {
    pub fn new() -> Self {
        Self::with_config(BedrockConfig::default())
    }

    pub fn with_config(config: BedrockConfig) -> Self {
        Self {
            client: Arc::new(OnceCell::new()),
            config,
        }
    }

    /// Use an already-constructed SDK client instead of building one from the
    /// default config chain.
    pub fn with_client(mut self, client: BedrockClient) -> Self {
        self.client = Arc::new(OnceCell::new_with(Some(client)));
        self
    }

    /// Override the AWS region. Resets any lazily-built client.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.config.region = Some(region.into());
        self.client = Arc::new(OnceCell::new());
        self
    }

    pub fn with_model(mut self, model: BedrockModel) -> Self {
        self.config.model = model;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.config.top_p = Some(top_p);
        self
    }

    pub fn with_stop_sequence(mut self, stop: impl Into<String>) -> Self {
        self.config.stop_sequences.push(stop.into());
        self
    }

    /// Set model-specific parameters sent as `additionalModelRequestFields`.
    pub fn with_additional_model_request_fields(mut self, fields: serde_json::Value) -> Self {
        self.config.additional_model_request_fields = fields;
        self
    }

    pub fn config(&self) -> &BedrockConfig {
        &self.config
    }

    async fn client(&self) -> &BedrockClient {
        self.client
            .get_or_init(|| async {
                let mut loader = aws_config::defaults(BehaviorVersion::latest());
                if let Some(region) = &self.config.region {
                    loader = loader.region(Region::new(region.clone()));
                }
                BedrockClient::new(&loader.load().await)
            })
            .await
    }

    /// Split langchain messages into the Converse API's `system` blocks and
    /// `messages` list.
    fn to_converse_messages(
        messages: &[Message],
    ) -> Result<(Vec<SystemContentBlock>, Vec<BedrockMessage>), BedrockError> {
        let mut system = Vec::new();
        let mut converse = Vec::new();

        for msg in messages {
            let role = match msg.message_type {
                MessageType::SystemMessage => {
                    system.push(SystemContentBlock::Text(msg.content.clone()));
                    continue;
                }
                MessageType::AIMessage => ConversationRole::Assistant,
                MessageType::HumanMessage | MessageType::ToolMessage => ConversationRole::User,
            };
            let built = BedrockMessage::builder()
                .role(role)
                .content(ContentBlock::Text(msg.content.clone()))
                .build()
                .map_err(|e| BedrockError::RequestError(e.to_string()))?;
            converse.push(built);
        }

        Ok((system, converse))
    }

    fn inference_config(&self) -> InferenceConfiguration {
        let mut cfg = InferenceConfiguration::builder();
        if let Some(v) = self.config.max_tokens {
            cfg = cfg.max_tokens(v);
        }
        if let Some(v) = self.config.temperature {
            cfg = cfg.temperature(v);
        }
        if let Some(v) = self.config.top_p {
            cfg = cfg.top_p(v);
        }
        for stop in &self.config.stop_sequences {
            cfg = cfg.stop_sequences(stop.clone());
        }
        cfg.build()
    }
}

impl Default for Bedrock {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a `serde_json::Value` into the Smithy `Document` type used for
/// `additionalModelRequestFields`.
fn json_to_document(value: &serde_json::Value) -> Document {
    match value {
        serde_json::Value::Null => Document::Null,
        serde_json::Value::Bool(b) => Document::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                Document::Number(Number::PosInt(u))
            } else if let Some(i) = n.as_i64() {
                Document::Number(Number::NegInt(i))
            } else {
                Document::Number(Number::Float(n.as_f64().unwrap_or(0.0)))
            }
        }
        serde_json::Value::String(s) => Document::String(s.clone()),
        serde_json::Value::Array(items) => {
            Document::Array(items.iter().map(json_to_document).collect())
        }
        serde_json::Value::Object(map) => Document::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), json_to_document(v)))
                .collect(),
        ),
    }
}

#[async_trait]
impl LLM for Bedrock {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let (system, converse_messages) = Self::to_converse_messages(messages)?;

        let mut request = self
            .client()
            .await
            .converse()
            .model_id(self.config.model.model_id())
            .set_system(Some(system))
            .set_messages(Some(converse_messages))
            .inference_config(self.inference_config());

        if !self.config.additional_model_request_fields.is_null() {
            request = request.additional_model_request_fields(json_to_document(
                &self.config.additional_model_request_fields,
            ));
        }

        let response = request
            .send()
            .await
            .map_err(|e| BedrockError::InvocationError(e.into_service_error().to_string()))?;

        let generation = response
            .output()
            .and_then(|o| o.as_message().ok())
            .map(|m| {
                m.content()
                    .iter()
                    .filter_map(|c| c.as_text().ok())
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("")
            })
            .filter(|s| !s.is_empty())
            .ok_or(BedrockError::EmptyResponse)?;

        let tokens = response.usage().map(|u| TokenUsage {
            prompt_tokens: u.input_tokens() as u32,
            completion_tokens: u.output_tokens() as u32,
            total_tokens: u.total_tokens() as u32,
        });

        Ok(GenerateResult { generation, tokens })
    }

    async fn stream(
        &self,
        _messages: &[Message],
    ) -> Result<
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<StreamData, LLMError>> + Send>>,
        LLMError,
    > {
        Err(LLMError::OtherError(
            "Streaming is not yet implemented for Bedrock".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn model_ids() {
        assert_eq!(
            BedrockModel::AnthropicClaudeV2.model_id(),
            "anthropic.claude-v2"
        );
        assert_eq!(
            BedrockModel::AnthropicClaude35Haiku.model_id(),
            "anthropic.claude-3-5-haiku-20241022-v1:0"
        );
        assert_eq!(
            BedrockModel::AmazonTitanTextExpress.model_id(),
            "amazon.titan-text-express-v1"
        );
        assert_eq!(
            BedrockModel::Custom("us.anthropic.claude-sonnet-4-20250514-v1:0".into()).model_id(),
            "us.anthropic.claude-sonnet-4-20250514-v1:0"
        );
    }

    #[test]
    fn builder_sets_config() {
        let llm = Bedrock::default()
            .with_model(BedrockModel::AnthropicClaude3Haiku)
            .with_region("us-west-2")
            .with_temperature(0.5)
            .with_max_tokens(1000)
            .with_top_p(0.9)
            .with_stop_sequence("STOP")
            .with_additional_model_request_fields(json!({"top_k": 40}));

        let cfg = llm.config();
        assert_eq!(cfg.model, BedrockModel::AnthropicClaude3Haiku);
        assert_eq!(cfg.region.as_deref(), Some("us-west-2"));
        assert_eq!(cfg.temperature, Some(0.5));
        assert_eq!(cfg.max_tokens, Some(1000));
        assert_eq!(cfg.top_p, Some(0.9));
        assert_eq!(cfg.stop_sequences, vec!["STOP"]);
        assert_eq!(cfg.additional_model_request_fields, json!({"top_k": 40}));
    }

    #[test]
    fn default_config() {
        let cfg = BedrockConfig::default();
        assert_eq!(cfg.region, None);
        assert_eq!(cfg.model, BedrockModel::AnthropicClaude35Haiku);
        assert_eq!(cfg.temperature, Some(0.7));
        assert_eq!(cfg.max_tokens, Some(512));
    }

    #[test]
    fn clone_shares_config() {
        let a = Bedrock::default()
            .with_model(BedrockModel::AnthropicClaudeV2)
            .with_temperature(0.9);
        let b = a.clone();
        assert_eq!(b.config().model, BedrockModel::AnthropicClaudeV2);
        assert_eq!(b.config().temperature, Some(0.9));
        assert!(Arc::ptr_eq(&a.client, &b.client));
    }

    #[test]
    fn converse_messages_split_system_and_roles() {
        let messages = [
            Message::new_system_message("be brief"),
            Message::new_human_message("hi"),
            Message::new_ai_message("hello"),
            Message::new_human_message("bye"),
        ];
        let (system, converse) = Bedrock::to_converse_messages(&messages).unwrap();

        assert_eq!(system.len(), 1);
        assert_eq!(system[0].as_text().unwrap(), "be brief");

        let roles: Vec<_> = converse.iter().map(|m| m.role().clone()).collect();
        assert_eq!(
            roles,
            vec![
                ConversationRole::User,
                ConversationRole::Assistant,
                ConversationRole::User
            ]
        );
        assert_eq!(converse[1].content()[0].as_text().unwrap(), "hello");
    }

    #[test]
    fn inference_config_carries_settings() {
        let llm = Bedrock::default()
            .with_temperature(0.2)
            .with_max_tokens(42)
            .with_top_p(0.8)
            .with_stop_sequence("END");
        let cfg = llm.inference_config();
        assert_eq!(cfg.max_tokens(), Some(42));
        assert_eq!(cfg.temperature(), Some(0.2));
        assert_eq!(cfg.top_p(), Some(0.8));
        assert_eq!(cfg.stop_sequences(), &["END".to_string()]);
    }

    #[test]
    fn json_to_document_round_trips_shapes() {
        let doc = json_to_document(&json!({
            "top_k": 40,
            "neg": -1,
            "f": 1.5,
            "s": "x",
            "b": true,
            "n": null,
            "arr": [1, "two"]
        }));
        let Document::Object(map) = doc else {
            panic!("expected object")
        };
        assert_eq!(map["top_k"], Document::Number(Number::PosInt(40)));
        assert_eq!(map["neg"], Document::Number(Number::NegInt(-1)));
        assert_eq!(map["f"], Document::Number(Number::Float(1.5)));
        assert_eq!(map["s"], Document::String("x".into()));
        assert_eq!(map["b"], Document::Bool(true));
        assert_eq!(map["n"], Document::Null);
        assert_eq!(
            map["arr"],
            Document::Array(vec![
                Document::Number(Number::PosInt(1)),
                Document::String("two".into())
            ])
        );
    }
}
