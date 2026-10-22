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
//! Tool calling and streaming are supported. Tools passed through
//! [`CallOptions::functions`] (which `OpenAiToolAgent` does for you) are sent as a
//! Converse `toolConfig`, and tool calls come back in the same OpenAI-like format as
//! the other providers.
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

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::types::{
    AutoToolChoice, ContentBlock, ContentBlockDelta, ContentBlockStart, ConversationRole,
    ConverseStreamOutput, ImageBlock, ImageFormat, ImageSource as BedrockImageSource,
    InferenceConfiguration, Message as BedrockMessage, SpecificToolChoice, SystemContentBlock,
    Tool, ToolChoice, ToolConfiguration, ToolInputSchema, ToolResultBlock, ToolResultContentBlock,
    ToolSpecification, ToolUseBlock,
};
use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_smithy_types::error::display::DisplayErrorContext;
use aws_smithy_types::{Blob, Document, Number};
use futures::Stream;
use serde_json::{json, Value};
use thiserror::Error;
use tokio::sync::OnceCell;

use crate::language_models::llm::LLM;
use crate::language_models::options::CallOptions;
use crate::language_models::{
    box_llm_events, GenerateResult, LLMError, LLMEvent, LLMEventStream, TokenUsage,
};
use crate::observability::{traced_event_stream, traced_generate};

/// `provider` field of the `llm.generate` span.
const PROVIDER: &str = "bedrock";
use crate::schemas::messages::{ImageSource, MessageType};
use crate::schemas::{FunctionCallBehavior, FunctionDefinition, Message, StreamData, ToolCall};

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
    options: CallOptions,
}

impl Bedrock {
    pub fn new() -> Self {
        Self::with_config(BedrockConfig::default())
    }

    pub fn with_config(config: BedrockConfig) -> Self {
        Self {
            client: Arc::new(OnceCell::new()),
            config,
            options: CallOptions::default(),
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

    /// Set call options (tools, tool choice, streaming callback, ...). Values set here
    /// override the matching [`BedrockConfig`] settings.
    pub fn with_options(mut self, options: CallOptions) -> Self {
        self.options = options;
        self
    }

    /// Tools the model may call. Tool calls come back in `generation` in the OpenAI-like
    /// format used across the crate, so `OpenAiToolAgent` works with Bedrock.
    pub fn with_tools(mut self, tools: Vec<FunctionDefinition>) -> Self {
        self.options.functions = Some(tools);
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
    ///
    /// AI tool calls become `toolUse` blocks and tool results become `toolResult`
    /// blocks in a `user` turn (Converse has no tool role). Consecutive turns with the
    /// same role are merged, because Converse requires alternating roles and wants all
    /// results of a parallel tool call in a single user message.
    fn to_converse_messages(
        messages: &[Message],
    ) -> Result<(Vec<SystemContentBlock>, Vec<BedrockMessage>), BedrockError> {
        let mut system = Vec::new();
        let mut turns: Vec<(ConversationRole, Vec<ContentBlock>)> = Vec::new();

        for msg in messages {
            let (role, blocks) = match msg.message_type {
                MessageType::SystemMessage => {
                    system.push(SystemContentBlock::Text(msg.content.clone()));
                    continue;
                }
                MessageType::AIMessage => (ConversationRole::Assistant, assistant_blocks(msg)?),
                MessageType::HumanMessage => (ConversationRole::User, user_blocks(msg)?),
                MessageType::ToolMessage => {
                    let result = ToolResultBlock::builder()
                        .tool_use_id(msg.id.clone().unwrap_or_default())
                        .content(ToolResultContentBlock::Text(msg.content.clone()))
                        .build()
                        .map_err(request_error)?;
                    (
                        ConversationRole::User,
                        vec![ContentBlock::ToolResult(result)],
                    )
                }
            };
            match turns.last_mut() {
                Some((last_role, last_blocks)) if *last_role == role => last_blocks.extend(blocks),
                _ => turns.push((role, blocks)),
            }
        }

        let converse = turns
            .into_iter()
            .map(|(role, blocks)| {
                BedrockMessage::builder()
                    .role(role)
                    .set_content(Some(blocks))
                    .build()
                    .map_err(request_error)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok((system, converse))
    }

    /// Inference settings. Call options (set by chains/agents) win over [`BedrockConfig`].
    fn inference_config(&self) -> InferenceConfiguration {
        let mut stop_sequences = self.config.stop_sequences.clone();
        if let Some(words) = &self.options.stop_words {
            stop_sequences.extend(words.iter().cloned());
        }
        InferenceConfiguration::builder()
            .set_max_tokens(
                self.options
                    .max_tokens
                    .map(|v| v as i32)
                    .or(self.config.max_tokens),
            )
            .set_temperature(self.options.temperature.or(self.config.temperature))
            .set_top_p(self.options.top_p.or(self.config.top_p))
            .set_stop_sequences((!stop_sequences.is_empty()).then_some(stop_sequences))
            .build()
    }

    fn tool_config(&self) -> Result<Option<ToolConfiguration>, BedrockError> {
        let Some(functions) = self.options.functions.as_ref().filter(|f| !f.is_empty()) else {
            return Ok(None);
        };
        // Converse has no "none" tool choice; leaving the tools out is the equivalent.
        if matches!(
            self.options.function_call_behavior,
            Some(FunctionCallBehavior::None)
        ) {
            return Ok(None);
        }

        let tools = functions
            .iter()
            .map(|f| {
                ToolSpecification::builder()
                    .name(&f.name)
                    .set_description((!f.description.is_empty()).then(|| f.description.clone()))
                    .input_schema(ToolInputSchema::Json(json_to_document(&f.parameters)))
                    .build()
                    .map(Tool::ToolSpec)
                    .map_err(request_error)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let choice = match &self.options.function_call_behavior {
            Some(FunctionCallBehavior::Named(name)) => Some(ToolChoice::Tool(
                SpecificToolChoice::builder()
                    .name(name)
                    .build()
                    .map_err(request_error)?,
            )),
            Some(FunctionCallBehavior::Auto) => {
                Some(ToolChoice::Auto(AutoToolChoice::builder().build()))
            }
            _ => None,
        };

        ToolConfiguration::builder()
            .set_tools(Some(tools))
            .set_tool_choice(choice)
            .build()
            .map(Some)
            .map_err(request_error)
    }

    fn additional_fields(&self) -> Option<Document> {
        let fields = &self.config.additional_model_request_fields;
        (!fields.is_null()).then(|| json_to_document(fields))
    }

    async fn generate_blocking(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let (system, converse_messages) = Self::to_converse_messages(messages)?;
        let response = self
            .client()
            .await
            .converse()
            .model_id(self.config.model.model_id())
            .set_system(Some(system))
            .set_messages(Some(converse_messages))
            .inference_config(self.inference_config())
            .set_tool_config(self.tool_config()?)
            .set_additional_model_request_fields(self.additional_fields())
            .send()
            .await
            .map_err(invocation_error)?;

        let blocks = response
            .output()
            .and_then(|o| o.as_message().ok())
            .map(|m| m.content())
            .unwrap_or_default();
        let result = result_from_blocks(blocks.iter().filter_map(|block| match block {
            ContentBlock::Text(text) => Some(Block::Text(text.clone())),
            ContentBlock::ToolUse(tool) => Some(Block::ToolUse {
                id: tool.tool_use_id().to_string(),
                name: tool.name().to_string(),
                input: document_to_json(tool.input()).to_string(),
            }),
            _ => None,
        }))?;

        Ok(GenerateResult {
            tokens: response.usage().map(token_usage),
            ..result
        })
    }

    async fn converse_stream(
        &self,
        messages: &[Message],
    ) -> Result<impl Stream<Item = Result<ConverseStreamOutput, LLMError>> + Send, LLMError> {
        let (system, converse_messages) = Self::to_converse_messages(messages)?;
        let output = self
            .client()
            .await
            .converse_stream()
            .model_id(self.config.model.model_id())
            .set_system(Some(system))
            .set_messages(Some(converse_messages))
            .inference_config(self.inference_config())
            .set_tool_config(self.tool_config()?)
            .set_additional_model_request_fields(self.additional_fields())
            .send()
            .await
            .map_err(invocation_error)?;

        let mut receiver = output.stream;
        Ok(async_stream::stream! {
            loop {
                match receiver.recv().await {
                    Ok(Some(event)) => yield Ok(event),
                    Ok(None) => break,
                    Err(e) => {
                        yield Err(invocation_error(e).into());
                        break;
                    }
                }
            }
        })
    }

    /// Streams the response, forwarding text deltas to `streaming_func` while rebuilding
    /// the full content blocks (tool input arrives as JSON fragments).
    async fn generate_streaming(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        use futures::StreamExt;

        let func = self
            .options
            .streaming_func
            .clone()
            .ok_or_else(|| LLMError::OtherError("streaming_func not set".into()))?;
        let events = self.converse_stream(messages).await?;
        futures::pin_mut!(events);

        let mut acc = StreamAccumulator::default();
        while let Some(event) = events.next().await {
            let event = event?;
            if let Some(text) = acc.push(&event) {
                let mut func = func.lock().await;
                let _ = func(text).await;
            }
        }
        Ok(GenerateResult {
            tokens: acc.usage,
            ..result_from_blocks(acc.blocks.into_values())?
        })
    }
}

impl Default for Bedrock {
    fn default() -> Self {
        Self::new()
    }
}

fn request_error(e: impl std::fmt::Display) -> BedrockError {
    BedrockError::RequestError(e.to_string())
}

/// A user turn: one `image` block per image, then the text. Converse takes image
/// bytes only, so base64 data is decoded and http(s) URLs are rejected.
fn user_blocks(msg: &Message) -> Result<Vec<ContentBlock>, BedrockError> {
    let mut blocks = Vec::new();
    for image in msg.images.iter().flatten() {
        let ImageSource::Base64 { media_type, data } = image.source() else {
            return Err(BedrockError::RequestError(format!(
                "Bedrock only accepts inline images; download {} and pass it as base64 or a data: URL",
                image.image_url
            )));
        };
        let format = match media_type {
            Some("image/png") => ImageFormat::Png,
            Some("image/gif") => ImageFormat::Gif,
            Some("image/webp") => ImageFormat::Webp,
            Some("image/jpeg") | Some("image/jpg") | None => ImageFormat::Jpeg,
            Some(other) => {
                return Err(BedrockError::RequestError(format!(
                    "unsupported image type {other} (Bedrock takes png, jpeg, gif or webp)"
                )))
            }
        };
        let bytes = aws_smithy_types::base64::decode(data)
            .map_err(|e| request_error(format!("invalid base64 image: {e}")))?;
        let block = ImageBlock::builder()
            .format(format)
            .source(BedrockImageSource::Bytes(Blob::new(bytes)))
            .build()
            .map_err(request_error)?;
        blocks.push(ContentBlock::Image(block));
    }
    if !msg.content.is_empty() || blocks.is_empty() {
        blocks.push(ContentBlock::Text(msg.content.clone()));
    }
    Ok(blocks)
}

/// Keeps the full error chain (the top-level SDK error alone is just "service error").
fn invocation_error<E: std::error::Error>(e: E) -> BedrockError {
    BedrockError::InvocationError(DisplayErrorContext(e).to_string())
}

fn token_usage(u: &aws_sdk_bedrockruntime::types::TokenUsage) -> TokenUsage {
    TokenUsage {
        prompt_tokens: u.input_tokens() as u32,
        completion_tokens: u.output_tokens() as u32,
        total_tokens: u.total_tokens() as u32,
    }
}

/// Assistant content: the text (if any) followed by one `toolUse` block per tool call
/// stored in the OpenAI-like format used across the crate.
fn assistant_blocks(msg: &Message) -> Result<Vec<ContentBlock>, BedrockError> {
    let calls = msg.parse_tool_calls();

    let mut blocks = Vec::new();
    if !msg.content.trim().is_empty() || calls.is_empty() {
        blocks.push(ContentBlock::Text(msg.content.clone()));
    }
    for call in calls {
        let block = ToolUseBlock::builder()
            .tool_use_id(&call.id)
            .name(&call.name)
            .input(json_to_document(&call.arguments_value()))
            .build()
            .map_err(request_error)?;
        blocks.push(ContentBlock::ToolUse(block));
    }
    Ok(blocks)
}

/// A provider-agnostic view of the response content we care about.
#[derive(Debug, Clone, PartialEq)]
enum Block {
    Text(String),
    /// `input` is the raw JSON string of the tool arguments.
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
}

/// The result: typed tool calls (mirrored into `generation` as OpenAI-like JSON, which is what
/// `OpenAiToolAgent` reads) when the model asked for tools, otherwise the joined text.
fn result_from_blocks(blocks: impl IntoIterator<Item = Block>) -> Result<GenerateResult, LLMError> {
    let mut text = String::new();
    let mut calls = Vec::new();
    for block in blocks {
        match block {
            Block::Text(t) => text.push_str(&t),
            Block::ToolUse { id, name, input } => calls.push(ToolCall::new(
                id,
                name,
                if input.trim().is_empty() {
                    "{}".to_string()
                } else {
                    input
                },
            )),
        }
    }
    if calls.is_empty() && text.is_empty() {
        return Err(BedrockError::EmptyResponse.into());
    }
    Ok(GenerateResult {
        generation: text,
        ..Default::default()
    }
    .with_tool_calls(calls))
}

/// Rebuilds content blocks and usage from `ConverseStream` events.
#[derive(Default)]
struct StreamAccumulator {
    blocks: BTreeMap<i32, Block>,
    usage: Option<TokenUsage>,
}

impl StreamAccumulator {
    /// Applies an event; returns the text delta, if the event carried one.
    fn push(&mut self, event: &ConverseStreamOutput) -> Option<String> {
        match event {
            ConverseStreamOutput::ContentBlockStart(start) => {
                if let Some(ContentBlockStart::ToolUse(tool)) = start.start() {
                    self.blocks.insert(
                        start.content_block_index(),
                        Block::ToolUse {
                            id: tool.tool_use_id().to_string(),
                            name: tool.name().to_string(),
                            input: String::new(),
                        },
                    );
                }
                None
            }
            ConverseStreamOutput::ContentBlockDelta(delta) => {
                let index = delta.content_block_index();
                match delta.delta()? {
                    ContentBlockDelta::Text(text) => {
                        match self
                            .blocks
                            .entry(index)
                            .or_insert_with(|| Block::Text(String::new()))
                        {
                            Block::Text(t) => t.push_str(text),
                            Block::ToolUse { .. } => return None,
                        }
                        Some(text.clone())
                    }
                    ContentBlockDelta::ToolUse(tool) => {
                        if let Some(Block::ToolUse { input, .. }) = self.blocks.get_mut(&index) {
                            input.push_str(tool.input());
                        }
                        None
                    }
                    _ => None,
                }
            }
            ConverseStreamOutput::Metadata(meta) => {
                self.usage = meta.usage().map(token_usage);
                None
            }
            _ => None,
        }
    }
}

/// Convert a `serde_json::Value` into the Smithy `Document` type used for
/// `additionalModelRequestFields`, tool schemas and tool input.
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

/// The inverse of [`json_to_document`], for tool input returned by the model.
fn document_to_json(doc: &Document) -> Value {
    match doc {
        Document::Null => Value::Null,
        Document::Bool(b) => Value::Bool(*b),
        Document::Number(Number::PosInt(u)) => json!(u),
        Document::Number(Number::NegInt(i)) => json!(i),
        Document::Number(Number::Float(f)) => json!(f),
        Document::String(s) => Value::String(s.clone()),
        Document::Array(items) => Value::Array(items.iter().map(document_to_json).collect()),
        Document::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), document_to_json(v)))
                .collect(),
        ),
    }
}

#[async_trait]
impl LLM for Bedrock {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        traced_generate(PROVIDER, self.config.model.model_id(), async {
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
        use futures::StreamExt;

        let events = self.converse_stream(messages).await?;
        let mut acc = StreamAccumulator::default();
        let stream = events.filter_map(move |event| {
            let item = match event {
                Err(e) => Some(Err(e)),
                Ok(event) => {
                    let text = acc.push(&event);
                    match event {
                        ConverseStreamOutput::Metadata(_) => Some(Ok(StreamData::new(
                            json!({"type": "metadata"}),
                            acc.usage.clone(),
                            "",
                        ))),
                        _ => text.map(|t| {
                            Ok(StreamData::new(
                                json!({"type": "content_block_delta", "text": t}),
                                None,
                                t,
                            ))
                        }),
                    }
                }
            };
            futures::future::ready(item)
        });
        Ok(Box::pin(stream))
    }

    fn generate_stream<'a>(&'a self, messages: &'a [Message]) -> LLMEventStream<'a> {
        use futures::StreamExt;

        let events = box_llm_events(async_stream::try_stream! {
            let events = self.converse_stream(messages).await?;
            futures::pin_mut!(events);

            let mut acc = StreamAccumulator::default();
            while let Some(event) = events.next().await {
                if let Some(text) = acc.push(&event?) {
                    yield LLMEvent::Text(text);
                }
            }
            let tokens = acc.usage;
            yield LLMEvent::Done(GenerateResult {
                tokens,
                ..result_from_blocks(acc.blocks.into_values())?
            });
        });
        traced_event_stream(PROVIDER, self.config.model.model_id(), events)
    }

    fn add_options(&mut self, options: CallOptions) {
        self.options.merge_options(options);
    }
}

#[cfg(test)]
mod mock_tests;

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
    fn user_images_become_image_blocks() {
        use crate::schemas::messages::ImageContent;
        // Real PNG signature, so the media type is sniffed from the bytes.
        const PNG: [u8; 9] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 7];
        let png = aws_smithy_types::base64::encode(PNG);
        let message = Message::new_human_message("what is it?").with_images(vec![
            ImageContent::from(png.as_str()),
            ImageContent::from("data:image/webp;base64,AAEC"),
            ImageContent::from("data:;base64,AAEC"),
        ]);
        let (_, converse) = Bedrock::to_converse_messages(&[message]).unwrap();
        let blocks = converse[0].content();
        assert_eq!(blocks.len(), 4);

        let image = |i: usize| blocks[i].as_image().unwrap();
        assert_eq!(image(0).format(), &ImageFormat::Png);
        assert_eq!(
            image(0).source().unwrap().as_bytes().unwrap().as_ref(),
            &PNG
        );
        assert_eq!(image(1).format(), &ImageFormat::Webp);
        assert_eq!(
            image(1).source().unwrap().as_bytes().unwrap().as_ref(),
            &[0, 1, 2]
        );
        // Unknown media type falls back to jpeg.
        assert_eq!(image(2).format(), &ImageFormat::Jpeg);
        assert_eq!(blocks[3].as_text().unwrap(), "what is it?");

        // Image-only turn: no empty text block.
        let only = Message::new_human_message("")
            .with_images(vec![ImageContent::from("data:image/gif;base64,AAEC")]);
        let (_, converse) = Bedrock::to_converse_messages(&[only]).unwrap();
        assert_eq!(converse[0].content().len(), 1);
        assert_eq!(
            converse[0].content()[0].as_image().unwrap().format(),
            &ImageFormat::Gif
        );
    }

    #[test]
    fn user_images_errors() {
        use crate::schemas::messages::ImageContent;
        let cases = [
            ("https://example.com/cat.png", "download"),
            ("data:image/tiff;base64,AAEC", "image/tiff"),
            ("data:image/png;base64,***", "invalid base64"),
        ];
        for (url, expected) in cases {
            let m = Message::new_human_message("x").with_images(vec![ImageContent::from(url)]);
            let err = Bedrock::to_converse_messages(&[m]).unwrap_err();
            assert!(err.to_string().contains(expected), "{url}: {err}");
        }
    }

    #[test]
    fn parallel_tool_results_share_one_user_turn() {
        let calls = json!([
            {"id": "a", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "t2", "arguments": "x"}}
        ]);
        let messages = [
            Message::new_human_message("go"),
            Message::new_ai_message("").with_tool_calls(calls),
            Message::new_tool_message("r1", "a"),
            Message::new_tool_message("r2", "b"),
        ];
        let (_, converse) = Bedrock::to_converse_messages(&messages).unwrap();
        assert_eq!(converse.len(), 3);

        let assistant = converse[1].content();
        assert_eq!(assistant.len(), 2, "no empty text block before tool calls");
        let second = assistant[1].as_tool_use().unwrap();
        assert_eq!(second.tool_use_id(), "b");
        assert_eq!(document_to_json(second.input()), json!({"input": "x"}));

        let results: Vec<_> = converse[2]
            .content()
            .iter()
            .map(|b| b.as_tool_result().unwrap().tool_use_id())
            .collect();
        assert_eq!(results, vec!["a", "b"]);
    }

    #[test]
    fn call_options_override_config_and_none_drops_tools() {
        let mut llm = Bedrock::default()
            .with_max_tokens(10)
            .with_stop_sequence("A")
            .with_tools(vec![FunctionDefinition::new("t", "", json!({}))]);
        llm.add_options(
            CallOptions::new()
                .with_max_tokens(99)
                .with_stop_words(vec!["B".into()]),
        );
        let cfg = llm.inference_config();
        assert_eq!(cfg.max_tokens(), Some(99));
        assert_eq!(cfg.stop_sequences(), &["A".to_string(), "B".to_string()]);
        assert!(llm.tool_config().unwrap().is_some());

        llm.add_options(CallOptions::new().with_function_call_behavior(FunctionCallBehavior::None));
        assert!(llm.tool_config().unwrap().is_none());
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
