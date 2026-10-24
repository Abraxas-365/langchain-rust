use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::language_models::{GenerateResult, ThinkingBlock, ThinkingConfig, TokenUsage};
use crate::schemas::messages::ImageSource;
use crate::schemas::{CacheControl, FunctionCallBehavior, FunctionDefinition, Message, ToolCall};

/// A message in the Anthropic Messages API. `content` is either a plain string or a list of
/// content blocks (`text`, `tool_use`, `tool_result`, ...).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(crate) struct ClaudeMessage {
    pub role: String,
    pub content: Value,
}

impl ClaudeMessage {
    pub fn new<S: Into<String>>(role: S, content: Value) -> Self {
        Self {
            role: role.into(),
            content,
        }
    }

    pub fn text<S: Into<String>>(role: S, text: &str) -> Self {
        Self::new(role, Value::String(text.to_string()))
    }

    /// A user turn. With images it becomes a list of `image` blocks followed by the
    /// text (Anthropic recommends images before the question).
    pub fn user(message: &Message) -> Self {
        let images = message.images.as_deref().unwrap_or_default();
        if images.is_empty() {
            return Self::text("user", &message.content);
        }
        let mut blocks: Vec<Value> = images
            .iter()
            .map(|image| {
                let source = match image.source() {
                    ImageSource::Url(url) => json!({"type": "url", "url": url}),
                    ImageSource::Base64 { media_type, data } => json!({
                        "type": "base64",
                        "media_type": media_type.unwrap_or("image/jpeg"),
                        "data": data,
                    }),
                };
                json!({"type": "image", "source": source})
            })
            .collect();
        if !message.content.is_empty() {
            blocks.push(json!({"type": "text", "text": message.content}));
        }
        Self::new("user", Value::Array(blocks))
    }

    /// An assistant turn. Thinking blocks (needed when Claude thought before calling
    /// tools) come first, then the text, then tool calls (stored in the OpenAI-like format
    /// used across the crate) as `tool_use` blocks.
    pub fn assistant(message: &Message) -> Self {
        let calls = message.parse_tool_calls();
        let mut blocks: Vec<Value> = thinking_blocks(message).collect();
        if calls.is_empty() && blocks.is_empty() {
            return Self::text("assistant", &message.content);
        }

        if !message.content.trim().is_empty() {
            blocks.push(json!({"type": "text", "text": message.content}));
        }
        for call in calls {
            blocks.push(json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments_value(),
            }));
        }
        Self::new("assistant", Value::Array(blocks))
    }

    /// The result of a tool call. Anthropic expects these as `tool_result` blocks inside a
    /// `user` message (there is no `tool` role).
    pub fn tool_result(message: &Message) -> Self {
        Self::new(
            "user",
            json!([{
                "type": "tool_result",
                "tool_use_id": message.id.clone().unwrap_or_default(),
                "content": message.content,
            }]),
        )
    }

    fn is_tool_result(&self) -> bool {
        self.role == "user"
            && self.content.as_array().is_some_and(|blocks| {
                !blocks.is_empty()
                    && blocks
                        .iter()
                        .all(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
            })
    }
}

/// Builds the Anthropic message list. Consecutive tool results are merged into one `user`
/// message, as the API requires all results for a parallel tool call in the same turn.
pub(crate) fn to_claude_messages(messages: &[&Message]) -> Vec<ClaudeMessage> {
    use crate::schemas::MessageType;

    let mut out: Vec<ClaudeMessage> = Vec::new();
    for message in messages {
        let mut next = match message.message_type {
            MessageType::AIMessage => ClaudeMessage::assistant(message),
            MessageType::HumanMessage | MessageType::SystemMessage => ClaudeMessage::user(message),
            MessageType::ToolMessage => ClaudeMessage::tool_result(message),
        };
        if let Some(cache_control) = message.cache_control {
            next.content = with_cache_breakpoint(std::mem::take(&mut next.content), cache_control);
        }
        match out.last_mut() {
            Some(prev) if prev.is_tool_result() && next.is_tool_result() => {
                if let (Some(prev_blocks), Value::Array(blocks)) =
                    (prev.content.as_array_mut(), next.content)
                {
                    prev_blocks.extend(blocks);
                }
            }
            _ => out.push(next),
        }
    }
    out
}

/// Puts `cache_control` on the last block of a message's content, so the prompt up to and
/// including it is cached. Plain-string content becomes one text block. Thinking blocks and
/// blank text can't carry a breakpoint, so they are skipped (nothing cacheable: unchanged).
fn with_cache_breakpoint(content: Value, cache_control: CacheControl) -> Value {
    let mut blocks = match content {
        Value::String(text) if text.is_empty() => return Value::String(text),
        Value::String(text) => vec![json!({"type": "text", "text": text})],
        Value::Array(blocks) => blocks,
        other => return other,
    };
    let target = blocks.iter_mut().rev().find(|b| {
        let kind = b.get("type").and_then(Value::as_str).unwrap_or_default();
        let empty_text = kind == "text" && b["text"].as_str().is_none_or(|t| t.trim().is_empty());
        !matches!(kind, "thinking" | "redacted_thinking") && !empty_text
    });
    if let Some(Value::Object(block)) = target {
        block.insert("cache_control".into(), cache_control.to_anthropic());
    }
    Value::Array(blocks)
}

/// The `system` parameter: the system messages joined into one string, or — when any of them
/// is marked for caching — one text block per message so the breakpoint lands where asked.
pub(crate) fn system_param(system_messages: &[&Message]) -> Option<Value> {
    if system_messages.is_empty() {
        return None;
    }
    if system_messages.iter().all(|m| m.cache_control.is_none()) {
        let joined: Vec<&str> = system_messages.iter().map(|m| m.content.as_str()).collect();
        return Some(Value::String(joined.join("\n\n")));
    }
    let blocks = system_messages
        .iter()
        .filter(|m| !m.content.is_empty())
        .map(|m| {
            let mut block = json!({"type": "text", "text": m.content});
            if let Some(cache_control) = m.cache_control {
                block["cache_control"] = cache_control.to_anthropic();
            }
            block
        })
        .collect();
    Some(Value::Array(blocks))
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(crate) struct ClaudeTool {
    pub name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
    pub input_schema: Value,
}

impl From<&FunctionDefinition> for ClaudeTool {
    fn from(f: &FunctionDefinition) -> Self {
        Self {
            name: f.name.clone(),
            description: f.description.clone(),
            input_schema: f.parameters.clone(),
        }
    }
}

pub(crate) fn tool_choice(behavior: &FunctionCallBehavior) -> Value {
    match behavior {
        FunctionCallBehavior::Auto => json!({"type": "auto"}),
        FunctionCallBehavior::None => json!({"type": "none"}),
        FunctionCallBehavior::Named(name) => json!({"type": "tool", "name": name}),
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Payload {
    pub model: String,
    pub messages: Vec<ClaudeMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ClaudeTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<Value>,
    /// Automatic prompt caching: the API puts the breakpoint on the last cacheable block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<Value>,
}

/// What a model accepts for reasoning, guessed from its name (as langchaingo does).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Capabilities {
    /// Extended thinking at all.
    pub thinking: bool,
    /// Only `{"type": "adaptive"}`; budgets are rejected.
    pub adaptive: bool,
    /// `output_config.effort`.
    pub effort: bool,
}

const ADAPTIVE: Capabilities = Capabilities {
    thinking: true,
    adaptive: true,
    effort: true,
};
const BUDGET: Capabilities = Capabilities {
    thinking: true,
    adaptive: false,
    effort: false,
};
const BUDGET_EFFORT: Capabilities = Capabilities {
    thinking: true,
    adaptive: false,
    effort: true,
};
const LEGACY: Capabilities = Capabilities {
    thinking: false,
    adaptive: false,
    effort: false,
};

/// First match wins, so more specific names come first.
const CAPABILITY_RULES: &[(&str, Capabilities)] = &[
    ("claude-fable", ADAPTIVE),
    ("claude-opus-4-7", ADAPTIVE),
    ("claude-opus-4-8", ADAPTIVE),
    ("claude-opus-4-6", BUDGET_EFFORT),
    ("claude-sonnet-4-6", BUDGET_EFFORT),
    ("claude-4", BUDGET),
    ("claude-3-7", BUDGET),
    ("claude-3.7", BUDGET),
    ("claude-3", LEGACY),
    ("claude-2", LEGACY),
    ("claude-instant", LEGACY),
];

/// 4.x families and the highest minor version whose behaviour is known; later minors are
/// assumed to be newest-generation (adaptive) models.
const FAMILY_RULES: &[(&str, u32)] = &[
    ("claude-opus-4", 8),
    ("claude-sonnet-4", 6),
    ("claude-haiku-4", 5),
];

pub(crate) fn model_capabilities(model: &str) -> Capabilities {
    let model = model.to_lowercase();
    if let Some((_, caps)) = CAPABILITY_RULES
        .iter()
        .find(|(name, _)| model.contains(name))
    {
        return *caps;
    }
    for (prefix, last_listed) in FAMILY_RULES {
        if let Some(minor) = family_minor(&model, prefix) {
            return if minor > *last_listed {
                ADAPTIVE
            } else {
                BUDGET
            };
        }
    }
    // Not a Claude name (gateway alias, compatible endpoint): keep the long-standing
    // budget contract. An unknown Claude name is most likely a newer model.
    if !model.contains("claude") {
        return BUDGET;
    }
    ADAPTIVE
}

/// The minor version in names like `claude-opus-4-5-20251101`; `Some(0)` for the bare
/// family name or a dated `.0` snapshot, `None` when `model` is not in the family.
fn family_minor(model: &str, prefix: &str) -> Option<u32> {
    let rest = &model[model.find(prefix)? + prefix.len()..];
    let Some(rest) = rest.strip_prefix('-') else {
        return Some(0);
    };
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() || digits.len() > 2 {
        return Some(0);
    }
    Some(digits.parse().unwrap_or(0))
}

/// Smallest thinking budget Anthropic accepts, also the default.
pub(crate) const MIN_THINKING_BUDGET: u32 = 1024;
const MAX_THINKING_BUDGET: u32 = 128_000;

/// The `thinking` and `output_config` request fields for `config` on `model`, plus the
/// budget (when one is sent) so the caller can size `max_tokens`.
pub(crate) fn thinking_params(
    model: &str,
    config: Option<&ThinkingConfig>,
) -> (Option<Value>, Option<Value>, Option<u32>) {
    let Some(config) = config else {
        return (None, None, None);
    };
    let caps = model_capabilities(model);
    if !caps.thinking {
        return (None, None, None);
    }
    let output_config = config
        .effort
        .filter(|_| caps.effort)
        .map(|effort| json!({"effort": effort.as_str()}));
    // Thinking off is sent as no parameter: some models reject `{"type": "disabled"}`.
    if !config.enabled {
        return (None, output_config, None);
    }
    if caps.adaptive {
        let mut thinking = json!({"type": "adaptive"});
        if let Some(display) = config.display {
            thinking["display"] = json!(display.as_str());
        }
        return (Some(thinking), output_config, None);
    }
    let budget = config
        .budget_tokens
        .unwrap_or(MIN_THINKING_BUDGET)
        .clamp(MIN_THINKING_BUDGET, MAX_THINKING_BUDGET);
    (
        Some(json!({"type": "enabled", "budget_tokens": budget})),
        output_config,
        Some(budget),
    )
}

/// Thinking blocks to send back in an assistant turn. Blocks without a signature come
/// from other providers and are dropped: Anthropic rejects unsigned thinking.
fn thinking_blocks(message: &Message) -> impl Iterator<Item = Value> + '_ {
    message.thinking.iter().filter_map(|block| match block {
        ThinkingBlock::Thinking {
            thinking,
            signature: Some(signature),
        } => Some(json!({"type": "thinking", "thinking": thinking, "signature": signature})),
        ThinkingBlock::Redacted { data } => {
            Some(json!({"type": "redacted_thinking", "data": data}))
        }
        _ => None,
    })
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct ApiResponse {
    pub content: Vec<Content>,
    pub id: String,
    pub model: String,
    pub role: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<serde_json::Value>,
    #[serde(rename = "type")]
    pub message_type: String,
    pub usage: Usage,
}

/// A response content block. `text`, `tool_use`, `thinking` and `redacted_thinking` are
/// used; other block types are accepted and ignored.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub(crate) struct Content {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Encrypted reasoning of a `redacted_thinking` block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

impl Content {
    fn thinking_block(&self) -> Option<ThinkingBlock> {
        match self.content_type.as_str() {
            "thinking" => Some(ThinkingBlock::Thinking {
                thinking: self.thinking.clone().unwrap_or_default(),
                signature: self.signature.clone().filter(|s| !s.is_empty()),
            }),
            "redacted_thinking" => Some(ThinkingBlock::Redacted {
                data: self.data.clone().unwrap_or_default(),
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub(crate) struct Usage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
}

impl Usage {
    /// `input_tokens` excludes cached tokens; `prompt_tokens` counts the whole prompt, as
    /// with the other providers.
    pub fn token_usage(&self) -> TokenUsage {
        let prompt =
            self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens;
        TokenUsage::new(prompt, self.output_tokens).with_cache_tokens(
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
        )
    }
}

/// Turns response blocks into a [`GenerateResult`]: the text, or — when Claude asks for tools —
/// typed `tool_calls`, mirrored into `generation` as an OpenAI-like JSON array (same contract
/// as the other providers).
pub(crate) fn result_from_blocks(blocks: &[Content], tokens: Option<TokenUsage>) -> GenerateResult {
    let calls: Vec<ToolCall> = blocks
        .iter()
        .filter(|b| b.content_type == "tool_use")
        .map(|b| {
            ToolCall::from_value(
                b.id.clone().unwrap_or_default(),
                b.name.clone().unwrap_or_default(),
                b.input.as_ref().unwrap_or(&json!({})),
            )
        })
        .collect();
    let text = blocks
        .iter()
        .filter(|b| b.content_type == "text")
        .filter_map(|b| b.text.as_deref())
        .collect();
    GenerateResult {
        tokens,
        generation: text,
        ..Default::default()
    }
    .with_tool_calls(calls)
    .with_thinking(blocks.iter().filter_map(Content::thinking_block).collect())
}
