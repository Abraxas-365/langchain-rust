use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::schemas::{FunctionCallBehavior, FunctionCallResponse, FunctionDefinition, Message};

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

    /// An assistant turn. Tool calls (stored in the OpenAI-like format used across the crate)
    /// become `tool_use` blocks, preceded by the text if there is any.
    pub fn assistant(message: &Message) -> Self {
        let calls = message
            .tool_calls
            .clone()
            .and_then(|v| serde_json::from_value::<Vec<FunctionCallResponse>>(v).ok())
            .unwrap_or_default();
        if calls.is_empty() {
            return Self::text("assistant", &message.content);
        }

        let mut blocks = Vec::new();
        if !message.content.trim().is_empty() {
            blocks.push(json!({"type": "text", "text": message.content}));
        }
        for call in calls {
            blocks.push(json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.function.name,
                "input": call.function.arguments_object(),
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
        let next = match message.message_type {
            MessageType::AIMessage => ClaudeMessage::assistant(message),
            MessageType::HumanMessage | MessageType::SystemMessage => {
                ClaudeMessage::text("user", &message.content)
            }
            MessageType::ToolMessage => ClaudeMessage::tool_result(message),
        };
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
    pub system: Option<String>,
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

/// A response content block. Only `text` and `tool_use` are used; other block types
/// (e.g. `thinking`) are accepted and ignored.
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
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub(crate) struct Usage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
}

/// Turns response blocks into the crate's `generation` string: the text, or — when Claude asks
/// for tools — the tool calls as an OpenAI-like JSON array, which is what the tools agent reads
/// (same contract as the OpenAI client).
pub(crate) fn generation_from_blocks(blocks: &[Content]) -> String {
    let calls: Vec<Value> = blocks
        .iter()
        .filter(|b| b.content_type == "tool_use")
        .map(|b| {
            json!({
                "id": b.id.clone().unwrap_or_default(),
                "type": "function",
                "function": {
                    "name": b.name.clone().unwrap_or_default(),
                    "arguments": b.input.clone().unwrap_or_else(|| json!({})).to_string(),
                },
            })
        })
        .collect();
    if !calls.is_empty() {
        return Value::Array(calls).to_string();
    }
    blocks
        .iter()
        .filter(|b| b.content_type == "text")
        .filter_map(|b| b.text.as_deref())
        .collect()
}
