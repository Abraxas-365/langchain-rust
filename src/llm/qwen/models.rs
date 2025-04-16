use serde::{Deserialize, Serialize};

use crate::schemas::{Message, MessageType};

#[derive(Serialize, Deserialize)]
pub(crate) struct QwenMessage {
    pub role: String,
    pub content: String,
}

impl QwenMessage {
    pub fn new<S: Into<String>>(role: S, content: S) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn from_message(message: &Message) -> Self {
        match message.message_type {
            MessageType::SystemMessage => Self::new("system", &message.content),
            MessageType::AIMessage => Self::new("assistant", &message.content),
            MessageType::HumanMessage => Self::new("user", &message.content),
            // Qwen may not have direct support for tool messages in the same way as Claude
            // For now, handle them as user messages
            MessageType::ToolMessage => Self::new("user", &message.content),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Payload {
    pub model: String,
    pub messages: Vec<QwenMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_format: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct ApiResponse {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct Choice {
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
    pub index: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// Stream response structures
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct StreamResponse {
    pub id: String,
    pub model: String,
    pub created: u64,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct StreamChoice {
    pub delta: Delta,
    pub finish_reason: Option<String>,
    pub index: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// Error response structure
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ErrorResponse {
    pub request_id: String,
    pub code: String,
    pub message: String,
} 