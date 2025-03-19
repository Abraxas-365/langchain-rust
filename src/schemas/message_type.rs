use serde::{Deserialize, Serialize};
use std::fmt;

/// Enum `MessageType` represents the type of a message.
/// It can be a `SystemMessage`, `AIMessage`, or `HumanMessage`.
///
/// # Usage
/// ```rust,ignore
/// let system_message_type = MessageType::SystemMessage;
/// let ai_message_type = MessageType::AIMessage;
/// let human_message_type = MessageType::HumanMessage;
/// ```
#[derive(PartialEq, Eq, Serialize, Deserialize, Debug, Clone)]
pub enum MessageType {
    #[serde(rename = "system")]
    SystemMessage,
    #[serde(rename = "ai")]
    AIMessage,
    #[serde(rename = "human")]
    HumanMessage,
    #[serde(rename = "tool")]
    ToolMessage,
}

impl Default for MessageType {
    fn default() -> Self {
        Self::SystemMessage
    }
}

impl fmt::Display for MessageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageType::SystemMessage => write!(f, "system"),
            MessageType::AIMessage => write!(f, "ai"),
            MessageType::HumanMessage => write!(f, "human"),
            MessageType::ToolMessage => write!(f, "tool"),
        }
    }
}
