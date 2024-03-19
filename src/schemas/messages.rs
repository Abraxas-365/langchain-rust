use std::error::Error;

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

/// Enum `MessageType` represents the type of a message.
/// It can be a `SystemMessage`, `AIMessage`, or `HumanMessage`.
///
/// # Usage
/// ```rust,ignore
/// let system_message_type = MessageType::SystemMessage;
/// let ai_message_type = MessageType::AIMessage;
/// let human_message_type = MessageType::HumanMessage;
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MessageType {
    #[serde(rename = "system")]
    SystemMessage,
    #[serde(rename = "ai")]
    AIMessage,
    #[serde(rename = "human")]
    HumanMessage,
}

impl Default for MessageType {
    fn default() -> Self {
        Self::SystemMessage
    }
}

impl MessageType {
    pub fn to_string(&self) -> String {
        match self {
            MessageType::SystemMessage => "system".to_owned(),
            MessageType::AIMessage => "ai".to_owned(),
            MessageType::HumanMessage => "human".to_owned(),
        }
    }
}

/// Struct `Message` represents a message with its content and type.
///
/// # Usage
/// ```rust,ignore
/// let human_message = Message::new_human_message("Hello");
/// let system_message = Message::new_system_message("System Alert");
/// let ai_message = Message::new_ai_message("AI Response");
/// ```
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Message {
    pub content: String,
    pub message_type: MessageType,
}

impl Message {
    // Function to create a new Human message with a generic type that implements Display
    pub fn new_human_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::HumanMessage,
        }
    }

    // Function to create a new System message with a generic type that implements Display
    pub fn new_system_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::SystemMessage,
        }
    }

    // Function to create a new AI message with a generic type that implements Display
    pub fn new_ai_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::AIMessage,
        }
    }

    pub fn messages_from_value(value: &Value) -> Result<Vec<Message>, Box<dyn Error>> {
        serde_json::from_value(value.clone()).map_err(|e| e.into())
    }

    pub fn messages_to_string(messages: &[Message]) -> String {
        messages
            .iter()
            .map(|m| format!("{:?}: {}", m.message_type, m.content))
            .collect::<Vec<String>>()
            .join("\n")
    }
}
