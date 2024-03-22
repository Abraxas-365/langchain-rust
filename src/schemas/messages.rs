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
    #[serde(rename = "tool")]
    ToolMessage,
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
            MessageType::ToolMessage => "tool".to_owned(),
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
    pub id: Option<String>,
    pub tool_calls: Option<Value>,
}

impl Message {
    // Function to create a new Human message with a generic type that implements Display
    pub fn new_human_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::HumanMessage,
            id: None,
            tool_calls: None,
        }
    }

    // Function to create a new System message with a generic type that implements Display
    pub fn new_system_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::SystemMessage,
            id: None,
            tool_calls: None,
        }
    }

    // Function to create a new AI message with a generic type that implements Display
    pub fn new_ai_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::AIMessage,
            id: None,
            tool_calls: None,
        }
    }

    // Function to create a new Tool message with a generic type that implements Display
    pub fn new_tool_message<T: std::fmt::Display, S: Into<String>>(content: T, id: S) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::ToolMessage,
            id: Some(id.into()),
            tool_calls: None,
        }
    }

    /// Sets the tool calls for the OpenAI-like API call.
    ///
    /// Use this method when you need to specify tool calls in the configuration.
    /// This is particularly useful in scenarios where interactions with specific
    /// tools are required for operation.
    ///
    /// # Arguments
    ///
    /// * `tool_calls` - A `serde_json::Value` representing the tool call configurations.
    pub fn with_tool_calls(mut self, tool_calls: Value) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    pub fn messages_from_value(value: &Value) -> Result<Vec<Message>, serde_json::error::Error> {
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
