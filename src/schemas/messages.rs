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

/// Struct `ImageContent` represents an image provided to an LLM.
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct ImageContent {
    pub image_url: String,
    pub detail: Option<String>,
}

impl<S: AsRef<str>> From<S> for ImageContent {
    fn from(image_url: S) -> Self {
        ImageContent {
            image_url: image_url.as_ref().into(),
            detail: None,
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
    pub images: Option<Vec<ImageContent>>,
}

impl Message {
    // Function to create a new Human message with a generic type that implements Display
    pub fn new_human_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::HumanMessage,
            id: None,
            tool_calls: None,
            images: None,
        }
    }

    pub fn new_human_message_with_images<T: Into<ImageContent>>(images: Vec<T>) -> Self {
        Message {
            content: String::default(),
            message_type: MessageType::HumanMessage,
            id: None,
            tool_calls: None,
            images: Some(images.into_iter().map(|i| i.into()).collect()),
        }
    }

    // Function to create a new System message with a generic type that implements Display
    pub fn new_system_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::SystemMessage,
            id: None,
            tool_calls: None,
            images: None,
        }
    }

    // Function to create a new AI message with a generic type that implements Display
    pub fn new_ai_message<T: std::fmt::Display>(content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::AIMessage,
            id: None,
            tool_calls: None,
            images: None,
        }
    }

    // Function to create a new Tool message with a generic type that implements Display
    pub fn new_tool_message<T: std::fmt::Display, S: Into<String>>(content: T, id: S) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::ToolMessage,
            id: Some(id.into()),
            tool_calls: None,
            images: None,
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
        serde_json::from_value(value.clone())
    }

    pub fn messages_to_string(messages: &[Message]) -> String {
        messages
            .iter()
            .map(|m| format!("{:?}: {}", m.message_type, m.content))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

/// Renders a prompt input value as plain text: JSON strings without surrounding quotes,
/// any other JSON value in its serialized form.
pub(crate) fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn constructors_set_type_and_content() {
        assert_eq!(
            Message::new_human_message("h").message_type,
            MessageType::HumanMessage
        );
        assert_eq!(
            Message::new_ai_message(42).content,
            "42",
            "Display impl is used"
        );
        assert_eq!(
            Message::new_system_message("s").message_type,
            MessageType::SystemMessage
        );
        let tool = Message::new_tool_message("result", "call_1");
        assert_eq!(tool.message_type, MessageType::ToolMessage);
        assert_eq!(tool.id.as_deref(), Some("call_1"));
        assert_eq!(MessageType::default(), MessageType::SystemMessage);
    }

    #[test]
    fn human_message_with_images() {
        let m = Message::new_human_message_with_images(vec!["http://a/img.png", "http://b"]);
        let images = m.images.unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].image_url, "http://a/img.png");
        assert!(images[0].detail.is_none());
        assert!(m.content.is_empty());
    }

    #[test]
    fn message_type_serde_uses_short_names() {
        for (t, s) in [
            (MessageType::SystemMessage, "system"),
            (MessageType::AIMessage, "ai"),
            (MessageType::HumanMessage, "human"),
            (MessageType::ToolMessage, "tool"),
        ] {
            assert_eq!(serde_json::to_value(&t).unwrap(), json!(s));
            assert_eq!(t.to_string(), s);
            assert_eq!(serde_json::from_value::<MessageType>(json!(s)).unwrap(), t);
        }
    }

    #[test]
    fn messages_roundtrip_through_value() {
        let messages = vec![
            Message::new_human_message("hi"),
            Message::new_ai_message("").with_tool_calls(json!([{"id": "1"}])),
        ];
        let value = json!(messages);
        let parsed = Message::messages_from_value(&value).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].content, "hi");
        assert_eq!(parsed[1].tool_calls, Some(json!([{"id": "1"}])));
        assert!(Message::messages_from_value(&json!("not a list")).is_err());
    }

    #[test]
    fn messages_to_string_uses_debug_type_names() {
        let s = Message::messages_to_string(&[
            Message::new_human_message("a"),
            Message::new_ai_message("b"),
        ]);
        assert_eq!(s, "HumanMessage: a\nAIMessage: b");
    }
}
