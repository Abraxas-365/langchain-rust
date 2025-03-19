use std::fmt;

use async_openai::types::ChatCompletionMessageToolCall;
use serde::Deserialize;
use serde::Serialize;

use super::MessageType;

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
#[derive(Debug, Default, Clone)]
pub struct Message {
    pub content: String,
    pub message_type: MessageType,
    pub id: Option<String>,
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCall>>,
    pub images: Option<Vec<ImageContent>>,
}

impl Message {
    pub fn new<T: std::fmt::Display>(message_type: MessageType, content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type,
            id: None,
            tool_calls: None,
            images: None,
        }
    }

    // Function to create a new Tool message with a generic type that implements Display
    pub fn new_tool_message<T: std::fmt::Display, S: Into<String>>(
        id: Option<S>,
        content: T,
    ) -> Self {
        Message {
            content: content.to_string(),
            message_type: MessageType::ToolMessage,
            id: id.map(|id| id.into()),
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
    pub fn with_tool_calls(mut self, tool_calls: Vec<ChatCompletionMessageToolCall>) -> Self {
        self.tool_calls = Some(tool_calls);
        self
    }

    pub fn with_images<T: Into<ImageContent>>(mut self, images: Vec<T>) -> Self {
        self.images = Some(images.into_iter().map(|i| i.into()).collect());
        self
    }

    pub fn messages_to_string(messages: &[Message]) -> String {
        messages
            .iter()
            .map(|m| m.to_string())
            .collect::<Vec<String>>()
            .join("\n")
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(tool_calls) = &self.tool_calls {
            write!(
                f,
                "Tool call:\n{}",
                serde_json::to_string_pretty(&tool_calls)
                    .unwrap_or("Tool call details unknown".into())
            )
        } else if let Some(images) = &self.images {
            write!(
                f,
                "{}: {}\nImages: {:?}",
                self.message_type, self.content, images
            )
        } else if !self.content.is_empty() {
            write!(f, "{}: {}", self.message_type, self.content)
        } else {
            log::warn!("Message without content nor tool calls found, possibly an error");
            Ok(())
        }
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        todo!()
    }
}
