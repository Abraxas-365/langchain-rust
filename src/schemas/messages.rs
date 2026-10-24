use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

use crate::language_models::ThinkingBlock;

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
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq, Eq)]
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

impl ImageContent {
    /// Sets the detail level (`"auto"`, `"low"` or `"high"`). Only OpenAI uses it.
    pub fn with_detail<S: Into<String>>(mut self, detail: S) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Where the image comes from: a remote URL, or inline base64 data.
    pub(crate) fn source(&self) -> ImageSource<'_> {
        ImageSource::parse(&self.image_url)
    }
}

/// An image reference as found in [`ImageContent::image_url`].
///
/// `image_url` can be an `http(s)` URL, a `data:<media type>;base64,<data>` URL or
/// bare base64 data. For bare data the media type is sniffed from the first bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ImageSource<'a> {
    Url(&'a str),
    Base64 {
        media_type: Option<&'a str>,
        data: &'a str,
    },
}

impl<'a> ImageSource<'a> {
    pub(crate) fn parse(image: &'a str) -> Self {
        let image = image.trim();
        if let Some(rest) = image.strip_prefix("data:") {
            if let Some((header, data)) = rest.split_once(',') {
                if let Some(media_type) = header.strip_suffix(";base64") {
                    let media_type = (!media_type.is_empty()).then_some(media_type);
                    return ImageSource::Base64 {
                        media_type: media_type.or_else(|| sniff_media_type(data)),
                        data,
                    };
                }
            }
            return ImageSource::Url(image);
        }
        if image.starts_with("http://") || image.starts_with("https://") {
            return ImageSource::Url(image);
        }
        ImageSource::Base64 {
            media_type: sniff_media_type(image),
            data: image,
        }
    }

    /// The image as something OpenAI-compatible APIs accept: the URL itself, or a
    /// `data:` URL for inline data.
    pub(crate) fn to_url(self) -> String {
        match self {
            ImageSource::Url(url) => url.to_string(),
            ImageSource::Base64 { media_type, data } => format!(
                "data:{};base64,{data}",
                media_type.unwrap_or("application/octet-stream")
            ),
        }
    }
}

/// Media type of base64 image data, from the encoded magic bytes.
fn sniff_media_type(data: &str) -> Option<&'static str> {
    [
        ("/9j/", "image/jpeg"),
        ("iVBORw0KGgo", "image/png"),
        ("R0lGOD", "image/gif"),
        ("UklGR", "image/webp"),
    ]
    .into_iter()
    .find(|(prefix, _)| data.starts_with(prefix))
    .map(|(_, media_type)| media_type)
}

/// Struct `Message` represents a message with its content and type.
///
/// `thinking` holds an AI message's reasoning blocks, so providers that need them back
/// in later turns (Claude, in a tool-use loop) get them unchanged.
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub thinking: Vec<ThinkingBlock>,
}

impl Message {
    fn of_type<T: std::fmt::Display>(message_type: MessageType, content: T) -> Self {
        Message {
            content: content.to_string(),
            message_type,
            ..Default::default()
        }
    }

    // Function to create a new Human message with a generic type that implements Display
    pub fn new_human_message<T: std::fmt::Display>(content: T) -> Self {
        Self::of_type(MessageType::HumanMessage, content)
    }

    pub fn new_human_message_with_images<T: Into<ImageContent>>(images: Vec<T>) -> Self {
        Self::of_type(MessageType::HumanMessage, "").with_images(images)
    }

    // Function to create a new System message with a generic type that implements Display
    pub fn new_system_message<T: std::fmt::Display>(content: T) -> Self {
        Self::of_type(MessageType::SystemMessage, content)
    }

    // Function to create a new AI message with a generic type that implements Display
    pub fn new_ai_message<T: std::fmt::Display>(content: T) -> Self {
        Self::of_type(MessageType::AIMessage, content)
    }

    // Function to create a new Tool message with a generic type that implements Display
    pub fn new_tool_message<T: std::fmt::Display, S: Into<String>>(content: T, id: S) -> Self {
        Message {
            id: Some(id.into()),
            ..Self::of_type(MessageType::ToolMessage, content)
        }
    }

    /// Attaches the model's reasoning to an AI message, so it is sent back in later turns.
    pub fn with_thinking(mut self, thinking: Vec<ThinkingBlock>) -> Self {
        self.thinking = thinking;
        self
    }

    /// Attaches images to the message (keeping its text). Providers that support
    /// vision send them along with the text.
    pub fn with_images<T: Into<ImageContent>>(mut self, images: Vec<T>) -> Self {
        self.images = Some(images.into_iter().map(Into::into).collect());
        self
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

    /// An AI message that requests tool calls, stored in the shared wire format.
    pub fn new_ai_message_with_tool_calls<T: std::fmt::Display>(
        content: T,
        tool_calls: &[super::ToolCall],
    ) -> Self {
        Self::new_ai_message(content).with_tool_calls(serde_json::json!(tool_calls))
    }

    /// The tool calls on this message as typed [`ToolCall`](super::ToolCall)s.
    /// Empty when there are none or `tool_calls` is not in the OpenAI-like format.
    pub fn parse_tool_calls(&self) -> Vec<super::ToolCall> {
        self.tool_calls
            .as_ref()
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default()
    }

    pub fn messages_from_value(value: &Value) -> Result<Vec<Message>, serde_json::error::Error> {
        serde_json::from_value(value.clone())
    }

    /// Renders messages as text, one per line. Images show up as `[image: <url>]`, or
    /// `[image]` for inline data, so a transcript shows that an image was there.
    pub fn messages_to_string(messages: &[Message]) -> String {
        messages
            .iter()
            .map(|m| format!("{:?}: {}", m.message_type, m.text_with_image_markers()))
            .collect::<Vec<String>>()
            .join("\n")
    }

    /// The message text followed by a marker per attached image.
    pub(crate) fn text_with_image_markers(&self) -> String {
        let mut text = self.content.clone();
        for image in self.images.iter().flatten() {
            if !text.is_empty() {
                text.push(' ');
            }
            match image.source() {
                ImageSource::Url(url) => text.push_str(&format!("[image: {url}]")),
                ImageSource::Base64 { .. } => text.push_str("[image]"),
            }
        }
        text
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

    #[test]
    fn typed_tool_calls_round_trip_through_message() {
        use crate::schemas::ToolCall;
        let calls = vec![
            ToolCall::new("c1", "weather", r#"{"city":"Lima"}"#),
            ToolCall::new("c2", "time", "{}"),
        ];
        let m = Message::new_ai_message_with_tool_calls("checking", &calls);
        assert_eq!(m.content, "checking");
        assert_eq!(m.message_type, MessageType::AIMessage);
        assert_eq!(m.tool_calls.as_ref().unwrap()[0]["type"], "function");
        assert_eq!(m.parse_tool_calls(), calls);

        // Raw JSON set the old way parses the same.
        let raw = Message::new_ai_message("").with_tool_calls(json!([
            {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}
        ]));
        assert_eq!(
            raw.parse_tool_calls(),
            vec![ToolCall::new("c1", "weather", "{}")]
        );

        assert!(Message::new_ai_message("x").parse_tool_calls().is_empty());
        assert!(Message::new_ai_message("")
            .with_tool_calls(json!([{"id": "1"}]))
            .parse_tool_calls()
            .is_empty());
    }

    #[test]
    fn image_source_parses_urls_data_urls_and_bare_base64() {
        use ImageSource::*;
        let cases = [
            ("https://a/b.png", Url("https://a/b.png")),
            ("http://a/b", Url("http://a/b")),
            (
                "data:image/webp;base64,AAAA",
                Base64 {
                    media_type: Some("image/webp"),
                    data: "AAAA",
                },
            ),
            // Empty media type in a data: URL is sniffed from the data.
            (
                "data:;base64,/9j/4AA",
                Base64 {
                    media_type: Some("image/jpeg"),
                    data: "/9j/4AA",
                },
            ),
            // Not base64 encoded: left as a URL.
            ("data:text/plain,hello", Url("data:text/plain,hello")),
            (
                "  iVBORw0KGgoAAA \n",
                Base64 {
                    media_type: Some("image/png"),
                    data: "iVBORw0KGgoAAA",
                },
            ),
            (
                "R0lGODlh",
                Base64 {
                    media_type: Some("image/gif"),
                    data: "R0lGODlh",
                },
            ),
            (
                "UklGRgAA",
                Base64 {
                    media_type: Some("image/webp"),
                    data: "UklGRgAA",
                },
            ),
            (
                "AAAA",
                Base64 {
                    media_type: None,
                    data: "AAAA",
                },
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(ImageSource::parse(input), expected, "{input}");
        }

        assert_eq!(
            ImageSource::parse("https://a/b.png").to_url(),
            "https://a/b.png"
        );
        assert_eq!(
            ImageSource::parse("iVBORw0KGgo").to_url(),
            "data:image/png;base64,iVBORw0KGgo"
        );
        assert_eq!(
            ImageSource::parse("AAAA").to_url(),
            "data:application/octet-stream;base64,AAAA"
        );
        assert_eq!(
            ImageSource::parse("data:image/gif;base64,R0l").to_url(),
            "data:image/gif;base64,R0l"
        );
    }

    #[test]
    fn with_images_keeps_text_and_transcripts_show_markers() {
        let m = Message::new_human_message("look").with_images(vec![
            ImageContent::from("https://a/cat.png").with_detail("high"),
            ImageContent::from("iVBORw0KGgo"),
        ]);
        assert_eq!(m.content, "look");
        assert_eq!(m.images.as_ref().unwrap().len(), 2);
        assert_eq!(
            m.images.as_ref().unwrap()[0].detail.as_deref(),
            Some("high")
        );
        assert_eq!(
            m.text_with_image_markers(),
            "look [image: https://a/cat.png] [image]"
        );

        let only = Message::new_human_message("").with_images(vec!["https://a/b.png"]);
        assert_eq!(only.text_with_image_markers(), "[image: https://a/b.png]");
        assert_eq!(
            Message::new_ai_message("hi").text_with_image_markers(),
            "hi"
        );

        assert_eq!(
            Message::messages_to_string(&[m, Message::new_ai_message("a cat")]),
            "HumanMessage: look [image: https://a/cat.png] [image]\nAIMessage: a cat"
        );
    }

    #[test]
    fn thinking_round_trips_and_old_messages_still_parse() {
        let blocks = vec![
            ThinkingBlock::Thinking {
                thinking: "plan".into(),
                signature: Some("sig".into()),
            },
            ThinkingBlock::Redacted { data: "enc".into() },
        ];
        let m = Message::new_ai_message("done").with_thinking(blocks.clone());
        let value = serde_json::to_value(&m).unwrap();
        assert_eq!(value["thinking"][0]["signature"], "sig");
        let back: Message = serde_json::from_value(value).unwrap();
        assert_eq!(back.thinking, blocks);

        // Stored before this field existed: no `thinking` key.
        let old: Message = serde_json::from_value(json!({
            "content": "hi", "message_type": "ai", "id": null, "tool_calls": null, "images": null
        }))
        .unwrap();
        assert!(old.thinking.is_empty());
        let plain = serde_json::to_value(Message::new_ai_message("x")).unwrap();
        assert!(plain.get("thinking").is_none(), "omitted when empty");
    }
}
