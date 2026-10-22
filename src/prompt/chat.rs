use serde_json::Value;

use crate::schemas::{
    messages::{ImageContent, Message},
    prompt::PromptValue,
};

use super::{
    FormatPrompter, MessageFormatter, PromptArgs, PromptError, PromptFromatter, PromptTemplate,
};

/// Struct `HumanMessagePromptTemplate` defines a template for creating human (user) messages.
/// `PromptTemplate` is used to generate the message template.
///
/// # Usage
/// ```rust,ignore
/// let human_message_prompt = HumanMessagePromptTemplate::new(template_fstring!(
///    "User says: {content}",
///    "content",
/// ));
/// ```
///
/// # Images
///
/// [`with_images`](Self::with_images) names an input variable that holds images to
/// attach to the message, so a chain or agent can take them like any other input:
///
/// ```rust,ignore
/// let prompt = HumanMessagePromptTemplate::new(template_fstring!("{input}", "input"))
///     .with_images("images");
/// chain.invoke(prompt_args! {
///     "input" => "What is in this picture?",
///     "images" => ["https://example.com/cat.png"],
/// }).await?;
/// ```
#[derive(Clone)]
pub struct HumanMessagePromptTemplate {
    prompt: PromptTemplate,
    images_key: Option<String>,
}

impl HumanMessagePromptTemplate {
    pub fn new(prompt: PromptTemplate) -> Self {
        Self {
            prompt,
            images_key: None,
        }
    }

    /// Attaches the images found in the `key` input variable to the message.
    ///
    /// The variable can be one image or a list of them. An image is either a string
    /// (an `http(s)` URL, a `data:` URL or bare base64 data) or an
    /// [`ImageContent`] object (`{"image_url": "...", "detail": "low"}`).
    /// A missing variable, `null` or an empty list means no images.
    pub fn with_images<S: Into<String>>(mut self, key: S) -> Self {
        self.images_key = Some(key.into());
        self
    }
}

/// Reads the images held in an input variable (see
/// [`HumanMessagePromptTemplate::with_images`]).
pub(crate) fn images_from_value(value: &Value) -> Result<Vec<ImageContent>, PromptError> {
    fn one(value: &Value) -> Result<ImageContent, PromptError> {
        match value {
            Value::String(url) if !url.trim().is_empty() => Ok(ImageContent::from(url)),
            Value::Object(_) => Ok(serde_json::from_value(value.clone())?),
            other => Err(PromptError::OtherError(format!(
                "expected an image URL, base64 string or {{\"image_url\": ...}} object, got {other}"
            ))),
        }
    }
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => items.iter().map(one).collect(),
        other => Ok(vec![one(other)?]),
    }
}

/// Input variable the built-in chain and agent prompts read images from.
pub const DEFAULT_IMAGES_KEY: &str = "images";

/// The human turn of a chain or agent call, as it is saved to memory: the text of
/// `input_key` plus any images in `images_key`.
pub(crate) fn human_message_from_inputs(
    inputs: &PromptArgs,
    input_key: &str,
    images_key: &str,
) -> Result<Message, PromptError> {
    let input = inputs
        .get(input_key)
        .ok_or_else(|| PromptError::MissingVariable(input_key.to_string()))?;
    let message = Message::new_human_message(crate::schemas::messages::value_to_text(input));
    let images = match inputs.get(images_key) {
        Some(value) => images_from_value(value)?,
        None => Vec::new(),
    };
    Ok(if images.is_empty() {
        message
    } else {
        message.with_images(images)
    })
}

impl MessageFormatter for HumanMessagePromptTemplate {
    fn format_messages(&self, input_variables: PromptArgs) -> Result<Vec<Message>, PromptError> {
        let images = match &self.images_key {
            Some(key) => match input_variables.get(key) {
                Some(value) => images_from_value(value)?,
                None => Vec::new(),
            },
            None => Vec::new(),
        };
        let mut message = Message::new_human_message(self.prompt.format(input_variables)?);
        if !images.is_empty() {
            message = message.with_images(images);
        }
        tracing::debug!("message: {:?}", message);
        Ok(vec![message])
    }
    fn input_variables(&self) -> Vec<String> {
        self.prompt.variables().clone()
    }
}

impl FormatPrompter for HumanMessagePromptTemplate {
    fn format_prompt(&self, input_variables: PromptArgs) -> Result<PromptValue, PromptError> {
        let messages = self.format_messages(input_variables)?;
        Ok(PromptValue::from_messages(messages))
    }
    fn get_input_variables(&self) -> Vec<String> {
        self.input_variables()
    }
}

/// Struct `SystemMessagePromptTemplate` defines a template for creating system-level messages.
/// `PromptTemplate` is used to generate the message template.
///
/// # Usage
/// ```rust,ignore
/// let system_message_prompt = SystemMessagePromptTemplate::new(template_fstring!(
///    "System alert: {alert_type} {alert_detail}",
///    "alert_type",
///    "alert_detail"
/// ));
/// ```
#[derive(Clone)]
pub struct SystemMessagePromptTemplate {
    prompt: PromptTemplate,
}

impl SystemMessagePromptTemplate {
    pub fn new(prompt: PromptTemplate) -> Self {
        Self { prompt }
    }
}

impl FormatPrompter for SystemMessagePromptTemplate {
    fn format_prompt(&self, input_variables: PromptArgs) -> Result<PromptValue, PromptError> {
        let messages = self.format_messages(input_variables)?;
        Ok(PromptValue::from_messages(messages))
    }
    fn get_input_variables(&self) -> Vec<String> {
        self.input_variables()
    }
}

impl MessageFormatter for SystemMessagePromptTemplate {
    fn format_messages(&self, input_variables: PromptArgs) -> Result<Vec<Message>, PromptError> {
        let message = Message::new_system_message(self.prompt.format(input_variables)?);
        tracing::debug!("message: {:?}", message);
        Ok(vec![message])
    }
    fn input_variables(&self) -> Vec<String> {
        self.prompt.variables().clone()
    }
}

/// Struct `AIMessagePromptTemplate` defines a template for creating AI (assistant) messages.
/// `PromptTemplate` is used to generate the message template.
///
/// # Usage
/// ```rust,ignore
/// let ai_message_prompt = AIMessagePromptTemplate::new(template_fstring!(
///    "AI response: {content} {additional_info}",
///    "content",
///    "additional_info"
/// ));
#[derive(Clone)]
pub struct AIMessagePromptTemplate {
    prompt: PromptTemplate,
}

impl FormatPrompter for AIMessagePromptTemplate {
    fn format_prompt(&self, input_variables: PromptArgs) -> Result<PromptValue, PromptError> {
        let messages = self.format_messages(input_variables)?;
        Ok(PromptValue::from_messages(messages))
    }
    fn get_input_variables(&self) -> Vec<String> {
        self.input_variables()
    }
}

impl MessageFormatter for AIMessagePromptTemplate {
    fn format_messages(&self, input_variables: PromptArgs) -> Result<Vec<Message>, PromptError> {
        let message = Message::new_ai_message(self.prompt.format(input_variables)?);
        tracing::debug!("message: {:?}", message);
        Ok(vec![message])
    }
    fn input_variables(&self) -> Vec<String> {
        self.prompt.variables().clone()
    }
}

impl AIMessagePromptTemplate {
    pub fn new(prompt: PromptTemplate) -> Self {
        Self { prompt }
    }
}

pub enum MessageOrTemplate {
    Message(Message),
    Template(Box<dyn MessageFormatter>),
    MessagesPlaceholder(String),
}

/// `fmt_message` is a utility macro used to create a `MessageOrTemplate::Message` variant.
///
/// # Usage
/// The macro is called with a `Message` object. For example:
/// ```rust,ignore
/// let message = Message::new_human_message("Hello World");
/// fmt_message!(message) // Returns a `MessageOrTemplate::Message` variant that wraps the `Message` object
/// ```
#[macro_export]
macro_rules! fmt_message {
    ($msg:expr) => {
        $crate::prompt::MessageOrTemplate::Message($msg)
    };
}

/// `fmt_template` is a utility macro used to create a `MessageOrTemplate::Template` variant.
///
/// # Usage
/// The macro is called with a `MessageFormatter` object, for instance `HumanMessagePromptTemplate`,
/// `SystemMessagePromptTemplate`, `AIMessagePromptTemplate` or any other implementation of `MessageFormatter`.
///
/// ```rust,ignore
/// let prompt_template = HumanMessagePromptTemplate::new(template);
/// fmt_template!(prompt_template)
/// ```
/// This returns a `MessageOrTemplate::Template` variant that wraps the `MessageFormatter` object within a Box.
#[macro_export]
macro_rules! fmt_template {
    ($template:expr) => {
        $crate::prompt::MessageOrTemplate::Template(Box::new($template))
    };
}

/// `fmt_placeholder` is a utility macro used to create a `MessageOrTemplate::MessagesPlaceholder` variant.
///
/// # Usage
/// The macro is called with a string literal or a String object:
/// ```rust,ignore
/// fmt_placeholder!("Placeholder message")
/// ```
/// This returns a `MessageOrTemplate::MessagesPlaceholder` variant that wraps the given string.
#[macro_export]
macro_rules! fmt_placeholder {
    ($placeholder:expr) => {
        $crate::prompt::MessageOrTemplate::MessagesPlaceholder($placeholder.into())
    };
}

pub struct MessageFormatterStruct {
    items: Vec<MessageOrTemplate>,
}

impl MessageFormatterStruct {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn add_message(&mut self, message: Message) {
        self.items.push(MessageOrTemplate::Message(message));
    }

    pub fn add_template(&mut self, template: Box<dyn MessageFormatter>) {
        self.items.push(MessageOrTemplate::Template(template));
    }

    pub fn add_messages_placeholder(&mut self, placeholder: &str) {
        self.items.push(MessageOrTemplate::MessagesPlaceholder(
            placeholder.to_string(),
        ));
    }

    fn format(&self, input_variables: PromptArgs) -> Result<Vec<Message>, PromptError> {
        let mut result: Vec<Message> = Vec::new();
        for item in &self.items {
            match item {
                MessageOrTemplate::Message(msg) => result.push(msg.clone()),
                MessageOrTemplate::Template(tmpl) => {
                    result.extend(tmpl.format_messages(input_variables.clone())?)
                }
                MessageOrTemplate::MessagesPlaceholder(placeholder) => {
                    let messages = input_variables
                        .get(placeholder)
                        .ok_or_else(|| PromptError::MissingVariable(placeholder.clone()))?;
                    result.extend(Message::messages_from_value(messages)?);
                }
            }
        }
        Ok(result)
    }
}

impl MessageFormatter for MessageFormatterStruct {
    fn format_messages(&self, input_variables: PromptArgs) -> Result<Vec<Message>, PromptError> {
        self.format(input_variables)
    }
    fn input_variables(&self) -> Vec<String> {
        let mut variables = Vec::new();
        for item in &self.items {
            match item {
                MessageOrTemplate::Message(_) => {}
                MessageOrTemplate::Template(tmpl) => {
                    variables.extend(tmpl.input_variables());
                }
                MessageOrTemplate::MessagesPlaceholder(placeholder) => {
                    variables.extend(vec![placeholder.clone()]);
                }
            }
        }
        variables
    }
}

impl FormatPrompter for MessageFormatterStruct {
    fn format_prompt(&self, input_variables: PromptArgs) -> Result<PromptValue, PromptError> {
        let messages = self.format(input_variables)?;
        Ok(PromptValue::from_messages(messages))
    }
    fn get_input_variables(&self) -> Vec<String> {
        self.input_variables()
    }
}

#[macro_export]
// A macro for creating a new MessageFormatterStruct with various types of messages.
///
///# Example
/// ```rust,ignore
/// // Create an AI message prompt template
/// let ai_message_prompt = AIMessagePromptTemplate::new(
/// template_fstring!(
///     "AI response: {content} {test}",
///     "content",
///     "test"
/// ));
///
///
/// let human_msg = Message::new_human_message("Hello from user");
///
/// // Use the `message_formatter` macro to construct the formatter.
/// let formatter = message_formatter![
///     fmt_message!(human_msg),
///     fmt_template!(ai_message_prompt),
///     fmt_placeholder!("history")
/// ];
/// ```
macro_rules! message_formatter {
($($item:expr),* $(,)?) => {{
    let mut formatter = $crate::prompt::MessageFormatterStruct::new();
    $(
        match $item {
            $crate::prompt::MessageOrTemplate::Message(msg) => formatter.add_message(msg),
            $crate::prompt::MessageOrTemplate::Template(tmpl) => formatter.add_template(tmpl),
            $crate::prompt::MessageOrTemplate::MessagesPlaceholder(placeholder) => formatter.add_messages_placeholder(&placeholder.clone()),
        }
    )*
    formatter
}};
}

#[cfg(test)]
mod tests {
    use crate::{
        message_formatter,
        prompt::{chat::AIMessagePromptTemplate, FormatPrompter, PromptError},
        prompt_args,
        schemas::messages::{Message, MessageType},
        template_fstring,
    };

    #[test]
    fn test_message_formatter_macro() {
        // Create a human message and system message
        let human_msg = Message::new_human_message("Hello from user");

        // Create an AI message prompt template
        let ai_message_prompt = AIMessagePromptTemplate::new(template_fstring!(
            "AI response: {content} {test}",
            "content",
            "test"
        ));

        // Use the `message_formatter` macro to construct the formatter
        let formatter = message_formatter![
            fmt_message!(human_msg),
            fmt_template!(ai_message_prompt),
            fmt_placeholder!("history")
        ];

        // Define input variables for the AI message template
        let input_variables = prompt_args! {
            "content" => "This is a test",
            "test" => "test2",
            "history" => vec![
                Message::new_human_message("Placeholder message 1"),
                Message::new_ai_message("Placeholder message 2"),
            ],


        };

        // Format messages
        let formatted_messages = formatter
            .format_prompt(input_variables)
            .unwrap()
            .to_chat_messages();

        // Verify the number of messages
        assert_eq!(formatted_messages.len(), 4);

        // Verify the content of each message
        assert_eq!(formatted_messages[0].content, "Hello from user");
        assert_eq!(
            formatted_messages[1].content,
            "AI response: This is a test test2"
        );
        assert_eq!(formatted_messages[2].content, "Placeholder message 1");
        assert_eq!(formatted_messages[3].content, "Placeholder message 2");
    }

    #[test]
    fn message_templates_roles_and_variables() {
        use crate::{
            prompt::{
                HumanMessagePromptTemplate, MessageFormatter, PromptError,
                SystemMessagePromptTemplate,
            },
            schemas::MessageType,
        };
        let formatter = message_formatter![
            fmt_template!(SystemMessagePromptTemplate::new(template_fstring!(
                "You are {role}",
                "role"
            ))),
            fmt_placeholder!("history"),
            fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
                "{q}", "q"
            ))),
        ];
        assert_eq!(
            formatter.input_variables(),
            vec!["role".to_string(), "history".into(), "q".into()]
        );

        let messages = formatter
            .format_messages(prompt_args! {
                "role" => "a bot", "q" => "hi", "history" => Vec::<Message>::new()
            })
            .unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].message_type, MessageType::SystemMessage);
        assert_eq!(messages[0].content, "You are a bot");
        assert_eq!(messages[1].message_type, MessageType::HumanMessage);

        // missing placeholder is an error, not a panic
        let err = formatter
            .format_messages(prompt_args! {"role" => "x", "q" => "y"})
            .unwrap_err();
        assert!(matches!(err, PromptError::MissingVariable(v) if v == "history"));

        // missing template variable
        let err = formatter
            .format_messages(prompt_args! {"history" => Vec::<Message>::new(), "q" => "y"})
            .unwrap_err();
        assert!(matches!(err, PromptError::MissingVariable(v) if v == "role"));

        // placeholder with wrong type
        let err = formatter
            .format_messages(prompt_args! {"role" => "x", "q" => "y", "history" => "nope"})
            .unwrap_err();
        assert!(matches!(err, PromptError::SerializationError(_)));
    }

    #[test]
    fn human_template_attaches_images_from_a_variable() {
        use crate::prompt::{HumanMessagePromptTemplate, MessageFormatter};
        use crate::schemas::messages::ImageContent;
        use serde_json::json;

        let template = HumanMessagePromptTemplate::new(template_fstring!("Q: {input}", "input"))
            .with_images("images");
        // Images are not prompt variables: the template only needs `input`.
        assert_eq!(template.input_variables(), vec!["input".to_string()]);

        let urls = |messages: &[Message]| -> Vec<String> {
            messages[0]
                .images
                .iter()
                .flatten()
                .map(|i| i.image_url.clone())
                .collect()
        };

        // A list mixing strings and ImageContent objects.
        let messages = template
            .format_messages(prompt_args! {
                "input" => "what is it?",
                "images" => json!(["https://a/cat.png", {"image_url": "iVBORw0KGgo", "detail": "low"}]),
            })
            .unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "Q: what is it?");
        assert_eq!(urls(&messages), vec!["https://a/cat.png", "iVBORw0KGgo"]);
        assert_eq!(
            messages[0].images.as_ref().unwrap()[1],
            ImageContent::from("iVBORw0KGgo").with_detail("low")
        );

        // A single string works too.
        let messages = template
            .format_messages(prompt_args! {"input" => "x", "images" => "https://a/b.png"})
            .unwrap();
        assert_eq!(urls(&messages), vec!["https://a/b.png"]);

        // Missing, null or empty: a plain text message (images stays None).
        for args in [
            prompt_args! {"input" => "x"},
            prompt_args! {"input" => "x", "images" => json!(null)},
            prompt_args! {"input" => "x", "images" => json!([])},
        ] {
            assert_eq!(template.format_messages(args).unwrap()[0].images, None);
        }

        // Without with_images the variable is ignored.
        let plain = HumanMessagePromptTemplate::new(template_fstring!("{input}", "input"));
        let messages = plain
            .format_messages(prompt_args! {"input" => "x", "images" => "https://a/b.png"})
            .unwrap();
        assert_eq!(messages[0].images, None);

        // Bad values are errors, not silently dropped.
        for bad in [
            json!(42),
            json!([""]),
            json!([{"detail": "low"}]),
            json!([[]]),
        ] {
            let err = template
                .format_messages(prompt_args! {"input" => "x", "images" => bad.clone()})
                .unwrap_err();
            assert!(
                matches!(
                    err,
                    PromptError::OtherError(_) | PromptError::SerializationError(_)
                ),
                "{bad}: {err:?}"
            );
        }
    }

    #[test]
    fn human_message_from_inputs_reads_text_and_images() {
        use crate::prompt::human_message_from_inputs;
        use serde_json::json;

        let m = human_message_from_inputs(
            &prompt_args! {"input" => "hi", "images" => json!(["https://a/b.png"])},
            "input",
            "images",
        )
        .unwrap();
        assert_eq!(m.message_type, MessageType::HumanMessage);
        assert_eq!(m.content, "hi");
        assert_eq!(m.images.unwrap()[0].image_url, "https://a/b.png");

        // Non-string input is rendered without JSON quotes for strings only.
        let m = human_message_from_inputs(
            &prompt_args! {"input" => json!({"a": 1})},
            "input",
            "images",
        )
        .unwrap();
        assert_eq!(m.content, r#"{"a":1}"#);
        assert_eq!(m.images, None);

        let err = human_message_from_inputs(&prompt_args! {}, "input", "images").unwrap_err();
        assert!(matches!(err, PromptError::MissingVariable(v) if v == "input"));
    }
}
