use std::{collections::HashMap, error::Error, sync::Arc};

use crate::schemas::messages::Message;

use super::{Prompt, PromptTemplate};

/// Represents a generic template for formatting messages.
pub trait BaseMessagePromptTemplate: Send + Sync {
    /// Formats a message using the provided input variables.
    fn format(&self, input_variables: HashMap<&str, &str>) -> Result<Message, Box<dyn Error>>;

    fn format_messages(
        &self,
        input_variables: HashMap<&str, &str>,
    ) -> Result<Vec<Message>, Box<dyn Error>> {
        Ok(vec![self.format(input_variables)?])
    }

    /// Returns a list of required input variable names for the template.
    fn input_variables(&self) -> Vec<String>;
}

/// A template for creating human-readable messages.
pub struct HumanMessagePromptTemplate {
    prompt: Arc<PromptTemplate>,
}

impl HumanMessagePromptTemplate {
    pub fn new(prompt: Arc<PromptTemplate>) -> Self {
        Self { prompt }
    }
}

impl BaseMessagePromptTemplate for HumanMessagePromptTemplate {
    fn format(&self, input_variables: HashMap<&str, &str>) -> Result<Message, Box<dyn Error>> {
        let text = self.prompt.format(input_variables)?;
        Ok(Message::new_human_message(&text))
    }

    fn input_variables(&self) -> Vec<String> {
        self.prompt.variables().clone()
    }
}

/// A template for creating system messages.
pub struct SystemMessagePromptTemplate {
    prompt: Arc<PromptTemplate>,
}

impl SystemMessagePromptTemplate {
    pub fn new(prompt: Arc<PromptTemplate>) -> Self {
        Self { prompt }
    }
}

impl BaseMessagePromptTemplate for SystemMessagePromptTemplate {
    fn format(&self, input_variables: HashMap<&str, &str>) -> Result<Message, Box<dyn Error>> {
        let text = self.prompt.format(input_variables)?;
        Ok(Message::new_system_message(&text))
    }

    fn input_variables(&self) -> Vec<String> {
        self.prompt.variables().clone()
    }
}

/// A template for creating AI (assistant) messages.
pub struct AIMessagePromptTemplate {
    prompt: Arc<PromptTemplate>,
}

impl AIMessagePromptTemplate {
    pub fn new(prompt: Arc<PromptTemplate>) -> Arc<Self> {
        Arc::new(Self { prompt })
    }
}

impl BaseMessagePromptTemplate for AIMessagePromptTemplate {
    fn format(&self, input_variables: HashMap<&str, &str>) -> Result<Message, Box<dyn Error>> {
        let text = self.prompt.format(input_variables)?;
        Ok(Message::new_ai_message(&text))
    }

    fn input_variables(&self) -> Vec<String> {
        self.prompt.variables().clone()
    }
}

pub enum MessageOrTemplate {
    Message(Message),
    Template(Arc<dyn BaseMessagePromptTemplate>),
    MessagesPlaceholder(MessagesPlaceholder),
}

#[derive(Clone)]
pub struct MessagesPlaceholder {
    messages: Vec<Message>,
}

impl MessagesPlaceholder {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn format(&self) -> Vec<Message> {
        self.messages.clone()
    }
}

pub struct MessageFormatter {
    items: Vec<MessageOrTemplate>,
}

impl MessageFormatter {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn add_message(&mut self, message: Message) {
        self.items.push(MessageOrTemplate::Message(message));
    }

    pub fn add_template(&mut self, template: Arc<dyn BaseMessagePromptTemplate>) {
        self.items.push(MessageOrTemplate::Template(template));
    }

    pub fn add_messages_placeholder(&mut self, placeholder: MessagesPlaceholder) {
        self.items
            .push(MessageOrTemplate::MessagesPlaceholder(placeholder));
    }

    pub fn format(
        &self,
        input_variables: HashMap<&str, &str>,
    ) -> Result<Vec<Message>, Box<dyn Error>> {
        let mut result = Vec::new();
        for item in &self.items {
            match item {
                MessageOrTemplate::Message(msg) => result.push(msg.clone()),
                MessageOrTemplate::Template(tmpl) => {
                    result.extend(tmpl.format_messages(input_variables.clone())?)
                }
                MessageOrTemplate::MessagesPlaceholder(placeholder) => {
                    result.extend(placeholder.format())
                }
            }
        }
        Ok(result)
    }
}

#[macro_export]
macro_rules! messages_placeholder {
    ($($msg:expr),* $(,)?) => {{
        let mut placeholder = crate::prompt::chat::MessagesPlaceholder::new();
        $(
            placeholder.add_message($msg);
        )*
        MessageOrTemplate::MessagesPlaceholder(placeholder)
    }};
}

#[macro_export]
macro_rules! message_formatter {
    ($($item:expr),* $(,)?) => {{
        let mut formatter = crate::prompt::chat::MessageFormatter::new();
        $(
            match $item {
                MessageOrTemplate::Message(msg) => formatter.add_message(msg),
                MessageOrTemplate::Template(tmpl) => formatter.add_template(tmpl),
                MessageOrTemplate::MessagesPlaceholder(placeholder) => formatter.add_messages_placeholder(placeholder.clone()),
            }
        )*
        formatter
    }};
}

#[cfg(test)]
mod tests {
    use crate::{
        message_formatter, messages_placeholder,
        prompt::{
            chat::{AIMessagePromptTemplate, MessageOrTemplate},
            PromptTemplate, TemplateFormat,
        },
        prompt_args,
        schemas::messages::Message,
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

        // Create a placeholder for multiple messages
        let messages_placeholder = messages_placeholder![
            Message::new_human_message("Placeholder message 1"),
            Message::new_system_message("Placeholder message 2"),
        ];

        // Use the `message_formatter` macro to construct the formatter
        let formatter = message_formatter![
            MessageOrTemplate::Message(human_msg),
            MessageOrTemplate::Template(ai_message_prompt),
            messages_placeholder,
        ];

        // Define input variables for the AI message template
        let input_variables = prompt_args! {
            "content" => "This is a test",
            "test" => "test2",

        };

        // Format messages
        let formatted_messages = formatter.format(input_variables).unwrap();

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
}
