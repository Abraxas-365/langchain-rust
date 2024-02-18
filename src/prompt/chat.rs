use std::error::Error;

use crate::schemas::{messages::Message, prompt::PromptValue};

use super::{FormatPrompter, MessageFormatter, PromptArgs, PromptFromatter, PromptTemplate};

/// A template for creating human-readable messages.
pub struct HumanMessagePromptTemplate {
    prompt: PromptTemplate,
}

impl Into<Box<dyn MessageFormatter>> for HumanMessagePromptTemplate {
    fn into(self) -> Box<dyn MessageFormatter> {
        Box::new(self)
    }
}

impl HumanMessagePromptTemplate {
    pub fn new(prompt: PromptTemplate) -> Self {
        Self { prompt }
    }
}
impl MessageFormatter for HumanMessagePromptTemplate {
    fn format_messages(&self, input_variables: PromptArgs) -> Result<Vec<Message>, Box<dyn Error>> {
        Ok(vec![Message::new_human_message(
            &self.prompt.format(input_variables)?,
        )])
    }
    fn input_variables(&self) -> Vec<String> {
        self.prompt.variables().clone()
    }
}

/// A template for creating system messages.
pub struct SystemMessagePromptTemplate {
    prompt: PromptTemplate,
}

impl Into<Box<dyn MessageFormatter>> for SystemMessagePromptTemplate {
    fn into(self) -> Box<dyn MessageFormatter> {
        Box::new(self)
    }
}

impl SystemMessagePromptTemplate {
    pub fn new(prompt: PromptTemplate) -> Self {
        Self { prompt }
    }
}
impl MessageFormatter for SystemMessagePromptTemplate {
    fn format_messages(&self, input_variables: PromptArgs) -> Result<Vec<Message>, Box<dyn Error>> {
        Ok(vec![Message::new_system_message(
            &self.prompt.format(input_variables)?,
        )])
    }
    fn input_variables(&self) -> Vec<String> {
        self.prompt.variables().clone()
    }
}

/// A template for creating AI (assistant) messages.
pub struct AIMessagePromptTemplate {
    prompt: PromptTemplate,
}

impl Into<Box<dyn MessageFormatter>> for AIMessagePromptTemplate {
    fn into(self) -> Box<dyn MessageFormatter> {
        Box::new(self)
    }
}

impl MessageFormatter for AIMessagePromptTemplate {
    fn format_messages(&self, input_variables: PromptArgs) -> Result<Vec<Message>, Box<dyn Error>> {
        Ok(vec![Message::new_ai_message(
            &self.prompt.format(input_variables)?,
        )])
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
}

impl MessageFormatter for MessagesPlaceholder {
    fn format_messages(
        &self,
        _input_variables: PromptArgs,
    ) -> Result<Vec<Message>, Box<dyn Error>> {
        Ok(self.messages.clone())
    }
    fn input_variables(&self) -> Vec<String> {
        Vec::new()
    }
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

    pub fn add_messages_placeholder(&mut self, placeholder: MessagesPlaceholder) {
        self.items
            .push(MessageOrTemplate::MessagesPlaceholder(placeholder));
    }

    fn format(&self, input_variables: PromptArgs) -> Result<Vec<Message>, Box<dyn Error>> {
        let mut result = Vec::new();
        for item in &self.items {
            match item {
                MessageOrTemplate::Message(msg) => result.push(msg.clone()),
                MessageOrTemplate::Template(tmpl) => {
                    result.extend(tmpl.format_messages(input_variables.clone())?)
                }
                MessageOrTemplate::MessagesPlaceholder(placeholder) => {
                    result.extend(placeholder.format_messages(input_variables.clone())?)
                }
            }
        }
        Ok(result)
    }
}

impl MessageFormatter for MessageFormatterStruct {
    fn format_messages(&self, input_variables: PromptArgs) -> Result<Vec<Message>, Box<dyn Error>> {
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
                    variables.extend(placeholder.input_variables());
                }
            }
        }
        variables
    }
}

impl FormatPrompter for MessageFormatterStruct {
    fn format_prompt(&self, input_variables: PromptArgs) -> Result<PromptValue, Box<dyn Error>> {
        let messages = self.format(input_variables)?;
        Ok(PromptValue::from_messages(messages))
    }
    fn get_input_variables(&self) -> Vec<String> {
        self.input_variables()
    }
}

#[macro_export]
macro_rules! messages_placeholder {
    ($($msg:expr),* $(,)?) => {{
        let mut placeholder = crate::prompt::MessagesPlaceholder::new();
        $(
            placeholder.add_message($msg);
        )*
        MessageOrTemplate::MessagesPlaceholder(placeholder)
    }};
}

#[macro_export]
macro_rules! message_formatter {
    ($($item:expr),* $(,)?) => {{
        let mut formatter = crate::prompt::MessageFormatterStruct::new();
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
            MessageFormatter,
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
            MessageOrTemplate::Template(ai_message_prompt.into()),
            messages_placeholder,
        ];

        // Define input variables for the AI message template
        let input_variables = prompt_args! {
            "content" => "This is a test",
            "test" => "test2",

        };

        // Format messages
        let formatted_messages = formatter.format_messages(input_variables).unwrap();

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
