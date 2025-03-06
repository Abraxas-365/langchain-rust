use std::fmt;

use super::messages::Message;

#[derive(Debug, Clone)]
pub struct PromptValue {
    messages: Vec<Message>,
}
impl PromptValue {
    pub fn from_string(text: &str) -> Self {
        let message = Message::new_human_message(text);
        Self {
            messages: vec![message],
        }
    }
    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    pub fn to_chat_messages(&self) -> Vec<Message> {
        self.messages.clone()
    }
}

impl fmt::Display for PromptValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for message in &self.messages {
            writeln!(f, "{message}")?;
        }
        Ok(())
    }
}
