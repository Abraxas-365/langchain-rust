use super::messages::Message;

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

    pub fn to_string(&self) -> String {
        self.messages
            .iter()
            .map(|m| format!("{}: {}", m.message_type.to_string(), m.content))
            .collect::<Vec<String>>()
            .join("\n")
    }

    pub fn to_chat_messages(&self) -> Vec<Message> {
        self.messages.clone()
    }
}
