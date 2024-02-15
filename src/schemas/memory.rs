use super::messages::Message;

pub trait BaseChatMessageHistory: Send + Sync {
    fn messages(&self) -> Vec<Message>;

    fn add_user_message(&mut self, message: &str) {
        self.add_message(Message::new_ai_message(message));
    }

    fn add_ai_message(&mut self, message: &str) {
        self.add_message(Message::new_ai_message(message));
    }

    fn add_message(&mut self, message: Message);

    fn clear(&mut self);

    fn to_string(&self) -> String {
        self.messages()
            .iter()
            .map(|msg| format!("{:?}:{:?}", msg.message_type, msg.content))
            .collect::<Vec<String>>()
            .join("\n")
    }
}
