use super::messages::{AIMessage, BaseMessage, HumanMessage};

pub trait BaseChatMessageHistory: Send + Sync {
    fn messages(&self) -> Vec<Box<dyn BaseMessage>>;

    fn add_user_message(&mut self, message: &str) {
        self.add_message(Box::new(HumanMessage {
            content: message.to_string(),
        }));
    }

    fn add_ai_message(&mut self, message: &str) {
        self.add_message(Box::new(AIMessage {
            content: message.to_string(),
        }));
    }

    fn add_message(&mut self, message: Box<dyn BaseMessage>);

    fn clear(&mut self);

    fn to_string(&self) -> String {
        self.messages()
            .iter()
            .map(|msg| format!("{}:{}", msg.get_type(), msg.get_content()))
            .collect::<Vec<String>>()
            .join("\n")
    }
}
