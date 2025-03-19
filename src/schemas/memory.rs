use super::{messages::Message, MessageType};

pub trait BaseMemory: Send + Sync {
    fn messages(&self) -> Vec<Message>;

    // Use a trait object for Display instead of a generic type
    fn add_user_message(&mut self, message: &dyn std::fmt::Display) {
        // Convert the Display trait object to a String and pass it to the constructor
        self.add_message(Message::new(
            MessageType::HumanMessage,
            &message.to_string(),
        ));
    }

    // Use a trait object for Display instead of a generic type
    fn add_ai_message(&mut self, message: &dyn std::fmt::Display) {
        // Convert the Display trait object to a String and pass it to the constructor
        self.add_message(Message::new(MessageType::AIMessage, &message.to_string()));
    }

    fn add_message(&mut self, message: Message);

    fn clear(&mut self);

    fn to_string(&self) -> String;
}

impl<M> From<M> for Box<dyn BaseMemory>
where
    M: BaseMemory + 'static,
{
    fn from(memory: M) -> Self {
        Box::new(memory)
    }
}
