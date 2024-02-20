use std::sync::Arc;

use tokio::sync::Mutex;

use super::messages::Message;

pub trait BaseMemory: Send + Sync {
    fn messages(&self) -> Vec<Message>;

    // Use a trait object for Display instead of a generic type
    fn add_user_message(&mut self, message: &dyn std::fmt::Display) {
        // Convert the Display trait object to a String and pass it to the constructor
        self.add_message(Message::new_human_message(&message.to_string()));
    }

    // Use a trait object for Display instead of a generic type
    fn add_ai_message(&mut self, message: &dyn std::fmt::Display) {
        // Convert the Display trait object to a String and pass it to the constructor
        self.add_message(Message::new_ai_message(&message.to_string()));
    }

    fn add_message(&mut self, message: Message);

    fn clear(&mut self);

    fn to_string(&self) -> String {
        self.messages()
            .iter()
            .map(|msg| format!("{}: {}", msg.message_type.to_string(), msg.content))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

pub struct SimpleMemory {
    messages: Vec<Message>,
}

impl SimpleMemory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }
}

impl Into<Arc<dyn BaseMemory>> for SimpleMemory {
    fn into(self) -> Arc<dyn BaseMemory> {
        Arc::new(self)
    }
}

impl Into<Arc<Mutex<dyn BaseMemory>>> for SimpleMemory {
    fn into(self) -> Arc<Mutex<dyn BaseMemory>> {
        Arc::new(Mutex::new(self))
    }
}

impl BaseMemory for SimpleMemory {
    fn messages(&self) -> Vec<Message> {
        self.messages.clone()
    }
    fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }
    fn clear(&mut self) {
        self.messages.clear();
    }
}
