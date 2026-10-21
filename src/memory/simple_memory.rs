use std::sync::Arc;

use tokio::sync::Mutex;

use crate::schemas::{memory::BaseMemory, messages::Message};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas::MessageType;

    #[test]
    fn stores_all_messages_in_order() {
        let mut memory = SimpleMemory::new();
        memory.add_user_message(&"question");
        memory.add_ai_message(&"answer");
        memory.add_message(Message::new_system_message("sys"));

        let messages = memory.messages();
        let types: Vec<_> = messages.iter().map(|m| m.message_type.clone()).collect();
        assert_eq!(
            types,
            vec![
                MessageType::HumanMessage,
                MessageType::AIMessage,
                MessageType::SystemMessage
            ]
        );
        assert_eq!(
            memory.to_string(),
            "human: question\nai: answer\nsystem: sys"
        );
    }

    #[test]
    fn clear_empties_memory() {
        let mut memory = SimpleMemory::new();
        memory.add_user_message(&"x");
        memory.clear();
        assert!(memory.messages().is_empty());
    }

    #[tokio::test]
    async fn converts_into_shared_mutex() {
        let memory: Arc<Mutex<dyn BaseMemory>> = SimpleMemory::new().into();
        memory.lock().await.add_user_message(&"shared");
        assert_eq!(memory.lock().await.messages().len(), 1);
    }
}
