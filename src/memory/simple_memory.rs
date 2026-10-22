use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::schemas::{
    memory::{BaseMemory, MemoryError},
    messages::Message,
};

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

impl Default for SimpleMemory {
    fn default() -> Self {
        Self::new()
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

#[async_trait]
impl BaseMemory for SimpleMemory {
    async fn messages(&self) -> Result<Vec<Message>, MemoryError> {
        Ok(self.messages.clone())
    }

    async fn add_message(&mut self, message: Message) -> Result<(), MemoryError> {
        self.messages.push(message);
        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        self.messages.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas::MessageType;

    #[tokio::test]
    async fn stores_all_messages_in_order() {
        let mut memory = SimpleMemory::new();
        memory.add_user_message(&"question").await.unwrap();
        memory.add_ai_message(&"answer").await.unwrap();
        memory
            .add_message(Message::new_system_message("sys"))
            .await
            .unwrap();

        let messages = memory.messages().await.unwrap();
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
            memory.to_string().await.unwrap(),
            "human: question\nai: answer\nsystem: sys"
        );
    }

    #[tokio::test]
    async fn clear_empties_memory() {
        let mut memory = SimpleMemory::new();
        memory.add_user_message(&"x").await.unwrap();
        memory.clear().await.unwrap();
        assert!(memory.messages().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn converts_into_shared_mutex() {
        let memory: Arc<Mutex<dyn BaseMemory>> = SimpleMemory::new().into();
        memory
            .lock()
            .await
            .add_user_message(&"shared")
            .await
            .unwrap();
        assert_eq!(memory.lock().await.messages().await.unwrap().len(), 1);
    }
}
