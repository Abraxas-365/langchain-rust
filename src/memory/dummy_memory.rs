use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::schemas::{
    memory::{BaseMemory, MemoryError},
    messages::Message,
};

pub struct DummyMemory {}

impl DummyMemory {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for DummyMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl Into<Arc<dyn BaseMemory>> for DummyMemory {
    fn into(self) -> Arc<dyn BaseMemory> {
        Arc::new(self)
    }
}

impl Into<Arc<Mutex<dyn BaseMemory>>> for DummyMemory {
    fn into(self) -> Arc<Mutex<dyn BaseMemory>> {
        Arc::new(Mutex::new(self))
    }
}

#[async_trait]
impl BaseMemory for DummyMemory {
    async fn messages(&self) -> Result<Vec<Message>, MemoryError> {
        Ok(vec![])
    }

    async fn add_message(&mut self, _message: Message) -> Result<(), MemoryError> {
        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn never_stores_messages() {
        let mut memory = DummyMemory::new();
        memory.add_user_message(&"ignored").await.unwrap();
        memory.add_ai_message(&"ignored").await.unwrap();
        assert!(memory.messages().await.unwrap().is_empty());
        assert_eq!(memory.to_string().await.unwrap(), "");
    }
}
