use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::schemas::{
    memory::{BaseMemory, MemoryError},
    messages::Message,
};

/// A sliding-window conversation memory that retains the last `window_size` messages.
///
/// Internally uses a [`VecDeque`] so that evicting the oldest message on overflow
/// is O(1) rather than the O(n) `Vec::remove(0)` it replaces.
pub struct WindowBufferMemory {
    window_size: usize,
    messages: VecDeque<Message>,
}

impl Default for WindowBufferMemory {
    fn default() -> Self {
        Self::new(10)
    }
}

impl WindowBufferMemory {
    pub fn new(window_size: usize) -> Self {
        Self {
            messages: VecDeque::new(),
            window_size,
        }
    }
}

impl Into<Arc<dyn BaseMemory>> for WindowBufferMemory {
    fn into(self) -> Arc<dyn BaseMemory> {
        Arc::new(self)
    }
}

impl Into<Arc<Mutex<dyn BaseMemory>>> for WindowBufferMemory {
    fn into(self) -> Arc<Mutex<dyn BaseMemory>> {
        Arc::new(Mutex::new(self))
    }
}

#[async_trait]
impl BaseMemory for WindowBufferMemory {
    async fn messages(&self) -> Result<Vec<Message>, MemoryError> {
        Ok(self.messages.iter().cloned().collect())
    }

    async fn add_message(&mut self, message: Message) -> Result<(), MemoryError> {
        if self.window_size == 0 {
            return Ok(());
        }
        if self.messages.len() >= self.window_size {
            self.messages.pop_front();
        }
        self.messages.push_back(message);
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
    async fn keeps_only_last_window_size_messages() {
        let mut memory = WindowBufferMemory::new(2);
        memory.add_user_message(&"one").await.unwrap();
        memory.add_ai_message(&"two").await.unwrap();
        memory.add_user_message(&"three").await.unwrap();

        let messages = memory.messages().await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "two");
        assert_eq!(messages[0].message_type, MessageType::AIMessage);
        assert_eq!(messages[1].content, "three");
        assert_eq!(messages[1].message_type, MessageType::HumanMessage);
    }

    #[tokio::test]
    async fn default_window_is_ten() {
        let mut memory = WindowBufferMemory::default();
        for i in 0..15 {
            memory.add_user_message(&i).await.unwrap();
        }
        let messages = memory.messages().await.unwrap();
        assert_eq!(messages.len(), 10);
        assert_eq!(messages.first().unwrap().content, "5");
        assert_eq!(messages.last().unwrap().content, "14");
    }

    #[tokio::test]
    async fn zero_window_stores_nothing_and_does_not_panic() {
        let mut memory = WindowBufferMemory::new(0);
        memory.add_user_message(&"hello").await.unwrap();
        assert!(memory.messages().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn clear_and_to_string() {
        let mut memory = WindowBufferMemory::new(5);
        memory.add_user_message(&"hi").await.unwrap();
        memory.add_ai_message(&"hello").await.unwrap();
        assert_eq!(memory.to_string().await.unwrap(), "human: hi\nai: hello");
        memory.clear().await.unwrap();
        assert!(memory.messages().await.unwrap().is_empty());
        assert_eq!(memory.to_string().await.unwrap(), "");
    }
}
