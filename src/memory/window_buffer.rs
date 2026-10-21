use std::sync::Arc;

use tokio::sync::Mutex;

use crate::schemas::{memory::BaseMemory, messages::Message};

pub struct WindowBufferMemory {
    window_size: usize,
    messages: Vec<Message>,
}

impl Default for WindowBufferMemory {
    fn default() -> Self {
        Self::new(10)
    }
}

impl WindowBufferMemory {
    pub fn new(window_size: usize) -> Self {
        Self {
            messages: Vec::new(),
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

impl BaseMemory for WindowBufferMemory {
    fn messages(&self) -> Vec<Message> {
        self.messages.clone()
    }
    fn add_message(&mut self, message: Message) {
        if self.window_size == 0 {
            return;
        }
        if self.messages.len() >= self.window_size {
            self.messages.remove(0);
        }
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
    fn keeps_only_last_window_size_messages() {
        let mut memory = WindowBufferMemory::new(2);
        memory.add_user_message(&"one");
        memory.add_ai_message(&"two");
        memory.add_user_message(&"three");

        let messages = memory.messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "two");
        assert_eq!(messages[0].message_type, MessageType::AIMessage);
        assert_eq!(messages[1].content, "three");
        assert_eq!(messages[1].message_type, MessageType::HumanMessage);
    }

    #[test]
    fn default_window_is_ten() {
        let mut memory = WindowBufferMemory::default();
        for i in 0..15 {
            memory.add_user_message(&i);
        }
        let messages = memory.messages();
        assert_eq!(messages.len(), 10);
        assert_eq!(messages.first().unwrap().content, "5");
        assert_eq!(messages.last().unwrap().content, "14");
    }

    #[test]
    fn zero_window_stores_nothing_and_does_not_panic() {
        let mut memory = WindowBufferMemory::new(0);
        memory.add_user_message(&"hello");
        assert!(memory.messages().is_empty());
    }

    #[test]
    fn clear_and_to_string() {
        let mut memory = WindowBufferMemory::new(5);
        memory.add_user_message(&"hi");
        memory.add_ai_message(&"hello");
        assert_eq!(memory.to_string(), "human: hi\nai: hello");
        memory.clear();
        assert!(memory.messages().is_empty());
        assert_eq!(memory.to_string(), "");
    }
}
