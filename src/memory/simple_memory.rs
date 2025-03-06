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
