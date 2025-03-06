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

impl From<SimpleMemory> for Arc<dyn BaseMemory> {
    fn from(val: SimpleMemory) -> Self {
        Arc::new(val)
    }
}

impl From<SimpleMemory> for Arc<Mutex<dyn BaseMemory>> {
    fn from(val: SimpleMemory) -> Self {
        Arc::new(Mutex::new(val))
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
