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

impl From<WindowBufferMemory> for Arc<dyn BaseMemory> {
    fn from(val: WindowBufferMemory) -> Self {
        Arc::new(val)
    }
}

impl From<WindowBufferMemory> for Arc<Mutex<dyn BaseMemory>> {
    fn from(val: WindowBufferMemory) -> Self {
        Arc::new(Mutex::new(val))
    }
}

impl BaseMemory for WindowBufferMemory {
    fn messages(&self) -> Vec<Message> {
        self.messages.clone()
    }
    fn add_message(&mut self, message: Message) {
        if self.messages.len() >= self.window_size {
            self.messages.remove(0);
        }
        self.messages.push(message);
    }
    fn clear(&mut self) {
        self.messages.clear();
    }
}
