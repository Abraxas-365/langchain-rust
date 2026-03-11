use std::collections::VecDeque;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::schemas::{memory::BaseMemory, messages::Message};

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

impl BaseMemory for WindowBufferMemory {
    fn messages(&self) -> Vec<Message> {
        self.messages.iter().cloned().collect()
    }
    fn add_message(&mut self, message: Message) {
        if self.window_size > 0 && self.messages.len() >= self.window_size {
            self.messages.pop_front();
        }
        self.messages.push_back(message);
    }
    fn clear(&mut self) {
        self.messages.clear();
    }
}
