use std::sync::Arc;

use tokio::sync::Mutex;

use crate::schemas::{memory::BaseMemory, messages::Message};

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

impl BaseMemory for DummyMemory {
    fn messages(&self) -> Vec<Message> {
        vec![]
    }
    fn add_message(&mut self, _message: Message) {}
    fn clear(&mut self) {}
}
