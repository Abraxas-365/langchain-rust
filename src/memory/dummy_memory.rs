use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::schemas::{memory::BaseMemory, messages::Message};

pub struct DummyMemory {}

impl DummyMemory {
    pub fn new() -> Self {
        Self {}
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
    async fn messages(&self) -> Vec<Message> {
        vec![]
    }
    async fn add_message(&mut self, _message: Message) {}
    async fn clear(&mut self) {}
}
