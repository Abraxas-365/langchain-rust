use std::sync::Arc;

use tokio::sync::Mutex;

use crate::schemas::{memory::BaseMemory, messages::Message};

pub struct DummyMemroy {}

impl DummyMemroy {
    pub fn new() -> Self {
        Self {}
    }
}

impl Into<Arc<dyn BaseMemory>> for DummyMemroy {
    fn into(self) -> Arc<dyn BaseMemory> {
        Arc::new(self)
    }
}

impl Into<Arc<Mutex<dyn BaseMemory>>> for DummyMemroy {
    fn into(self) -> Arc<Mutex<dyn BaseMemory>> {
        Arc::new(Mutex::new(self))
    }
}

impl BaseMemory for DummyMemroy {
    fn messages(&self) -> Vec<Message> {
        vec![]
    }
    fn add_message(&mut self, _message: Message) {}
    fn clear(&mut self) {}
}
