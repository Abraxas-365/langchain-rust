use super::messages::Message;
use async_trait::async_trait;
use std::future::Future;

#[async_trait]
pub trait BaseMemory: Send + Sync {
    async fn messages(&self) -> Vec<Message>;

    // Use String to maintain object safety while ensuring Send+Sync
    async fn add_user_message(&mut self, message: String) {
        self.add_message(Message::new_human_message(message)).await;
    }

    // Use String to maintain object safety while ensuring Send+Sync
    async fn add_ai_message(&mut self, message: String) {
        self.add_message(Message::new_ai_message(message)).await;
    }

    async fn add_message(&mut self, message: Message);

    async fn clear(&mut self);

    async fn to_string(&self) -> String {
        let messages = self.messages().await;
        messages
            .iter()
            .map(|msg| format!("{}: {}", msg.message_type.to_string(), msg.content))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

// Extension trait to provide generic methods that would otherwise make BaseMemory non-object-safe
pub trait BaseMemoryExt: BaseMemory {
    // Add a user message from any type that implements Display and is Send
    fn add_user_message_display<'a, T: std::fmt::Display + Send + 'a>(
        &'a mut self, 
        message: T
    ) -> impl Future<Output = ()> + Send + 'a {
        async move {
            self.add_user_message(message.to_string()).await;
        }
    }
    
    // Add an AI message from any type that implements Display and is Send
    fn add_ai_message_display<'a, T: std::fmt::Display + Send + 'a>(
        &'a mut self, 
        message: T
    ) -> impl Future<Output = ()> + Send + 'a {
        async move {
            self.add_ai_message(message.to_string()).await;
        }
    }
}

// Implement the extension trait for all types that implement BaseMemory
impl<T: BaseMemory + ?Sized> BaseMemoryExt for T {}

impl<M> From<M> for Box<dyn BaseMemory>
where
    M: BaseMemory + 'static,
{
    fn from(memory: M) -> Self {
        Box::new(memory)
    }
}
