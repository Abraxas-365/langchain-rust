use async_trait::async_trait;
use thiserror::Error;

use super::messages::Message;

/// Errors returned by [`BaseMemory`] implementations (mostly persistent backends).
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Memory storage error: {0}")]
    Storage(String),

    #[error("Memory serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Memory error: {0}")]
    Other(String),
}

#[cfg(feature = "sqlx")]
impl From<sqlx::Error> for MemoryError {
    fn from(e: sqlx::Error) -> Self {
        MemoryError::Storage(e.to_string())
    }
}

#[cfg(feature = "redis")]
impl From<redis::RedisError> for MemoryError {
    fn from(e: redis::RedisError) -> Self {
        MemoryError::Storage(e.to_string())
    }
}

#[cfg(feature = "mongodb")]
impl From<mongodb::error::Error> for MemoryError {
    fn from(e: mongodb::error::Error) -> Self {
        MemoryError::Storage(e.to_string())
    }
}

/// Conversation memory used by chains and agents.
///
/// All methods are async and fallible so that memories can be backed by a database
/// or a remote service (see `SqliteMessageHistory` / `PostgresMessageHistory`).
#[async_trait]
pub trait BaseMemory: Send + Sync {
    async fn messages(&self) -> Result<Vec<Message>, MemoryError>;

    async fn add_user_message(
        &mut self,
        message: &(dyn std::fmt::Display + Sync),
    ) -> Result<(), MemoryError> {
        self.add_message(Message::new_human_message(message.to_string()))
            .await
    }

    async fn add_ai_message(
        &mut self,
        message: &(dyn std::fmt::Display + Sync),
    ) -> Result<(), MemoryError> {
        self.add_message(Message::new_ai_message(message.to_string()))
            .await
    }

    async fn add_message(&mut self, message: Message) -> Result<(), MemoryError>;

    async fn clear(&mut self) -> Result<(), MemoryError>;

    async fn to_string(&self) -> Result<String, MemoryError> {
        Ok(self
            .messages()
            .await?
            .iter()
            .map(|msg| format!("{}: {}", msg.message_type.to_string(), msg.content))
            .collect::<Vec<String>>()
            .join("\n"))
    }
}

impl<M> From<M> for Box<dyn BaseMemory>
where
    M: BaseMemory + 'static,
{
    fn from(memory: M) -> Self {
        Box::new(memory)
    }
}
