use crate::schemas::messages::BaseMessage;
use std::error::Error;

pub trait PromptValue: Send + Sync {
    fn to_string(&self) -> Result<String, Box<dyn Error>>;
    fn to_chat_messages(&self) -> Result<Vec<Box<dyn BaseMessage>>, Box<dyn Error>>;
}
