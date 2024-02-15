use std::error::Error;

use super::messages::Message;

pub trait PromptValue: Send + Sync {
    fn to_string(&self) -> Result<String, Box<dyn Error>>;
    fn to_chat_messages(&self) -> Result<Vec<Message>, Box<dyn Error>>;
}
