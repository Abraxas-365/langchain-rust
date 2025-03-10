mod chat;
mod error;
mod plain_prompt_args;
#[allow(clippy::module_inception)]
mod prompt;

pub use chat::*;
pub use error::*;
pub use plain_prompt_args::*;
pub use prompt::*;

use crate::schemas::{messages::Message, prompt::PromptValue};

pub trait PromptArgs: Send + Sync + Clone {
    fn contains_key(&self, key: &str) -> bool;

    fn get(&self, key: &str) -> Option<&str>;

    fn insert(&mut self, key: String, value: String) -> Option<String>;

    fn iter(&self) -> Box<dyn Iterator<Item = (&String, &String)> + '_>;
}

pub trait PromptFromatter<T>: Send + Sync
where
    T: PromptArgs,
{
    fn template(&self) -> String;
    fn variables(&self) -> Vec<String>;
    fn format(&self, input_variables: &T) -> Result<String, PromptError>;
}

/// Represents a generic template for formatting messages.
pub trait MessageFormatter<T>: Send + Sync
where
    T: PromptArgs,
{
    fn format_messages(&self, input_variables: &T) -> Result<Vec<Message>, PromptError>;

    /// Returns a list of required input variable names for the template.
    fn input_variables(&self) -> Vec<String>;
}

pub trait FormatPrompter<T>: Send + Sync
where
    T: PromptArgs,
{
    fn format_prompt(&self, input_variables: &T) -> Result<PromptValue, PromptError>;
    fn get_input_variables(&self) -> Vec<String>;
}
