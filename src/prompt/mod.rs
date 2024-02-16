mod chat;
mod prompt;

use std::{collections::HashMap, error::Error};

pub use chat::*;
pub use prompt::*;

use crate::schemas::{messages::Message, prompt::PromptValue};

pub type PromptArgs<'a> = HashMap<&'a str, &'a str>;
pub trait PromptFromatter: Send + Sync {
    fn template(&self) -> String;
    fn variables(&self) -> Vec<String>;
    fn format(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>>;
}

/// Represents a generic template for formatting messages.
pub trait MessageFormatter: Send + Sync {
    fn format_messages(&self, input_variables: PromptArgs) -> Result<Vec<Message>, Box<dyn Error>>;

    /// Returns a list of required input variable names for the template.
    fn input_variables(&self) -> Vec<String>;
}

pub trait FormatPrompter {
    fn format_prompt(&self, input_variables: PromptArgs) -> Result<PromptValue, Box<dyn Error>>;
    fn get_input_variables(&self) -> Vec<String>;
}
