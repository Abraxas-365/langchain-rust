mod chat;
mod error;
mod prompt;

use std::collections::HashMap;

pub use chat::*;
pub use error::*;
pub use prompt::*;
use serde_json::Value;

use crate::schemas::{messages::Message, prompt::PromptValue};

// pub type PromptArgs<'a> = HashMap<&'a str, &'a str>;
pub type PromptArgs = HashMap<String, Value>;
pub trait PromptFromatter: Send + Sync {
    fn template(&self) -> String;
    fn variables(&self) -> Vec<String>;
    fn format(&self, input_variables: PromptArgs) -> Result<String, PromptError>;
}
impl<PA> From<PA> for Box<dyn PromptFromatter>
where
    PA: PromptFromatter + 'static,
{
    fn from(prompt: PA) -> Self {
        Box::new(prompt)
    }
}

/// Represents a generic template for formatting messages.
pub trait MessageFormatter: Send + Sync {
    fn format_messages(&self, input_variables: PromptArgs) -> Result<Vec<Message>, PromptError>;

    /// Returns a list of required input variable names for the template.
    fn input_variables(&self) -> Vec<String>;
}
impl<MF> From<MF> for Box<dyn MessageFormatter>
where
    MF: MessageFormatter + 'static,
{
    fn from(prompt: MF) -> Self {
        Box::new(prompt)
    }
}

pub trait FormatPrompter: Send + Sync {
    fn format_prompt(&self, input_variables: PromptArgs) -> Result<PromptValue, PromptError>;
    fn get_input_variables(&self) -> Vec<String>;
}
impl<FP> From<FP> for Box<dyn FormatPrompter>
where
    FP: FormatPrompter + 'static,
{
    fn from(prompt: FP) -> Self {
        Box::new(prompt)
    }
}
