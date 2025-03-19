use std::collections::HashSet;

use derive_new::new;
use gix::hashtable::HashMap;

use crate::prompt::PromptError;

use super::{InputVariables, Message, MessageTemplate, Prompt};

#[derive(Clone)]
pub enum MessageOrTemplate {
    Message(Message),
    Template(MessageTemplate),
    Placeholder(String),
}

#[derive(new)]
pub struct PromptTemplate {
    messages: Vec<MessageOrTemplate>,
}

impl PromptTemplate {
    pub fn insert_message(&mut self, message: Message) {
        self.messages.push(MessageOrTemplate::Message(message));
    }

    pub fn insert_template(&mut self, template: MessageTemplate) {
        self.messages.push(MessageOrTemplate::Template(template));
    }

    pub fn insert_placeholder(&mut self, placeholder: String) {
        self.messages
            .push(MessageOrTemplate::Placeholder(placeholder));
    }

    /// Insert variables into a prompt template to create a full-fletched prompt.
    ///
    /// replace_placeholder() must be called before format().
    pub fn format(&self, input_variables: &InputVariables) -> Result<Prompt, PromptError> {
        let messages = self
            .messages
            .iter()
            .filter_map(|m| -> Option<Result<Message, PromptError>> {
                match m {
                    MessageOrTemplate::Message(m) => Some(Ok(m.clone())),
                    MessageOrTemplate::Template(t) => Some(t.format(input_variables)),
                    MessageOrTemplate::Placeholder(_) => None,
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Prompt::new(messages))
    }

    /// Replaces placeholder messages with the actual messages.
    ///
    /// Must be called before .format()
    pub fn replace_placeholder(&self, input: &HashMap<String, Vec<Message>>) -> Self {
        let messages = self
            .messages
            .iter()
            .flat_map(|m| match m {
                MessageOrTemplate::Placeholder(p) if input.get(p).is_some() => input
                    .get(p)
                    .unwrap()
                    .iter()
                    .map(|m| MessageOrTemplate::Message(m.clone()))
                    .collect(),
                _ => vec![m.clone()],
            })
            .collect();

        Self::new(messages)
    }

    /// Returns a list of required input variable names for the template.
    pub fn variables(&self) -> HashSet<String> {
        let variables = self
            .messages
            .iter()
            .flat_map(|m| match m {
                MessageOrTemplate::Template(t) => t.variables(),
                _ => HashSet::new(),
            })
            .collect();

        variables
    }

    pub fn placeholders(&self) -> HashSet<String> {
        let placeholders = self
            .messages
            .iter()
            .filter_map(|m| match m {
                MessageOrTemplate::Placeholder(p) => Some(p.clone()),
                _ => None,
            })
            .collect();

        placeholders
    }
}

impl From<MessageTemplate> for PromptTemplate {
    fn from(template: MessageTemplate) -> Self {
        Self::new(vec![MessageOrTemplate::Template(template)])
    }
}

impl From<Message> for MessageOrTemplate {
    fn from(message: Message) -> Self {
        MessageOrTemplate::Message(message)
    }
}

impl From<MessageTemplate> for MessageOrTemplate {
    fn from(template: MessageTemplate) -> Self {
        MessageOrTemplate::Template(template)
    }
}

#[macro_export]
macro_rules! prompt_template {
    ($($x:expr),*) => {
        $crate::schemas::PromptTemplate::new(vec![$($x.into()),*])
    };
}
