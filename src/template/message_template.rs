use std::collections::HashSet;

use derive_new::new;

use crate::schemas::{InputVariables, Message, MessageType};
use crate::template::TemplateError;

#[derive(Clone)]
pub enum TemplateFormat {
    FString,
    Jinja2,
}

#[derive(Clone, new)]
pub struct MessageTemplate {
    message_type: MessageType,
    template: String,
    variables: HashSet<String>,
    format: TemplateFormat,
}

impl MessageTemplate {
    pub fn from_fstring(message_type: MessageType, content: &str) -> Self {
        let re = regex::Regex::new(r"\{(\w+)\}").unwrap();
        let variables = re
            .captures_iter(content)
            .map(|cap| cap[1].to_string())
            .collect();

        Self::new(
            message_type,
            content.into(),
            variables,
            TemplateFormat::FString,
        )
    }

    pub fn from_jinja2(message_type: MessageType, content: &str) -> Self {
        let re = regex::Regex::new(r"\{\{(\w+)\}\}").unwrap();
        let variables = re
            .captures_iter(content)
            .map(|cap| cap[1].to_string())
            .collect();

        Self::new(
            message_type,
            content.into(),
            variables,
            TemplateFormat::Jinja2,
        )
    }

    pub fn format(&self, input_variables: &InputVariables) -> Result<Message, TemplateError> {
        let mut content = self.template.clone();

        // check if all variables are in the input variables
        for key in &self.variables {
            if !input_variables.contains_text_key(key.as_str()) {
                return Err(TemplateError::MissingVariable(key.clone()));
            }
        }

        for (key, value) in input_variables.iter_test_replacements() {
            let key = match self.format {
                TemplateFormat::FString => format!("{{{}}}", key),
                TemplateFormat::Jinja2 => format!("{{{{{}}}}}", key),
            };
            content = content.replace(&key, value);
        }

        Ok(Message::new(self.message_type.clone(), &content))
    }

    /// Returns a list of required input variable names for the template.
    pub fn variables(&self) -> HashSet<String> {
        self.variables.clone()
    }
}
