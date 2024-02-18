use std::error::Error;
use std::sync::Arc;

use super::{PromptArgs, PromptFromatter};

pub enum TemplateFormat {
    FString,
    Jinja2,
}

pub struct PromptTemplate {
    template: String,
    variables: Vec<String>,
    format: TemplateFormat,
}

impl PromptTemplate {
    pub fn new(template: String, variables: Vec<String>, format: TemplateFormat) -> Self {
        Self {
            template,
            variables,
            format,
        }
    }
}

impl PromptFromatter for PromptTemplate {
    fn template(&self) -> String {
        self.template.clone()
    }

    fn variables(&self) -> Vec<String> {
        self.variables.clone()
    }

    fn format(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>> {
        let mut prompt = self.template();

        // check if all variables are in the input variables
        for key in self.variables() {
            if !input_variables.contains_key(key.as_str()) {
                return Err(format!("Variable {} is missing from input variables", key).into());
            }
        }

        for (key, value) in input_variables {
            let key = match self.format {
                TemplateFormat::FString => format!("{{{}}}", key),
                TemplateFormat::Jinja2 => format!("{{{{{}}}}}", key),
            };
            prompt = prompt.replace(&key, &value);
        }

        Ok(prompt)
    }
}

#[macro_export]
macro_rules! prompt_args {
    ( $($key:expr => $value:expr),* $(,)? ) => {
        {
            #[allow(unused_mut)]
            let mut args = std::collections::HashMap::<String, String>::new();
            $(
                args.insert($key.to_string(), $value.to_string());
            )*
            args
        }
    };
}

#[macro_export]
macro_rules! template_fstring {
    ($template:expr, $($var:expr),* $(,)?) => {
        crate::prompt::PromptTemplate::new(
            $template.to_string(),
            vec![$($var.to_string()),*],
            crate::prompt::TemplateFormat::FString,
        )
    };
}

#[macro_export]
macro_rules! template_jinja2 {
    ($template:expr, $($var:expr),* $(,)?) => {
        PromptTemplate::new(
            $template.to_string(),
            vec![$($var.to_string()),*],
            TemplateFormat::Jinja2,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt_args;

    #[test]
    fn should_format_jinja2_template() {
        let template = PromptTemplate::new(
            "Hello {{name}}!".to_string(),
            vec!["name".to_string()],
            TemplateFormat::Jinja2,
        );

        let input_variables = prompt_args! {};
        let result = template.format(input_variables);
        assert!(result.is_err());

        let input_variables = prompt_args! {
            "name" => "world",
        };
        let result = template.format(input_variables);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello world!");
    }

    #[test]
    fn should_format_fstring_template() {
        let template = PromptTemplate::new(
            "Hello {name}!".to_string(),
            vec!["name".to_string()],
            TemplateFormat::FString,
        );

        let input_variables = prompt_args! {};
        let result = template.format(input_variables);
        assert!(result.is_err());

        let input_variables = prompt_args! {
            "name" => "world",
        };
        let result = template.format(input_variables);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Hello world!");
    }

    #[test]
    fn should_prompt_macro_work() {
        let args = prompt_args! {};
        assert!(args.is_empty());

        let args = prompt_args! {
            "name" => "world",
        };
        assert_eq!(args.len(), 1);
        assert_eq!(args.get("name").unwrap(), &"world");

        let args = prompt_args! {
            "name" => "world",
            "age" => "18",
        };
        assert_eq!(args.len(), 2);
        assert_eq!(args.get("name").unwrap(), &"world");
        assert_eq!(args.get("age").unwrap(), &"18");
    }

    #[test]
    fn test_chat_template_macros() {
        // Creating an FString chat template
        let fstring_template = template_fstring!(
            "FString Chat: {user} says {message} {test}",
            "user",
            "message",
            "test"
        );

        // Creating a Jinja2 chat template
        let jinja2_template =
            template_jinja2!("Jinja2 Chat: {{user}} says {{message}}", "user", "message");

        // Define input variables for the templates
        let input_variables_fstring = prompt_args! {
            "user" => "Alice",
            "message" => "Hello, Bob!",
            "test"=>"test2"
        };

        let input_variables_jinja2 = prompt_args! {
            "user" => "Bob",
            "message" => "Hi, Alice!",
        };

        // Format the FString chat template
        let formatted_fstring = fstring_template.format(input_variables_fstring).unwrap();
        assert_eq!(
            formatted_fstring,
            "FString Chat: Alice says Hello, Bob! test2"
        );

        // Format the Jinja2 chat template
        let formatted_jinja2 = jinja2_template.format(input_variables_jinja2).unwrap();
        assert_eq!(formatted_jinja2, "Jinja2 Chat: Bob says Hi, Alice!");
    }
}
