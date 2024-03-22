use crate::schemas::{messages::Message, prompt::PromptValue};

use super::{FormatPrompter, PromptArgs, PromptError, PromptFromatter};

#[derive(Clone)]
pub enum TemplateFormat {
    FString,
    Jinja2,
}

#[derive(Clone)]
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

//PromptTemplate will be default transformed to an Human Input when used as FromatPrompter
impl FormatPrompter for PromptTemplate {
    fn format_prompt(&self, input_variables: PromptArgs) -> Result<PromptValue, PromptError> {
        let messages = vec![Message::new_human_message(self.format(input_variables)?)];
        Ok(PromptValue::from_messages(messages))
    }
    fn get_input_variables(&self) -> Vec<String> {
        self.variables.clone()
    }
}

impl PromptFromatter for PromptTemplate {
    fn template(&self) -> String {
        self.template.clone()
    }

    fn variables(&self) -> Vec<String> {
        self.variables.clone()
    }

    fn format(&self, input_variables: PromptArgs) -> Result<String, PromptError> {
        let mut prompt = self.template();

        // check if all variables are in the input variables
        for key in self.variables() {
            if !input_variables.contains_key(key.as_str()) {
                return Err(PromptError::MissingVariable(key));
            }
        }

        for (key, value) in input_variables {
            let key = match self.format {
                TemplateFormat::FString => format!("{{{}}}", key),
                TemplateFormat::Jinja2 => format!("{{{{{}}}}}", key),
            };
            let value_str = match &value {
                serde_json::Value::String(s) => s.clone(),
                _ => value.to_string(),
            };
            prompt = prompt.replace(&key, &value_str);
        }

        log::debug!("Formatted prompt: {}", prompt);
        Ok(prompt)
    }
}

/// `prompt_args!` is a utility macro used for creating a `std::collections::HashMap<String, serde_json::Value>`.
/// This HashMap can then be passed as arguments to a function or method.
///
/// # Usage
/// In this macro, the keys are `&str` and values are arbitrary types that get serialized into `serde_json::Value`:
/// ```rust,ignore
/// prompt_args! {
///     "input" => "Who is the writer of 20,000 Leagues Under the Sea, and what is my name?",
///     "history" => vec![
///         Message::new_human_message("My name is: Luis"),
///         Message::new_ai_message("Hi Luis"),
///     ],
/// }
/// ```
///
/// # Arguments
/// * `key` - A `&str` that will be used as the key in the resulting HashMap.<br>
/// * `value` - An arbitrary type that will be serialized into `serde_json::Value` and associated with the corresponding key.
///
/// The precise keys and values are dependent on your specific use case. In this example, "input" and "history" are keys,
/// and
#[macro_export]
macro_rules! prompt_args {
    ( $($key:expr => $value:expr),* $(,)? ) => {
        {
            #[allow(unused_mut)]
            let mut args = std::collections::HashMap::<String, serde_json::Value>::new();
            $(
                // Convert the value to serde_json::Value before inserting
                args.insert($key.to_string(), serde_json::json!($value));
            )*
            args
        }
    };
}

/// `template_fstring` is a utility macro that creates a new `PromptTemplate` with FString as the template format.
///
/// # Usage
/// The macro is called with a template string and a list of variables that exist in the template. For example:
/// ```rust,ignore
/// template_fstring!(
///     "Hello {name}",
///     "name"
/// )
/// ```
/// This returns a `PromptTemplate` object that contains the string "Hello {name}" as the template and ["name"] as the variables, with TemplateFormat set to FString.
#[macro_export]
macro_rules! template_fstring {
    ($template:expr, $($var:expr),* $(,)?) => {
        $crate::prompt::PromptTemplate::new(
            $template.to_string(),
            vec![$($var.to_string()),*],
            $crate::prompt::TemplateFormat::FString,
        )
    };
}

/// `template_jinja2` is a utility macro that creates a new `PromptTemplate` with Jinja2 as the template format.
///
/// # Usage
/// The macro is called with a template string and a list of variables that exist in the template. For example:
/// ```rust,ignore
/// template_jinja2!(
///     "Hello {{ name }}",
///     "name"
/// )
/// ```
/// This returns a `PromptTemplate` object that contains the string "Hello {{ name }}" as the template and ["name"] as the variables, with TemplateFormat set to Jinja2.
#[macro_export]
macro_rules! template_jinja2 {
    ($template:expr, $($var:expr),* $(,)?) => {
        $crate::prompt::PromptTemplate::new(
            $template.to_string(),
            vec![$($var.to_string()),*],
            $crate::prompt::TemplateFormat::Jinja2,
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
        println!("{:?}", result);
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
