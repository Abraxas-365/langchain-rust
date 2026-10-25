//! The MRKL / ReAct prompt (as in LangChain's `ZeroShotAgent` and langchaingo's
//! `OneShotZeroAgent`).

use std::collections::HashMap;

use serde_json::Value;

use crate::{
    prompt::{FormatPrompter, PromptArgs, PromptError},
    schemas::{messages::Message, prompt::PromptValue},
};

/// Introduces the tools. `{{tool_descriptions}}` becomes one `name: description` line per
/// tool.
pub const PREFIX: &str =
    "Answer the following questions as best you can. You have access to the following tools:

{{tool_descriptions}}";

/// How to answer. `{{tool_names}}` becomes the comma-separated tool names.
pub const FORMAT_INSTRUCTIONS: &str = "Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question";

/// The question and the work so far. `{{chat_history}}` is empty, or the previous
/// conversation when the executor has memory; `{{agent_scratchpad}}` holds the previous
/// actions and observations of this run.
pub const SUFFIX: &str = "Begin!

{{chat_history}}Question: {{input}}
Thought:{{agent_scratchpad}}";

/// The ReAct prompt, as one human message. Every `{{name}}` is filled in a single pass
/// over the template, so a value that itself contains `{{...}}` (a question, a tool
/// description, a tool's output) is never substituted again. Unknown names are left as
/// written.
pub(super) struct ReActPrompt {
    template: String,
    /// Values known when the agent is built (the tools).
    fixed: HashMap<&'static str, String>,
    /// Variables each run must provide.
    variables: Vec<String>,
}

impl ReActPrompt {
    pub(super) fn new(template: String, fixed: HashMap<&'static str, String>) -> Self {
        let mut variables = vec!["input".to_string(), "agent_scratchpad".to_string()];
        if template.contains("{{chat_history}}") {
            variables.push("chat_history".into());
        }
        Self {
            template,
            fixed,
            variables,
        }
    }

    pub(super) fn render(&self, inputs: &PromptArgs) -> Result<String, PromptError> {
        if let Some(missing) = self.variables.iter().find(|v| !inputs.contains_key(*v)) {
            return Err(PromptError::MissingVariable(missing.clone()));
        }
        let mut out = String::with_capacity(self.template.len());
        let mut rest = self.template.as_str();
        while let Some(open) = rest.find("{{") {
            out.push_str(&rest[..open]);
            let after = &rest[open + 2..];
            let Some(close) = after.find("}}") else {
                out.push_str(&rest[open..]);
                return Ok(out);
            };
            let name = after[..close].trim();
            match self.fixed.get(name) {
                Some(value) => out.push_str(value),
                None => match inputs.get(name) {
                    Some(Value::String(value)) => out.push_str(value),
                    Some(value) => out.push_str(&value.to_string()),
                    None => out.push_str(&rest[open..open + 2 + close + 2]),
                },
            }
            rest = &after[close + 2..];
        }
        out.push_str(rest);
        Ok(out)
    }
}

impl FormatPrompter for ReActPrompt {
    fn format_prompt(&self, input_variables: PromptArgs) -> Result<PromptValue, PromptError> {
        let text = self.render(&input_variables)?;
        Ok(PromptValue::from_messages(vec![
            Message::new_human_message(text),
        ]))
    }

    fn get_input_variables(&self) -> Vec<String> {
        self.variables.clone()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::prompt_args;

    fn prompt(template: &str) -> ReActPrompt {
        ReActPrompt::new(
            template.into(),
            HashMap::from([("tool_names", "calc, {{input}}".to_string())]),
        )
    }

    #[test]
    fn renders_in_one_pass() {
        let p = prompt("[{{tool_names}}] Q: {{input}} S:{{ agent_scratchpad }}");
        let out = p
            .render(&prompt_args! {
                "input" => "say {{agent_scratchpad}} and {{tool_names}}",
                "agent_scratchpad" => "obs {{input}}",
            })
            .unwrap();
        assert_eq!(
            out,
            "[calc, {{input}}] Q: say {{agent_scratchpad}} and {{tool_names}} S:obs {{input}}"
        );
    }

    #[test]
    fn unknown_names_and_stray_braces_stay() {
        let p = prompt("{{input}} {{other}} {{ unclosed");
        let out = p
            .render(&prompt_args! {"input" => 7, "agent_scratchpad" => ""})
            .unwrap();
        assert_eq!(out, "7 {{other}} {{ unclosed");
    }

    #[test]
    fn requires_its_variables() {
        let p = prompt("{{chat_history}}{{input}}{{agent_scratchpad}}");
        assert_eq!(
            p.get_input_variables(),
            vec!["input", "agent_scratchpad", "chat_history"]
        );
        let err = p
            .render(&prompt_args! {"input" => "q", "agent_scratchpad" => ""})
            .unwrap_err();
        assert!(matches!(err, PromptError::MissingVariable(v) if v == "chat_history"));
        assert_eq!(
            prompt("{{input}}{{agent_scratchpad}}").get_input_variables(),
            vec!["input", "agent_scratchpad"]
        );
        let value = p
            .format_prompt(prompt_args! {
                "input" => "q", "agent_scratchpad" => "s", "chat_history" => json!("h ")
            })
            .unwrap();
        let messages = value.to_chat_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "h qs");
    }
}
