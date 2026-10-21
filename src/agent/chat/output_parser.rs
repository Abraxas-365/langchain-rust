use std::collections::VecDeque;

use regex::Regex;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    agent::AgentError,
    schemas::agent::{AgentAction, AgentEvent, AgentFinish},
};

use super::prompt::FORMAT_INSTRUCTIONS;

#[derive(Debug, Deserialize)]
struct AgentOutput {
    action: String,
    /// Usually a string, but models often send an object for tools with several
    /// arguments; that is kept as its JSON text so `Tool::parse_input` can read it back.
    #[serde(deserialize_with = "string_or_json")]
    action_input: String,
}

fn string_or_json<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(match Value::deserialize(d)? {
        Value::String(s) => s,
        other => other.to_string(),
    })
}

pub struct ChatOutputParser {}
impl ChatOutputParser {
    pub fn new() -> Self {
        Self {}
    }
}

impl ChatOutputParser {
    pub fn parse(&self, text: &str) -> Result<AgentEvent, AgentError> {
        log::debug!("Parsing to Agent Action: {}", text);
        match parse_json_markdown(text) {
            Some(value) => {
                // Deserialize the Value into AgentOutput
                let agent_output: AgentOutput = serde_json::from_value(value)?;

                if agent_output.action == "Final Answer" {
                    Ok(AgentEvent::Finish(AgentFinish {
                        output: agent_output.action_input,
                    }))
                } else {
                    Ok(AgentEvent::Action(vec![AgentAction {
                        tool: agent_output.action,
                        tool_input: agent_output.action_input,
                        log: text.to_string(),
                    }]))
                }
            }
            None => {
                log::debug!("No JSON found or malformed JSON in text: {}", text);
                Ok(AgentEvent::Finish(AgentFinish {
                    output: text.to_string(),
                }))
            }
        }
    }

    pub fn get_format_instructions(&self) -> &str {
        FORMAT_INSTRUCTIONS
    }
}

fn parse_partial_json(s: &str, strict: bool) -> Option<Value> {
    // First, attempt to parse the string as-is.
    match serde_json::from_str::<Value>(s) {
        Ok(val) => return Some(val),
        Err(_) if !strict => (),
        Err(_) => return None,
    }

    let mut new_s = String::new();
    let mut stack: VecDeque<char> = VecDeque::new();
    let mut is_inside_string = false;
    let mut escaped = false;

    for char in s.chars() {
        match char {
            '"' if !escaped => is_inside_string = !is_inside_string,
            '{' if !is_inside_string => stack.push_back('}'),
            '[' if !is_inside_string => stack.push_back(']'),
            '}' | ']' if !is_inside_string => {
                if let Some(c) = stack.pop_back() {
                    if c != char {
                        return None; // Mismatched closing character
                    }
                } else {
                    return None; // Unbalanced closing character
                }
            }
            '\\' if is_inside_string => escaped = !escaped,
            _ => escaped = false,
        }
        new_s.push(char);
    }

    // Close any open structures.
    while let Some(c) = stack.pop_back() {
        new_s.push(c);
    }

    // Attempt to parse again.
    serde_json::from_str(&new_s).ok()
}

fn parse_json_markdown(json_markdown: &str) -> Option<Value> {
    let re = Regex::new(r"```(?:json)?\s*([\s\S]+?)\s*```").unwrap();
    if let Some(caps) = re.captures(json_markdown) {
        if let Some(json_str) = caps.get(1) {
            return parse_partial_json(json_str.as_str(), false);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_tool_action() {
        let text = "Thought\n```json\n{\"action\": \"Calculator\", \"action_input\": \"2+2\"}\n```";
        match ChatOutputParser::new().parse(text).unwrap() {
            AgentEvent::Action(actions) => {
                assert_eq!(actions.len(), 1);
                assert_eq!(actions[0].tool, "Calculator");
                assert_eq!(actions[0].tool_input, "2+2");
                assert_eq!(actions[0].log, text);
            }
            other => panic!("expected action, got {other:?}"),
        }
    }

    #[test]
    fn parses_final_answer_without_json_tag() {
        let text = "```\n{\"action\": \"Final Answer\", \"action_input\": \"4\"}\n```";
        match ChatOutputParser::new().parse(text).unwrap() {
            AgentEvent::Finish(f) => assert_eq!(f.output, "4"),
            other => panic!("expected finish, got {other:?}"),
        }
    }

    #[test]
    fn plain_text_is_final_answer() {
        match ChatOutputParser::new().parse("just text").unwrap() {
            AgentEvent::Finish(f) => assert_eq!(f.output, "just text"),
            other => panic!("expected finish, got {other:?}"),
        }
    }

    #[test]
    fn json_with_wrong_shape_is_an_error() {
        let text = "```json\n{\"foo\": 1}\n```";
        assert!(matches!(
            ChatOutputParser::new().parse(text),
            Err(AgentError::SerdeJsonError(_))
        ));
    }

    #[test]
    fn partial_json_is_completed() {
        assert_eq!(
            parse_partial_json(r#"{"a": [1, 2"#, false),
            Some(json!({"a": [1, 2]}))
        );
        assert_eq!(
            parse_partial_json(r#"{"a": "}{"}"#, false),
            Some(json!({"a": "}{"}))
        );
        assert_eq!(parse_partial_json(r#"{"a": 1]"#, false), None);
        assert_eq!(parse_partial_json(r#"{"a": 1"#, true), None);
        assert_eq!(parse_partial_json(r#"[1]"#, true), Some(json!([1])));
    }

    #[test]
    fn markdown_with_truncated_json_is_repaired() {
        let text = "```json\n{\"action\": \"Final Answer\", \"action_input\": \"ok\"\n```";
        match ChatOutputParser::new().parse(text).unwrap() {
            AgentEvent::Finish(f) => assert_eq!(f.output, "ok"),
            other => panic!("expected finish, got {other:?}"),
        }
    }

    #[test]
    fn object_action_input_is_passed_as_json() {
        // #293: tools with several arguments get an object from the model
        let text = r#"```json
{"action": "Weather", "action_input": {"city": "Lima", "days": 3}}
```"#;
        match ChatOutputParser::new().parse(text).unwrap() {
            AgentEvent::Action(actions) => {
                let input: Value = serde_json::from_str(&actions[0].tool_input).unwrap();
                assert_eq!(input, json!({"city": "Lima", "days": 3}));
            }
            other => panic!("expected action, got {other:?}"),
        }
    }

    #[test]
    fn format_instructions_mention_final_answer() {
        assert!(ChatOutputParser::new()
            .get_format_instructions()
            .contains("Final Answer"));
    }
}
