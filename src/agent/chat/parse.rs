use std::collections::VecDeque;

use regex::Regex;
use serde_json::Value;

use crate::{agent::AgentError, schemas::agent::AgentEvent};

pub fn parse_agent_output(text: &str) -> Result<AgentEvent, AgentError> {
    let agent_event = parse_json_markdown(text)
        .or_else(|| parse_partial_json(text, false))
        .ok_or(AgentError::InvalidFormatError)?;
    let agent_event: AgentEvent =
        serde_json::from_value(agent_event).map_err(|_| AgentError::InvalidFormatError)?;
    Ok(agent_event)
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
    use indoc::indoc;

    use super::*;

    #[test]
    fn test_parse_agent_output() {
        let test_output = indoc! {r#"
            ```json
            {
                "thought": "I'm thinking...",
                "action": "generate",
                "action_input": "Hello, world!"
            }
            ```
        "#};

        let parsed_output = parse_agent_output(test_output).unwrap();

        match parsed_output {
            AgentEvent::Action(agent_actions) => {
                assert!(agent_actions.len() == 1);
                let agent_action = &agent_actions[0];
                assert_eq!(agent_action.thought, Some("I'm thinking...".to_string()));
                assert_eq!(agent_action.action, "generate");
                assert_eq!(agent_action.action_input, "Hello, world!");
            }
            AgentEvent::Finish(_) => panic!("Expected AgentEvent::Action, got AgentEvent::Finish"),
        }

        let test_final_answer = indoc! {r#"
            ```json
            {
                "thought": "I now can give a great answer",
                "final_answer": "Goodbye, world!"
            }
            ```
        "#};

        let parsed_output = parse_agent_output(test_final_answer).unwrap();

        match parsed_output {
            AgentEvent::Action(_) => panic!("Expected AgentEvent::Finish, got AgentEvent::Action"),
            AgentEvent::Finish(final_answer) => {
                assert_eq!(final_answer, "Goodbye, world!");
            }
        }
    }
}
