use std::{collections::VecDeque, error::Error};

use regex::Regex;
use serde::Deserialize;
use serde_json::Value;

use crate::{
    agent::agent::AgentOutputParser,
    schemas::agent::{AgentAction, AgentEvent, AgentFinish},
};

use super::prompt::FORMAT_INSTRUCTIONS;

#[derive(Debug, Deserialize)]
struct AgentOutput {
    action: String,
    action_input: String,
}

pub struct ChatOutputParser {}
impl ChatOutputParser {
    pub fn new() -> Self {
        Self {}
    }
}

impl Into<Box<dyn AgentOutputParser>> for ChatOutputParser {
    fn into(self) -> Box<dyn AgentOutputParser> {
        Box::new(self)
    }
}
impl AgentOutputParser for ChatOutputParser {
    fn parse(&self, text: &str) -> Result<AgentEvent, Box<dyn std::error::Error>> {
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
                    Ok(AgentEvent::Action(AgentAction {
                        tool: agent_output.action,
                        tool_input: agent_output.action_input,
                        log: text.to_string(),
                    }))
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

    fn get_format_instructions(&self) -> &str {
        FORMAT_INSTRUCTIONS
    }
}

fn parse_partial_json(s: &str, strict: bool) -> Option<Value> {
    // First, attempt to parse the string as-is.
    match serde_json::from_str::<Value>(s) {
        Ok(val) => return Some(val),
        Err(_) if !strict => (),
        Err(e) => return None,
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
