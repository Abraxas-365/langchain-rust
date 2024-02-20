use std::error::Error;

use regex::Regex;
use serde::Deserialize;

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
    fn parse(&self, text: &str) -> Result<AgentEvent, Box<dyn Error>> {
        let sanitized_text = text
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect::<String>();

        log::debug!("Parsing to Agent Action: {}", sanitized_text);
        let re = Regex::new(r"```json?\s*(.*?)\s*```").unwrap();
        let json_match = re.captures(&sanitized_text).and_then(|cap| cap.get(1));
        log::debug!("Finish extracting json");
        let agent_output: AgentOutput = match json_match {
            Some(json_str) => serde_json::from_str(&json_str.as_str())?,
            None => {
                log::debug!("No JSON found in text: {}", sanitized_text);
                return Ok(AgentEvent::Finish(AgentFinish {
                    output: sanitized_text,
                }));
            }
        };

        if &agent_output.action == "Final Answer" {
            Ok(AgentEvent::Finish(AgentFinish {
                output: agent_output.action_input,
            }))
        } else {
            Ok(AgentEvent::Action(AgentAction {
                tool: agent_output.action,
                tool_input: agent_output.action_input,
                log: sanitized_text,
            }))
        }
    }

    fn get_format_instructions(&self) -> &str {
        FORMAT_INSTRUCTIONS
    }
}
