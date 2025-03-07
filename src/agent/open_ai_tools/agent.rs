use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use indoc::indoc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    agent::{Agent, AgentError},
    chain::Chain,
    fmt_message, fmt_placeholder, fmt_template, message_formatter,
    prompt::{HumanMessagePromptTemplate, MessageFormatterStruct, PromptArgs},
    schemas::{
        agent::{AgentAction, AgentEvent},
        messages::Message,
        FunctionCallResponse,
    },
    template_jinja2,
    tools::Tool,
};

///Log tools is a struct used by the openai-like agents
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LogTools {
    pub tool_id: String,
    pub tools: String,
}

pub struct OpenAiToolAgent {
    pub(crate) chain: Box<dyn Chain>,
    pub(crate) tools: Vec<Arc<dyn Tool>>,
}

impl OpenAiToolAgent {
    pub fn create_prompt(prefix: &str) -> Result<MessageFormatterStruct, AgentError> {
        let prompt = message_formatter![
            fmt_message!(Message::new_system_message(prefix)),
            fmt_placeholder!("chat_history"),
            fmt_template!(HumanMessagePromptTemplate::new(template_jinja2!(
                "{{input}}",
                "input"
            ))),
            fmt_placeholder!("agent_scratchpad")
        ];

        Ok(prompt)
    }

    fn construct_scratchpad(&self, intermediate_steps: &[(AgentAction, String)]) -> String {
        intermediate_steps
            .iter()
            .map(|(action, result)| {
                if let Some(thought) = &action.thought {
                    format!(
                        indoc! {"
                            Thought: {}
                            Action: {}
                            Action input: {}
                            Result:
                            {}
                        "},
                        thought, action.action, action.action_input, result
                    )
                } else {
                    format!(
                        indoc! {"
                            Action: {}
                            Action input: {}
                            Result:
                            {}
                        "},
                        action.action, action.action_input, result
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[async_trait]
impl Agent for OpenAiToolAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, AgentError> {
        let mut inputs = inputs.clone();
        let scratchpad = self.construct_scratchpad(intermediate_steps);
        inputs.insert("agent_scratchpad".to_string(), json!(scratchpad));
        let output = self.chain.call(inputs).await?.generation;
        match serde_json::from_str::<Vec<FunctionCallResponse>>(&output) {
            Ok(tools) => {
                let mut actions: Vec<AgentAction> = Vec::new();
                for tool in tools {
                    actions.push(AgentAction {
                        thought: None,
                        action: tool.function.name.clone(),
                        action_input: Value::String(tool.function.arguments),
                    });
                }
                return Ok(AgentEvent::Action(actions));
            }
            Err(_) => return Ok(AgentEvent::Finish(output)),
        }
    }

    fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    fn log_messages(&self, inputs: PromptArgs) -> Result<(), Box<dyn Error>> {
        self.chain.log_messages(inputs)
    }
}
