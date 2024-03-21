use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use serde_json::json;

use crate::{
    agent::Agent,
    chain::Chain,
    fmt_message, fmt_placeholder, fmt_template, message_formatter,
    prompt::{HumanMessagePromptTemplate, MessageFormatterStruct, PromptArgs},
    schemas::{
        agent::{AgentAction, AgentEvent, AgentFinish, LogTools},
        messages::Message,
        FunctionCallResponse,
    },
    template_jinja2,
    tools::Tool,
};

pub struct OpenAiToolAgent {
    pub(crate) chain: Box<dyn Chain>,
    pub(crate) tools: Vec<Arc<dyn Tool>>,
}

impl OpenAiToolAgent {
    pub fn create_prompt(prefix: &str) -> Result<MessageFormatterStruct, Box<dyn Error>> {
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

    fn construct_scratchpad(
        &self,
        intermediate_steps: &[(AgentAction, String)],
    ) -> Result<Vec<Message>, Box<dyn Error>> {
        let mut thoughts: Vec<Message> = Vec::new();
        for (action, observation) in intermediate_steps.into_iter() {
            let tool_calls: LogTools = serde_json::from_str(&action.log)?;
            let tools: Vec<FunctionCallResponse> = serde_json::from_str(&tool_calls.tools)?;
            thoughts.push(Message::new_ai_message("").with_tool_calls(json!(tools)));
            thoughts.push(Message::new_tool_message(observation, tool_calls.tool_id));
        }
        Ok(thoughts)
    }
}

#[async_trait]
impl Agent for OpenAiToolAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, Box<dyn Error>> {
        let mut inputs = inputs.clone();
        let scratchpad = self.construct_scratchpad(&intermediate_steps)?;
        inputs.insert("agent_scratchpad".to_string(), json!(scratchpad));
        let output = self.chain.call(inputs).await?.generation;
        match serde_json::from_str::<Vec<FunctionCallResponse>>(&output) {
            Ok(tools) => {
                let tool_to_call = tools.first().ok_or("No tool to call")?;
                let log: LogTools = LogTools {
                    tool_id: tool_to_call.id.clone(),
                    tools: output.clone(),
                };
                return Ok(AgentEvent::Action(AgentAction {
                    tool: tool_to_call.function.name.clone(),
                    tool_input: tool_to_call.function.arguments.clone(),
                    log: serde_json::to_string(&log)?,
                }));
            }
            Err(_) => return Ok(AgentEvent::Finish(AgentFinish { output })),
        }
    }

    fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }
}
