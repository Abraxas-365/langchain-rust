use std::sync::Arc;
use std::{collections::HashMap, error::Error};

use async_trait::async_trait;
use indoc::indoc;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::prompt_template;
use crate::schemas::{
    InputVariables, Message, MessageOrTemplate, MessageTemplate, MessageType, PromptTemplate,
};
use crate::{
    agent::{Agent, AgentError},
    chain::Chain,
    schemas::{
        agent::{AgentAction, AgentEvent},
        FunctionCallResponse,
    },
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
    pub(crate) tools: HashMap<String, Arc<dyn Tool>>,
}

impl OpenAiToolAgent {
    pub fn create_prompt(prefix: &str) -> Result<PromptTemplate, AgentError> {
        let prompt = prompt_template![
            MessageOrTemplate::Message(Message::new(MessageType::SystemMessage, prefix)),
            MessageOrTemplate::Placeholder("chat_history".into()),
            MessageOrTemplate::Template(MessageTemplate::from_jinja2(
                MessageType::HumanMessage,
                "{{input}}"
            )),
            MessageOrTemplate::Placeholder("chat_history".into())
        ];

        Ok(prompt)
    }

    fn construct_scratchpad(&self, intermediate_steps: &[(Option<AgentAction>, String)]) -> String {
        intermediate_steps
            .iter()
            .map(|(action, result)| match action {
                Some(action) => format!(
                    indoc! {"
                        Action: {}
                        Action input: {}
                        Result:
                        {}
                    "},
                    &action.action, &action.action_input, result
                ),
                None => result.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[async_trait]
impl Agent for OpenAiToolAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(Option<AgentAction>, String)],
        inputs: &mut InputVariables,
    ) -> Result<AgentEvent, AgentError> {
        let scratchpad = self.construct_scratchpad(intermediate_steps);
        inputs.insert("agent_scratchpad".to_string(), scratchpad);
        let output = self.chain.call(inputs).await?.generation;
        match serde_json::from_str::<Vec<FunctionCallResponse>>(&output) {
            Ok(tools) => {
                let mut actions: Vec<AgentAction> = Vec::new();
                for tool in tools {
                    actions.push(AgentAction {
                        action: tool.function.name.clone(),
                        action_input: Value::String(tool.function.arguments),
                    });
                }
                return Ok(AgentEvent::Action(actions));
            }
            Err(_) => return Ok(AgentEvent::Finish(output)),
        }
    }

    fn get_tool(&self, tool_name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(tool_name).cloned()
    }

    fn log_messages(&self, inputs: &InputVariables) -> Result<(), Box<dyn Error>> {
        self.chain.log_messages(inputs)
    }
}
