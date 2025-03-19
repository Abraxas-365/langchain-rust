use std::sync::Arc;
use std::{collections::HashMap, error::Error};

use async_openai::types::{ChatCompletionMessageToolCall, ChatCompletionToolType, FunctionCall};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    agent::{Agent, AgentError},
    chain::Chain,
    prompt_template,
    schemas::{
        agent::{AgentAction, AgentEvent},
        FunctionCallResponse, InputVariables, Message, MessageType,
    },
    template::{MessageOrTemplate, MessageTemplate, PromptTemplate},
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
            Message::new(MessageType::SystemMessage, prefix),
            MessageOrTemplate::Placeholder("chat_history".into()),
            MessageTemplate::from_jinja2(MessageType::HumanMessage, "{{input}}"),
            MessageOrTemplate::Placeholder("chat_history".into())
        ];

        Ok(prompt)
    }

    fn construct_scratchpad(&self, intermediate_steps: &[(AgentAction, String)]) -> Vec<Message> {
        intermediate_steps
            .iter()
            .flat_map(|(action, observation)| {
                vec![
                    Message::new(MessageType::AIMessage, "").with_tool_calls(vec![
                        ChatCompletionMessageToolCall {
                            id: action.id.clone(),
                            r#type: ChatCompletionToolType::Function,
                            function: FunctionCall {
                                name: action.action.clone(),
                                arguments: serde_json::to_string_pretty(&action.action_input)
                                    .unwrap_or("Input parameters unknown".into()),
                            },
                        },
                    ]),
                    Message::new_tool_message(Some(action.id.clone()), observation),
                ]
            })
            .collect::<Vec<_>>()
    }
}

#[async_trait]
impl Agent for OpenAiToolAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: &mut InputVariables,
    ) -> Result<AgentEvent, AgentError> {
        let scratchpad = self.construct_scratchpad(intermediate_steps);
        inputs.insert_placeholder_replacement("agent_scratchpad", scratchpad);
        let output: String = self.chain.call(inputs).await?.generation;
        match serde_json::from_str::<Vec<FunctionCallResponse>>(&output) {
            Ok(tools) => {
                let mut actions: Vec<AgentAction> = Vec::new();
                for tool in tools {
                    actions.push(AgentAction {
                        id: tool.id,
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
