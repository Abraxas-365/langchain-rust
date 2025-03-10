use std::sync::Arc;
use std::{collections::HashMap, error::Error};

use async_trait::async_trait;
use indoc::indoc;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    agent::{Agent, AgentError},
    chain::Chain,
    fmt_message, fmt_placeholder, fmt_template, message_formatter,
    prompt::{HumanMessagePromptTemplate, MessageFormatterStruct, PlainPromptArgs, PromptArgs},
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
    pub(crate) chain: Box<dyn Chain<PlainPromptArgs>>,
    pub(crate) tools: HashMap<String, Arc<dyn Tool>>,
}

impl OpenAiToolAgent {
    pub fn create_prompt(
        prefix: &str,
    ) -> Result<MessageFormatterStruct<PlainPromptArgs>, AgentError> {
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

    fn construct_scratchpad(&self, intermediate_steps: &[(Option<AgentAction>, String)]) -> String {
        intermediate_steps
            .iter()
            .map(|(action, result)| match action {
                Some(action) => format!(
                    indoc! {"
                        Thought: {}
                        Action: {}
                        Action input: {}
                        Result:
                        {}
                    "},
                    action.thought.as_deref().unwrap_or("None"),
                    &action.action,
                    &action.action_input,
                    result
                ),
                None => result.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[async_trait]
impl Agent<PlainPromptArgs> for OpenAiToolAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(Option<AgentAction>, String)],
        inputs: &mut PlainPromptArgs,
    ) -> Result<AgentEvent, AgentError> {
        let scratchpad = self.construct_scratchpad(intermediate_steps);
        inputs.insert("agent_scratchpad".to_string(), scratchpad);
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

    fn get_tool(&self, tool_name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(tool_name).cloned()
    }

    fn log_messages(&self, inputs: &PlainPromptArgs) -> Result<(), Box<dyn Error>> {
        self.chain.log_messages(inputs)
    }
}
