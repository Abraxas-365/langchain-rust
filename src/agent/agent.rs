use std::{error::Error, sync::Arc};

use async_trait::async_trait;

use crate::{
    prompt::{PromptArgs, PromptFromatter},
    prompt_args,
    schemas::{
        agent::{AgentAction, AgentEvent},
        messages::Message,
    },
    template_jinja2,
    tools::tool::Tool,
};

use super::chat::prompt::TEMPLATE_TOOL_RESPONSE;

#[async_trait]
pub trait Agent: Send + Sync {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, Box<dyn Error>>;

    fn get_tools(&self) -> Vec<Arc<dyn Tool>>;

    fn construct_scratchpad(
        &self,
        intermediate_steps: &[(AgentAction, String)],
    ) -> Result<Vec<Message>, Box<dyn Error>> {
        let mut thoughts: Vec<Message> = Vec::new();
        for (action, observation) in intermediate_steps.into_iter() {
            thoughts.push(Message::new_ai_message(&action.log));
            let tool_response = template_jinja2!(TEMPLATE_TOOL_RESPONSE, "observation")
                .format(prompt_args!("observation"=>observation))?;
            thoughts.push(Message::new_human_message(&tool_response));
        }
        Ok(thoughts)
    }
}

pub trait AgentOutputParser: Send + Sync {
    fn parse(&self, text: &str) -> Result<AgentEvent, Box<dyn Error>>;
    fn get_format_instructions(&self) -> &str;
}
