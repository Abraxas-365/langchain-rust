use std::{error::Error, sync::Arc};

use async_trait::async_trait;

use crate::{
    prompt::PromptArgs,
    schemas::agent::{AgentAction, AgentEvent},
    tools::Tool,
};

#[async_trait]
pub trait Agent: Send + Sync {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, Box<dyn Error>>;

    fn get_tools(&self) -> Vec<Arc<dyn Tool>>;
}

pub trait AgentOutputParser: Send + Sync {
    fn parse(&self, text: &str) -> Result<AgentEvent, Box<dyn Error>>;
    fn get_format_instructions(&self) -> &str;
}
