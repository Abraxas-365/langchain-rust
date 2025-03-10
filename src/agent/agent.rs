use std::{error::Error, sync::Arc};

use async_trait::async_trait;

use crate::{
    prompt::PromptArgs,
    schemas::agent::{AgentAction, AgentEvent},
    tools::Tool,
};

use super::AgentError;

#[async_trait]
pub trait Agent<T>: Send + Sync
where
    T: PromptArgs,
{
    async fn plan(
        &self,
        intermediate_steps: &[(Option<AgentAction>, String)],
        inputs: &mut T,
    ) -> Result<AgentEvent, AgentError>;

    fn get_tool(&self, tool_name: &str) -> Option<Arc<dyn Tool>>;

    fn log_messages(&self, inputs: &T) -> Result<(), Box<dyn Error>>;
}
