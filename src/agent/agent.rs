use std::{error::Error, sync::Arc};

use async_trait::async_trait;

use crate::{
    prompt::PromptArgs,
    schemas::agent::{AgentAction, AgentEvent},
    tools::Tool,
};

use super::AgentError;

#[async_trait]
pub trait Agent: Send + Sync {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, AgentError>;

    fn get_tools(&self) -> Vec<Arc<dyn Tool>>;

    fn log_messages(&self, inputs: PromptArgs) -> Result<(), Box<dyn Error>>;
}
