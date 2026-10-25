use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::{
    agent::{agent::Agent, AgentError},
    chain::chain_trait::Chain,
    language_models::TokenUsage,
    prompt::PromptArgs,
    schemas::{
        agent::{AgentAction, AgentEvent},
        messages::Message,
    },
    tools::Tool,
};

use super::output_parser::ReActOutputParser;

/// Stop sequences that end a turn before the model invents the tool's answer.
pub const OBSERVATION_STOP_WORDS: [&str; 2] = ["\nObservation:", "\n\tObservation:"];

/// A ReAct (MRKL) agent: the model is told about the tools in the prompt and answers in
/// plain text with `Thought:` / `Action:` / `Action Input:`, or `Final Answer:`. It works
/// with any model, including ones without native tool calling (for those that have it,
/// [`OpenAiToolAgent`](crate::agent::OpenAiToolAgent) is more reliable).
///
/// Build it with [`ReActAgentBuilder`](super::ReActAgentBuilder) and run it with an
/// [`AgentExecutor`](crate::agent::AgentExecutor). Models sometimes break the format;
/// [`AgentExecutor::with_parser_error_handler`](crate::agent::AgentExecutor::with_parser_error_handler)
/// sends the problem back to the model instead of failing the run.
pub struct ReActAgent {
    pub(crate) chain: Box<dyn Chain>,
    pub(crate) tools: Vec<Arc<dyn Tool>>,
    pub(crate) output_parser: ReActOutputParser,
}

impl ReActAgent {
    /// The previous steps as the prompt shows them: each action's text, its observation,
    /// and a new `Thought:` for the model to continue from.
    pub fn construct_scratchpad(intermediate_steps: &[(AgentAction, String)]) -> String {
        intermediate_steps
            .iter()
            .map(|(action, observation)| {
                format!("{}\nObservation: {observation}\nThought: ", action.log)
            })
            .collect()
    }

    /// The conversation so far (from the executor's memory) as text, or empty.
    fn chat_history(inputs: &PromptArgs) -> Result<String, AgentError> {
        let messages: Vec<Message> = match inputs.get("chat_history") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::String(text)) => return Ok(text.clone()),
            Some(value) => serde_json::from_value(value.clone())?,
        };
        if messages.is_empty() {
            return Ok(String::new());
        }
        Ok(format!(
            "Previous conversation:\n{}\n\n",
            Message::messages_to_string(&messages)
        ))
    }
}

#[async_trait]
impl Agent for ReActAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, AgentError> {
        Ok(self.plan_with_usage(intermediate_steps, inputs).await?.0)
    }

    async fn plan_with_usage(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        mut inputs: PromptArgs,
    ) -> Result<(AgentEvent, Option<TokenUsage>), AgentError> {
        let history = Self::chat_history(&inputs)?;
        inputs.insert("chat_history".into(), json!(history));
        inputs.insert(
            "agent_scratchpad".into(),
            json!(Self::construct_scratchpad(intermediate_steps)),
        );
        let result = self.chain.call(inputs).await?;
        let event = self.output_parser.parse(&result.generation)?;
        Ok((event, result.tokens))
    }

    fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }
}
