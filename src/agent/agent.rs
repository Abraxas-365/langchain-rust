use std::{pin::Pin, sync::Arc};

use async_trait::async_trait;
use futures::Stream;

use crate::{
    language_models::TokenUsage,
    prompt::PromptArgs,
    schemas::agent::{AgentAction, AgentEvent, AgentFinish},
    tools::Tool,
};

use super::AgentError;

/// An event from [`Agent::plan_stream`].
#[derive(Debug)]
#[non_exhaustive]
pub enum PlanEvent {
    /// Text the model writes while planning, as it arrives.
    Text(String),
    /// Reasoning the model produces while planning, as it arrives.
    Thinking(String),
    /// Tokens used by the model call(s) behind this plan, when the LLM reports them.
    /// Comes before `Done`.
    Usage(TokenUsage),
    /// The plan: run tools or finish. Always the last event.
    Done(AgentEvent),
}

/// Stream returned by [`Agent::plan_stream`].
pub type PlanEventStream<'a> =
    Pin<Box<dyn Stream<Item = Result<PlanEvent, AgentError>> + Send + 'a>>;

/// Boxes a plan event stream; also pins down the error type for `try_stream!` blocks.
pub(crate) fn box_plan_events<'a, S>(stream: S) -> PlanEventStream<'a>
where
    S: Stream<Item = Result<PlanEvent, AgentError>> + Send + 'a,
{
    Box::pin(stream)
}

#[async_trait]
pub trait Agent: Send + Sync {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, AgentError>;

    fn get_tools(&self) -> Vec<Arc<dyn Tool>>;

    /// Like [`plan`](Agent::plan), and also returns the tokens the model call(s) used, so
    /// [`AgentExecutor`](crate::agent::AgentExecutor) can report a run's total.
    ///
    /// The default returns no usage. Agents built on an LLM chain override it.
    async fn plan_with_usage(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<(AgentEvent, Option<TokenUsage>), AgentError> {
        Ok((self.plan(intermediate_steps, inputs).await?, None))
    }

    /// Like [`plan`](Agent::plan), but streams the model's text while it is generated.
    ///
    /// The default implementation calls `plan_with_usage` and, when the agent finishes,
    /// emits the final answer as one `Text` chunk, then the usage (if any), then `Done`.
    /// Agents whose raw model output is readable text (like
    /// [`OpenAiToolAgent`](crate::agent::OpenAiToolAgent)) override it to stream token by
    /// token.
    fn plan_stream<'a>(
        &'a self,
        intermediate_steps: &'a [(AgentAction, String)],
        inputs: PromptArgs,
    ) -> PlanEventStream<'a> {
        box_plan_events(async_stream::try_stream! {
            let (event, usage) = self.plan_with_usage(intermediate_steps, inputs).await?;
            if let AgentEvent::Finish(AgentFinish { output }) = &event {
                if !output.is_empty() {
                    yield PlanEvent::Text(output.clone());
                }
            }
            if let Some(usage) = usage {
                yield PlanEvent::Usage(usage);
            }
            yield PlanEvent::Done(event);
        })
    }
}
