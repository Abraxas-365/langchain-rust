use std::{pin::Pin, sync::Arc};

use async_trait::async_trait;
use futures::Stream;

use crate::{
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

    /// Like [`plan`](Agent::plan), but streams the model's text while it is generated.
    ///
    /// The default implementation calls `plan` and, when the agent finishes, emits the
    /// final answer as one `Text` chunk before `Done`. Agents whose raw model output is
    /// readable text (like [`OpenAiToolAgent`](crate::agent::OpenAiToolAgent)) override it
    /// to stream token by token.
    fn plan_stream<'a>(
        &'a self,
        intermediate_steps: &'a [(AgentAction, String)],
        inputs: PromptArgs,
    ) -> PlanEventStream<'a> {
        box_plan_events(async_stream::try_stream! {
            let event = self.plan(intermediate_steps, inputs).await?;
            if let AgentEvent::Finish(AgentFinish { output }) = &event {
                if !output.is_empty() {
                    yield PlanEvent::Text(output.clone());
                }
            }
            yield PlanEvent::Done(event);
        })
    }
}
