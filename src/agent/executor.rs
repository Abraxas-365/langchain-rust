use std::{collections::HashMap, pin::Pin, sync::Arc};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::json;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::{
    agent::{Agent, PlanEvent},
    AgentError,
};
use crate::observability::traced;
use crate::schemas::{LogTools, Message};
use crate::{
    chain::{chain_trait::Chain, ChainError},
    language_models::GenerateResult,
    prompt::PromptArgs,
    schemas::{
        agent::{AgentAction, AgentEvent},
        memory::BaseMemory,
    },
    tools::Tool,
};
use tracing::Instrument;

/// What an agent run is doing, streamed by [`AgentExecutor::stream_events`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AgentStreamEvent {
    /// A chunk of text from the model, as it is generated. With tool-calling agents this
    /// is the answer itself (or a short preamble before tool calls).
    Text(String),
    /// The agent is about to run a tool.
    ToolStart { tool: String, input: String },
    /// A tool finished. `output` is what the agent sees; a failed tool shows up here as
    /// its error message unless the executor uses `with_break_if_error(true)`.
    ToolEnd { tool: String, output: String },
    /// The run is over. Always the last event of a successful run.
    Finish { output: String },
}

/// Stream returned by [`AgentExecutor::stream_events`].
pub type AgentEventStream<'a> =
    Pin<Box<dyn Stream<Item = Result<AgentStreamEvent, ChainError>> + Send + 'a>>;

/// Output of a run that hit `max_iterations`.
const MAX_ITERATIONS_OUTPUT: &str = "Max iterations reached";
/// Error message of a run stopped through `with_cancellation`.
const CANCELLED: &str = "agent run cancelled";

pub struct AgentExecutor<A>
where
    A: Agent,
{
    agent: A,
    max_iterations: Option<i32>,
    break_if_error: bool,
    pub memory: Option<Arc<Mutex<dyn BaseMemory>>>,
    cancellation: Option<CancellationToken>,
}

impl<A> AgentExecutor<A>
where
    A: Agent,
{
    pub fn from_agent(agent: A) -> Self {
        Self {
            agent,
            max_iterations: Some(10),
            break_if_error: false,
            memory: None,
            cancellation: None,
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: i32) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    pub fn with_memory(mut self, memory: Arc<Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_break_if_error(mut self, break_if_error: bool) -> Self {
        self.break_if_error = break_if_error;
        self
    }

    /// Stops runs when `token` is cancelled: the in-flight LLM or tool call is dropped and
    /// the run returns a `ChainError::AgentError` ("agent run cancelled"); check
    /// `token.is_cancelled()` to tell it apart from other failures. Nothing is written to
    /// memory for a cancelled run.
    ///
    /// Applies to every run of this executor. Dropping the future (or the stream from
    /// [`stream_events`](Self::stream_events)) also cancels a run.
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = Some(token);
        self
    }

    fn get_name_to_tools(&self) -> HashMap<String, Arc<dyn Tool>> {
        let mut name_to_tool = HashMap::new();
        for tool in self.agent.get_tools().iter() {
            tracing::debug!("Loading Tool:{}", tool.name());
            name_to_tool.insert(tool.name().trim().replace(" ", "_"), tool.clone());
        }
        name_to_tool
    }
}

impl<A> AgentExecutor<A>
where
    A: Agent + Send + Sync,
{
    /// Runs the agent and streams what it does: model text as it is generated, each tool
    /// call and its result, and finally the answer.
    ///
    /// Memory is updated the same way as [`Chain::call`], once the run finishes. Stop early
    /// by dropping the stream or with [`with_cancellation`](Self::with_cancellation).
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    /// use langchain_rust::agent::AgentStreamEvent;
    ///
    /// let mut events = executor.stream_events(prompt_args! { "input" => "weather in Lima?" });
    /// while let Some(event) = events.next().await {
    ///     match event? {
    ///         AgentStreamEvent::Text(chunk) => print!("{chunk}"),
    ///         AgentStreamEvent::ToolStart { tool, input } => println!("\n-> {tool}({input})"),
    ///         AgentStreamEvent::ToolEnd { tool, output } => println!("<- {tool}: {output}"),
    ///         AgentStreamEvent::Finish { .. } => println!(),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn stream_events(&self, input_variables: PromptArgs) -> AgentEventStream<'_> {
        let run = async_stream::stream! {
            // Capacity 1: the run is at most one event ahead of the consumer, so a slow
            // consumer slows the run down and dropping the stream stops it.
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            let run = self.run(input_variables, Emitter(Some(tx)));
            tokio::pin!(run);
            let result = loop {
                tokio::select! {
                    biased;
                    Some(event) = rx.recv() => yield Ok(event),
                    result = &mut run => break result,
                }
            };
            while let Ok(event) = rx.try_recv() {
                yield Ok(event);
            }
            match result {
                Ok(output) => yield Ok(AgentStreamEvent::Finish { output }),
                Err(e) => yield Err(e),
            }
        };
        Box::pin(run)
    }

    /// The agent loop shared by `call` and `stream_events`. Returns the final output.
    async fn run(
        &self,
        input_variables: PromptArgs,
        emitter: Emitter,
    ) -> Result<String, ChainError> {
        let span = tracing::info_span!(
            "agent.run",
            max_iterations = self.max_iterations,
            steps = tracing::field::Empty,
            error = tracing::field::Empty,
        );
        let run = self.run_inner(input_variables, emitter);
        let run = async {
            match &self.cancellation {
                Some(token) => token
                    .run_until_cancelled(run)
                    .await
                    .unwrap_or_else(|| Err(ChainError::AgentError(CANCELLED.into()))),
                None => run.await,
            }
        };
        traced(span, run, |_, _| {}).await
    }

    async fn run_inner(
        &self,
        input_variables: PromptArgs,
        emitter: Emitter,
    ) -> Result<String, ChainError> {
        let mut input_variables = input_variables.clone();
        let name_to_tools = self.get_name_to_tools();
        let mut steps: Vec<(AgentAction, String)> = Vec::new();
        if let Some(memory) = &self.memory {
            let messages = memory.lock().await.messages().await?;
            input_variables.insert("chat_history".to_string(), json!(messages));
        } else {
            input_variables.insert("chat_history".to_string(), json!(Vec::<Message>::new()));
        }

        loop {
            let plan_span = tracing::info_span!("agent.plan", step = steps.len());
            let agent_event = async {
                if emitter.is_streaming() {
                    self.plan_streaming(&steps, input_variables.clone(), &emitter)
                        .await
                } else {
                    self.agent.plan(&steps, input_variables.clone()).await
                }
            }
            .instrument(plan_span)
            .await
            .map_err(|e| ChainError::AgentError(format!("Error in agent planning: {}", e)))?;

            match agent_event {
                AgentEvent::Action(actions) => {
                    for action in actions {
                        tracing::debug!("Action: {:?}", action.tool_input);
                        let tool = name_to_tools
                            .get(&action.tool.trim().replace(" ", "_"))
                            .ok_or_else(|| {
                                AgentError::ToolError(format!("Tool {} not found", action.tool))
                            })
                            .map_err(|e| ChainError::AgentError(e.to_string()))?;

                        emitter
                            .emit(AgentStreamEvent::ToolStart {
                                tool: action.tool.clone(),
                                input: action.tool_input.clone(),
                            })
                            .await;
                        let tool_span = tracing::info_span!(
                            "tool.call",
                            tool = %action.tool,
                            error = tracing::field::Empty,
                        );
                        let observation =
                            match traced(tool_span, tool.call(&action.tool_input), |_, _| {}).await
                            {
                                Ok(result) => result,
                                Err(err) => {
                                    tracing::info!(
                                        "The tool return the following error: {}",
                                        err.to_string()
                                    );
                                    if self.break_if_error {
                                        return Err(ChainError::AgentError(
                                            AgentError::ToolError(err.to_string()).to_string(),
                                        ));
                                    } else {
                                        format!("The tool return the following error: {}", err)
                                    }
                                }
                            };
                        emitter
                            .emit(AgentStreamEvent::ToolEnd {
                                tool: action.tool.clone(),
                                output: observation.clone(),
                            })
                            .await;

                        steps.push((action, observation));
                    }
                }
                AgentEvent::Finish(finish) => {
                    tracing::Span::current().record("steps", steps.len());
                    if let Some(memory) = &self.memory {
                        let mut memory = memory.lock().await;

                        memory
                            .add_user_message(match &input_variables["input"] {
                                // This avoids adding extra quotes to the user input in the history.
                                serde_json::Value::String(s) => s,
                                x => x, // this the json encoded value.
                            })
                            .await?;

                        let mut tools_ai_message_seen: HashMap<String, ()> = HashMap::default();
                        for (action, observation) in steps {
                            // Only OpenAI-tools style agents encode `LogTools` in the log;
                            // other agents (e.g. ConversationalAgent) store raw LLM text.
                            let Ok(LogTools { tool_id, tools }) =
                                serde_json::from_str::<LogTools>(&action.log)
                            else {
                                continue;
                            };
                            let tools_value: serde_json::Value = serde_json::from_str(&tools)?;
                            if tools_ai_message_seen.insert(tools, ()).is_none() {
                                memory
                                    .add_message(
                                        Message::new_ai_message("").with_tool_calls(tools_value),
                                    )
                                    .await?;
                            }
                            memory
                                .add_message(Message::new_tool_message(observation, tool_id))
                                .await?;
                        }

                        memory.add_ai_message(&finish.output).await?;
                    }
                    return Ok(finish.output);
                }
            }

            if let Some(max_iterations) = self.max_iterations {
                if steps.len() >= max_iterations as usize {
                    tracing::Span::current().record("steps", steps.len());
                    tracing::warn!(max_iterations, "agent stopped: max iterations reached");
                    return Ok(MAX_ITERATIONS_OUTPUT.to_string());
                }
            }
        }
    }

    /// Runs `plan_stream`, forwarding its text chunks, and returns the plan.
    async fn plan_streaming(
        &self,
        steps: &[(AgentAction, String)],
        inputs: PromptArgs,
        emitter: &Emitter,
    ) -> Result<AgentEvent, AgentError> {
        let mut plan = self.agent.plan_stream(steps, inputs);
        while let Some(event) = plan.next().await {
            match event? {
                PlanEvent::Text(text) => emitter.emit(AgentStreamEvent::Text(text)).await,
                PlanEvent::Done(event) => return Ok(event),
            }
        }
        Err(AgentError::OtherError(
            "agent plan stream ended without a plan".into(),
        ))
    }
}

/// Where the agent loop sends its events: nowhere for `call`, a channel for `stream_events`.
struct Emitter(Option<tokio::sync::mpsc::Sender<AgentStreamEvent>>);

impl Emitter {
    fn is_streaming(&self) -> bool {
        self.0.is_some()
    }

    async fn emit(&self, event: AgentStreamEvent) {
        if let Some(tx) = &self.0 {
            // Fails only when the receiver is gone, i.e. the run is being dropped.
            let _ = tx.send(event).await;
        }
    }
}

#[async_trait]
impl<A> Chain for AgentExecutor<A>
where
    A: Agent + Send + Sync,
{
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let output = self.run(input_variables, Emitter(None)).await?;
        Ok(GenerateResult {
            generation: output,
            ..Default::default()
        })
    }

    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, ChainError> {
        let result = self.call(input_variables).await?;
        Ok(result.generation)
    }
}
