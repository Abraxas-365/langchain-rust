use std::collections::HashMap;
use std::error::Error;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use indoc::indoc;
use tokio::sync::Mutex;

use super::{agent::Agent, AgentError};
use crate::schemas::Message;
use crate::{
    chain::{chain_trait::Chain, ChainError},
    language_models::GenerateResult,
    prompt::PromptArgs,
    schemas::{
        agent::{AgentAction, AgentEvent},
        memory::BaseMemory,
    },
};

const FORCE_FINAL_ANSWER: &str = "Now it's time you MUST give your absolute best final answer. You'll ignore all previous instructions, stop using any tools, and just return your absolute BEST Final answer.";

pub struct AgentExecutor<A, T>
where
    A: Agent<T>,
    T: PromptArgs,
{
    agent: A,
    max_iterations: Option<usize>,
    break_if_error: bool,
    pub memory: Option<Arc<Mutex<dyn BaseMemory>>>,
    phantom: PhantomData<T>,
}

impl<A, T> AgentExecutor<A, T>
where
    A: Agent<T>,
    T: PromptArgs,
{
    pub fn from_agent(agent: A) -> Self {
        Self {
            agent,
            max_iterations: Some(10),
            break_if_error: false,
            memory: None,
            phantom: PhantomData,
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
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
}

#[async_trait]
impl<A, T> Chain<T> for AgentExecutor<A, T>
where
    A: Agent<T> + Send + Sync,
    T: PromptArgs,
{
    async fn call(&self, input_variables: &mut T) -> Result<GenerateResult, ChainError> {
        let mut steps: Vec<(Option<AgentAction>, String)> = Vec::new();
        let mut use_counts: HashMap<String, usize> = HashMap::new();
        if let Some(memory) = &self.memory {
            let memory: tokio::sync::MutexGuard<'_, dyn BaseMemory> = memory.lock().await;
            input_variables.insert("chat_history".to_string(), memory.to_string());
        // TODO: Possibly implement messages parsing
        } else {
            input_variables.insert("chat_history".to_string(), "".to_string());
        }

        {
            let mut input_variables_demo = input_variables.clone();
            input_variables_demo.insert("agent_scratchpad".to_string(), "".to_string());
            self.log_messages(&input_variables_demo).map_err(|e| {
                ChainError::AgentError(format!("Error formatting initial messages: {e}"))
            })?;
        }

        loop {
            let agent_event = self.agent.plan(&steps, input_variables).await;

            match agent_event {
                Ok(AgentEvent::Action(actions)) => {
                    for action in actions {
                        if let Some(max_iterations) = self.max_iterations {
                            if steps.len() >= max_iterations {
                                log::debug!(
                                    "Max iteration ({}) reached, forcing final answer",
                                    max_iterations
                                );
                                steps.push((None, FORCE_FINAL_ANSWER.to_string()));
                                continue;
                            }
                        }

                        log::debug!(
                            indoc! {"
                                Agent Action:
                                  action: {}
                                  input:
                                    {:#?}
                            "},
                            &action.action,
                            &action.action_input
                        );

                        let tool_name = action.action.to_lowercase().replace(" ", "_");
                        let tool = match self.agent.get_tool(&tool_name) {
                            Some(tool) => tool,
                            None => {
                                log::debug!("Tool {} not found", action.action);

                                steps.push((
                                    None,
                                    format!("{} is not a tool, You MUST use a tool OR give your best final answer.", action.action),
                                ));
                                continue;
                            }
                        };

                        if let Some(usage_limit) = tool.usage_limit() {
                            let count = use_counts.entry(tool_name.clone()).or_insert(0);
                            *count += 1;
                            if *count > usage_limit {
                                log::debug!(
                                    "Tool {} usage limit ({}) reached, preventing further use",
                                    action.action,
                                    usage_limit
                                );
                                steps.push((None, format!("You have used the tool {} too many times, you CANNOT and MUST NOT use it again", action.action)));
                                continue;
                            }
                        }

                        let observation_result = tool.call(action.action_input.clone()).await;

                        let observation = match observation_result {
                            Ok(result) => result,
                            Err(err) => {
                                if self.break_if_error {
                                    return Err(ChainError::AgentError(
                                        AgentError::ToolError(err.to_string()).to_string(),
                                    ));
                                } else {
                                    format!("The tool return the following error: {}", err)
                                }
                            }
                        };

                        log::debug!("Tool {} result:\n{}", &action.action, &observation);

                        steps.push((Some(action), observation));
                    }
                }
                Ok(AgentEvent::Finish(final_answer)) => {
                    if let Some(memory) = &self.memory {
                        let mut memory = memory.lock().await;

                        memory.add_user_message(match &input_variables.get("input") {
                            Some(input) => input,
                            None => &"",
                        });

                        for (action, observation) in steps {
                            if let Some(action) = action {
                                // TODO: change message type entirely
                                memory.add_ai_message(&action.action);
                                memory.add_message(Message::new_tool_message(
                                    observation,
                                    action.action,
                                ));
                            }
                        }

                        memory.add_ai_message(&final_answer);
                    }

                    log::debug!("Agent finished with result:\n{}", &final_answer);

                    return Ok(GenerateResult {
                        generation: final_answer,
                        ..GenerateResult::default()
                    });
                }
                Err(e) => {
                    log::debug!("Error: {}", e);
                    steps.push((None, e.to_string()));
                }
            }
        }
    }

    async fn invoke(&self, input_variables: &mut T) -> Result<String, ChainError> {
        let result = self.call(input_variables).await?;
        Ok(result.generation)
    }

    fn log_messages(&self, inputs: &T) -> Result<(), Box<dyn Error>> {
        self.agent.log_messages(inputs)
    }
}
