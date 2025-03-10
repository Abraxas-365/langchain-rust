use std::error::Error;
use std::marker::PhantomData;
use std::{collections::HashMap, sync::Arc};

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
    tools::Tool,
};

pub struct AgentExecutor<A, T>
where
    A: Agent<T>,
    T: PromptArgs,
{
    agent: A,
    max_iterations: Option<i32>,
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

    fn get_name_to_tools(&self) -> HashMap<String, Arc<dyn Tool>> {
        let mut name_to_tool = HashMap::new();
        for tool in self.agent.get_tools().iter() {
            name_to_tool.insert(tool.name().to_lowercase().replace(" ", ""), tool.clone());
        }
        name_to_tool
    }
}

#[async_trait]
impl<A, T> Chain<T> for AgentExecutor<A, T>
where
    A: Agent<T> + Send + Sync,
    T: PromptArgs,
{
    async fn call(&self, input_variables: &mut T) -> Result<GenerateResult, ChainError> {
        let name_to_tools = self.get_name_to_tools();
        let mut steps: Vec<(AgentAction, String)> = Vec::new();
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
            let agent_event = self
                .agent
                .plan(&steps, input_variables)
                .await
                .map_err(|e| ChainError::AgentError(format!("Error in agent planning: {}", e)))?;
            match agent_event {
                AgentEvent::Action(actions) => {
                    for action in actions {
                        log::debug!(
                            indoc! {"
                                Agent Action:
                                  thought: {}
                                  action: {}
                                  input:
                                    {:#?}
                            "},
                            action.thought.as_deref().unwrap_or("None"),
                            &action.action,
                            &action.action_input
                        );

                        let tool = name_to_tools
                            .get(&action.action.to_lowercase().replace(" ", ""))
                            .ok_or_else(|| {
                                AgentError::ToolError(format!("Tool {} not found", action.action))
                            })
                            .map_err(|e| ChainError::AgentError(e.to_string()))?;

                        let observation_result = tool.call(&action.action_input).await;

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

                        steps.push((action, observation));
                    }
                }
                AgentEvent::Finish(final_answer) => {
                    if let Some(memory) = &self.memory {
                        let mut memory = memory.lock().await;

                        memory.add_user_message(match &input_variables.get("input") {
                            Some(input) => input,
                            None => &"",
                        });

                        for (action, observation) in steps {
                            // TODO: change message type entirely
                            memory.add_ai_message(&action.action);
                            memory
                                .add_message(Message::new_tool_message(observation, action.action));
                        }

                        memory.add_ai_message(&final_answer);
                    }

                    log::debug!("Agent finished with result:\n{}", &final_answer);

                    return Ok(GenerateResult {
                        generation: final_answer,
                        ..GenerateResult::default()
                    });
                }
            }

            if let Some(max_iterations) = self.max_iterations {
                if steps.len() >= max_iterations as usize {
                    return Ok(GenerateResult {
                        generation: "Max iterations reached".to_string(),
                        ..Default::default()
                    });
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
