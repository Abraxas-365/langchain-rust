use std::{collections::HashMap, error::Error, sync::Arc};

use async_trait::async_trait;
use serde_json::json;
use tokio::sync::Mutex;

use crate::{
    chain::chain_trait::Chain,
    language_models::GenerateResult,
    memory::SimpleMemory,
    prompt::PromptArgs,
    schemas::{
        agent::{AgentAction, AgentEvent},
        memory::BaseMemory,
    },
    tools::Tool,
};

use super::agent::Agent;

pub struct AgentExecutor<A>
where
    A: Agent,
{
    agent: A,
    max_iterations: Option<i32>,
    break_if_error: bool,
    pub memory: Option<Arc<Mutex<dyn BaseMemory>>>,
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
            log::debug!("Loading Tool:{}", tool.name());
            name_to_tool.insert(tool.name(), tool.clone());
        }
        name_to_tool
    }
}

#[async_trait]
impl<A> Chain for AgentExecutor<A>
where
    A: Agent + Send + Sync,
{
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>> {
        let mut input_variables = input_variables.clone();
        let name_to_tools = self.get_name_to_tools();
        let mut steps: Vec<(AgentAction, String)> = Vec::new();
        log::debug!("steps: {:?}", steps);
        if let Some(memory) = &self.memory {
            let memory = memory.lock().await;
            input_variables.insert("chat_history".to_string(), json!(memory.messages()));
        } else {
            input_variables.insert(
                "chat_history".to_string(),
                json!(SimpleMemory::new().messages()),
            );
        }

        loop {
            let agent_event = self.agent.plan(&steps, input_variables.clone()).await?;
            match agent_event {
                AgentEvent::Action(actions) => {
                    for action in actions {
                        log::debug!("Action: {:?}", action.tool_input);
                        let tool = name_to_tools.get(&action.tool).ok_or("Tool not found")?; //TODO:Check
                                                                                             //what to do with the error

                        let observation_result = tool.call(&action.tool_input).await;

                        let observation = match observation_result {
                            Ok(result) => result,
                            Err(err) => {
                                log::info!(
                                    "The tool return the following error: {}",
                                    err.to_string()
                                );
                                if self.break_if_error {
                                    return Err(err); // return the error immediately
                                } else {
                                    format!(
                                        "The tool return the following error: {}",
                                        err.to_string()
                                    )
                                    // convert the error to a string and continue
                                }
                            }
                        };

                        steps.push((action, observation));
                    }
                }
                AgentEvent::Finish(finish) => {
                    if let Some(memory) = &self.memory {
                        let mut memory = memory.lock().await;
                        memory.add_user_message(&input_variables["input"]);
                        memory.add_ai_message(&finish.output);
                    }
                    return Ok(GenerateResult {
                        generation: finish.output,
                        ..Default::default()
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

    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>> {
        let result = self.call(input_variables).await?;
        Ok(result.generation)
    }
}
