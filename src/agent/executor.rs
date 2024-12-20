use std::pin::Pin;
use std::{collections::HashMap, sync::Arc};

use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use super::{agent::Agent, AgentError};
use crate::schemas::{LogTools, Message, StreamData};
use crate::{
    chain::{chain_trait::Chain, ChainError},
    language_models::GenerateResult,
    memory::SimpleMemory,
    prompt::PromptArgs,
    schemas::{
        agent::{AgentAction, AgentEvent},
        memory::BaseMemory,
    },
    tools::Tool,
};

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
            name_to_tool.insert(tool.name().trim().replace(" ", "_"), tool.clone());
        }
        name_to_tool
    }
}

#[async_trait]
impl<A> Chain for AgentExecutor<A>
where
    A: Agent + Send + Sync,
{
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
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
            let agent_event = self
                .agent
                .plan(&steps, input_variables.clone())
                .await
                .map_err(|e| ChainError::AgentError(format!("Error in agent planning: {}", e)))?;
            match agent_event {
                AgentEvent::Action(actions) => {
                    for action in actions {
                        log::debug!("Action: {:?}", action.tool_input);
                        let tool = name_to_tools
                            .get(&action.tool)
                            .ok_or_else(|| {
                                AgentError::ToolError(format!("Tool {} not found", action.tool))
                            })
                            .map_err(|e| ChainError::AgentError(e.to_string()))?;

                        let observation_result = tool.call(&action.tool_input).await;

                        let observation = match observation_result {
                            Ok(result) => result,
                            Err(err) => {
                                log::info!(
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

                        steps.push((action, observation));
                    }
                }
                AgentEvent::Finish(finish) => {
                    if let Some(memory) = &self.memory {
                        let mut memory = memory.lock().await;

                        memory.add_user_message(match &input_variables["input"] {
                            // This avoids adding extra quotes to the user input in the history.
                            serde_json::Value::String(s) => s,
                            x => x, // this the json encoded value.
                        });

                        let mut tools_ai_message_seen: HashMap<String, ()> = HashMap::default();
                        for (action, observation) in steps {
                            let LogTools { tool_id, tools } = serde_json::from_str(&action.log)?;
                            let tools_value: serde_json::Value = serde_json::from_str(&tools)?;
                            if tools_ai_message_seen.insert(tools, ()).is_none() {
                                memory.add_message(
                                    Message::new_ai_message("").with_tool_calls(tools_value),
                                );
                            }
                            memory.add_message(Message::new_tool_message(observation, tool_id));
                        }

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

    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, ChainError> {
        let result = self.call(input_variables).await?;
        Ok(result.generation)
    }

    async fn stream<'life>(
        &'life self,
        input_variables: PromptArgs,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send + 'life>>, ChainError>
    {
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

        // let my_agent = self.agent.clone();

        let main_stream = stream! {

            // pin_mut!(steps);
            // let input_variables = pin!(input_variables);
            // let name_to_tools = pin!(name_to_tools);

                loop {
                let agent_event = self
                    .agent
                    .plan(&steps, input_variables.clone())
                    .await
                    .map_err(|e| ChainError::AgentError(format!("Error in agent planning: {}", e)))?;
                match agent_event {
                    AgentEvent::Action(actions) => {
                        for action in actions {
                            log::debug!("Action: {:?}", action.tool_input);
                            let tool = name_to_tools
                                .get(&action.tool)
                                .ok_or_else(|| {
                                    AgentError::ToolError(format!("Tool {} not found", action.tool))
                                })
                                .map_err(|e| ChainError::AgentError(e.to_string()))?;

                            let observation_result = tool.call(&action.tool_input).await;

                            // let observation = match observation_result.map_err(|e| Box::into_pin(e)) {
                            //     Ok(result) => result,
                            //     Err(err) => {
                            //         // let err = pin!(err);
                            //         log::info!(
                            //             "The tool return the following error: {}",
                            //             err.to_string()
                            //         );
                            //         if self.break_if_error {
                            //             let err_string = err.to_string();
                            //             let intermed_err = AgentError::ToolError(err_string).to_string();
                            //             drop(err);
                            //             yield Err(ChainError::AgentError(
                            //                 intermed_err,
                            //             ));
                            //             // yield Err(ChainError::AgentError(
                            //             //     AgentError::ToolError(err.to_string()).to_string(),
                            //             // ));
                            //             return; // TODO: Don't always exit the loop on yield.
                            //         } else {
                            //             format!("The tool return the following error: {}", err)
                            //         }
                            //     }
                            // };

                            // steps.push((action, observation));

                            match observation_result.map_err(|e| Box::new(e.to_string())) {
                                Ok(result) => {
                                    let observation = result;
                                    steps.push((action, observation));
                                }
                                Err(err_str) => {
                                    log::info!(
                                        "The tool return the following error: {}",
                                        err_str
                                    );
                                    if self.break_if_error {
                                        let intermed_err = AgentError::ToolError(*err_str).to_string();
                                        yield Err(ChainError::AgentError(
                                            intermed_err,
                                        ));
                                        return; // TODO: Don't always exit the loop on yield.
                                    } else {
                                        let observation = format!("The tool return the following error: {}", err_str);
                                        steps.push((action, observation));
                                    }
                                }
                            }

                        }
                    }
                    AgentEvent::Finish(finish) => {
                        if let Some(memory) = &self.memory { //FIXME: This would be a problem if the lifetime of memory is not 'self_life
                            let mut memory = memory.lock().await;

                            memory.add_user_message(match &input_variables["input"] {
                                // This avoids adding extra quotes to the user input in the history.
                                serde_json::Value::String(s) => s,
                                x => x, // this the json encoded value.
                            });

                            let mut tools_ai_message_seen: HashMap<String, ()> = HashMap::default();
                            for (action, observation) in steps.clone() {
                                let LogTools { tool_id, tools } = serde_json::from_str(&action.log)?;
                                let tools_value: serde_json::Value = serde_json::from_str(&tools)?;
                                if tools_ai_message_seen.insert(tools, ()).is_none() {
                                    memory.add_message(
                                        Message::new_ai_message("").with_tool_calls(tools_value),
                                    );
                                }
                                memory.add_message(Message::new_tool_message(observation, tool_id));
                            }

                            memory.add_ai_message(&finish.output);
                        }

                        // So, we cannot access the memory here...

                        // yield Ok(GenerateResult {
                        //     generation: finish.output,
                        //     ..Default::default()
                        // });
                        yield Ok(StreamData {
                            value: Value::String(finish.output.clone()), //TODO: this might be a problem
                            content: finish.output.clone(),
                            tokens: None,
                        });
                        return;
                    }
                }

                if let Some(max_iterations) = self.max_iterations {
                    if steps.len() >= max_iterations as usize {
                        // yield Ok(GenerateResult {
                        //     generation: "Max iterations reached".to_string(),
                        //     ..Default::default()
                        // });
                        yield Ok(StreamData {
                            value: Value::String("Max iterations reached".to_string()), //TODO: this might be a problem
                            content: "Max iterations reached".to_string(),
                            tokens: None,
                        });
                        return;
                    }
                }
            }
        };

        return Ok(Box::pin(main_stream));
    }
}
