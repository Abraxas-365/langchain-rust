use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

use async_openai::types::{ChatCompletionMessageToolCall, ChatCompletionToolType, FunctionCall};
use async_trait::async_trait;
use indoc::indoc;
use tokio::sync::Mutex;

use super::{agent::Agent, AgentError};
use crate::schemas::{InputVariables, Message, MessageType};
use crate::{
    chain::{chain_trait::Chain, ChainError},
    language_models::GenerateResult,
    schemas::{
        agent::{AgentAction, AgentEvent},
        memory::BaseMemory,
    },
};

const FORCE_FINAL_ANSWER: &str = "Now it's time you MUST give your absolute best final answer. You'll ignore all previous instructions, stop using any tools, and just return your absolute BEST Final answer.";

pub struct AgentExecutor<A>
where
    A: Agent,
{
    agent: A,
    max_iterations: Option<usize>,
    max_consecutive_fails: Option<usize>,
    break_if_tool_error: bool,
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
            max_consecutive_fails: Some(3),
            break_if_tool_error: false,
            memory: None,
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

    pub fn with_break_if_tool_error(mut self, break_if_tool_error: bool) -> Self {
        self.break_if_tool_error = break_if_tool_error;
        self
    }
}

#[async_trait]
impl<A> Chain for AgentExecutor<A>
where
    A: Agent + Send + Sync,
{
    async fn call(
        &self,
        input_variables: &mut InputVariables,
    ) -> Result<GenerateResult, ChainError> {
        let mut steps: Vec<(AgentAction, String)> = Vec::new();
        let mut use_counts: HashMap<String, usize> = HashMap::new();
        let mut consecutive_fails: usize = 0;

        if let Some(memory) = &self.memory {
            let memory: tokio::sync::MutexGuard<'_, dyn BaseMemory> = memory.lock().await;
            input_variables.insert_placeholder_replacement("chat_history", memory.messages());
        // TODO: Possibly implement messages parsing
        } else {
            input_variables.insert_placeholder_replacement("chat_history", vec![]);
        }

        {
            let mut input_variables_demo = input_variables.clone();
            input_variables_demo.insert_placeholder_replacement("agent_scratchpad", vec![]);
            self.log_messages(&input_variables_demo).map_err(|e| {
                ChainError::AgentError(format!("Error formatting initial messages: {e}"))
            })?;
        }

        'step: loop {
            if self
                .max_consecutive_fails
                .is_some_and(|max_consecutive_fails| consecutive_fails >= max_consecutive_fails)
            {
                log::error!(
                    "Too many consecutive fails ({} in a row), aborting",
                    consecutive_fails
                );
                return Err(ChainError::AgentError("Too many consecutive fails".into()));
            }

            let agent_event = self.agent.plan(&steps, input_variables).await;

            match agent_event {
                Ok(AgentEvent::Action(actions)) => {
                    for action in actions {
                        if self
                            .max_iterations
                            .is_some_and(|max_iterations| steps.len() >= max_iterations)
                        {
                            log::warn!(
                                "Max iteration ({}) reached, forcing final answer",
                                self.max_iterations.unwrap()
                            );
                            input_variables.insert_placeholder_replacement(
                                "ultimatum",
                                vec![
                                    Message::new(MessageType::AIMessage, ""),
                                    Message::new(MessageType::HumanMessage, FORCE_FINAL_ANSWER),
                                ],
                            );
                            // TODO: Add ultimatum template
                            continue 'step;
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
                        let Some(tool) = self.agent.get_tool(&tool_name) else {
                            consecutive_fails += 1;
                            log::warn!(
                                "Agent tried to use nonexistent tool {}, retrying ({} consecutive fails)",
                                action.action,
                                consecutive_fails
                            );
                            continue 'step;
                        };

                        if let Some(usage_limit) = tool.usage_limit() {
                            let count = use_counts.entry(tool_name.clone()).or_insert(0);
                            *count += 1;
                            if *count > usage_limit {
                                consecutive_fails += 1;
                                log::warn!(
                                    "Agent repeatedly using tool {} (usage limit: {}), preventing further use ({} consecutive fails)",
                                    action.action,
                                    usage_limit,
                                    consecutive_fails
                                );
                                continue 'step;
                            }
                        }

                        let observation = match tool.call(action.action_input.clone()).await {
                            Ok(observation) => observation,
                            Err(e) => {
                                log::error!(
                                    "Tool '{}' encountered an error: {}",
                                    &action.action,
                                    e
                                );
                                if self.break_if_tool_error {
                                    return Err(ChainError::AgentError(
                                        AgentError::ToolError(e.to_string()).to_string(),
                                    ));
                                } else {
                                    format!(
                                        indoc! {"
                                            Tool call failed: {}
                                            If the error doesn't make sense to you, it means that the tool is broken. DO NOT use this tool again.
                                        "},
                                        e
                                    )
                                }
                            }
                        };

                        log::debug!("Tool {} result:\n{}", &action.action, &observation);

                        steps.push((action, observation));
                        consecutive_fails = 0;
                    }
                }
                Ok(AgentEvent::Finish(final_answer)) => {
                    if let Some(memory) = &self.memory {
                        let mut memory = memory.lock().await;

                        memory.add_user_message(
                            match &input_variables.get_text_replacement("input") {
                                Some(input) => input,
                                None => &"",
                            },
                        );

                        for (action, observation) in steps {
                            memory.add_message(
                                Message::new(MessageType::AIMessage, "").with_tool_calls(vec![
                                    ChatCompletionMessageToolCall {
                                        id: action.id.clone(),
                                        r#type: ChatCompletionToolType::Function,
                                        function: FunctionCall {
                                            name: action.action,
                                            arguments: serde_json::to_string_pretty(
                                                &action.action_input,
                                            )
                                            .unwrap_or("Input parameters unknown".into()),
                                        },
                                    },
                                ]),
                            );
                            memory.add_message(Message::new_tool_message::<_, &str>(
                                Some(&action.id),
                                observation,
                            ));
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
                    consecutive_fails += 1;
                    log::warn!("Error: {} ({} consecutive fails)", e, consecutive_fails);
                }
            }
        }
    }

    async fn invoke(&self, input_variables: &mut InputVariables) -> Result<String, ChainError> {
        let result = self.call(input_variables).await?;
        Ok(result.generation)
    }

    fn log_messages(&self, inputs: &InputVariables) -> Result<(), Box<dyn Error>> {
        self.agent.log_messages(inputs)
    }
}
