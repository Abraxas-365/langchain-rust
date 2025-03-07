use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use indoc::indoc;
use serde_json::Value;

use crate::{
    agent::{agent::Agent, AgentError},
    chain::chain_trait::Chain,
    message_formatter,
    prompt::{
        HumanMessagePromptTemplate, MessageFormatterStruct, MessageOrTemplate, PromptArgs,
        PromptFromatter,
    },
    prompt_args,
    schemas::{
        agent::{AgentAction, AgentEvent},
        messages::Message,
    },
    template_jinja2,
    tools::Tool,
};

use super::{parse::parse_agent_output, prompt::SUFFIX};

pub struct ConversationalAgent {
    pub(crate) chain: Box<dyn Chain>,
    pub(crate) tools: Vec<Arc<dyn Tool>>,
}

impl ConversationalAgent {
    pub fn create_prompt(
        system_prompt: &str,
        initial_prompt: &str,
        tools: &[Arc<dyn Tool>],
    ) -> Result<MessageFormatterStruct, AgentError> {
        let tool_string = tools
            .iter()
            .map(|tool: &Arc<dyn Tool>| tool.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let tool_names = tools
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>()
            .join(", ");
        let input_variables_fstring = prompt_args! {
            "tools" => tool_string,
            "tool_names" => tool_names
        };

        let system_prompt = template_jinja2!(
            &format!("{}{}", system_prompt, SUFFIX),
            "tools",
            "tool_names"
        )
        .format(input_variables_fstring)?;

        let formatter = message_formatter![
            MessageOrTemplate::Message(Message::new_system_message(system_prompt)),
            MessageOrTemplate::MessagesPlaceholder("chat_history".to_string()),
            MessageOrTemplate::Template(
                HumanMessagePromptTemplate::new(template_jinja2!(initial_prompt, "input")).into()
            ),
            MessageOrTemplate::MessagesPlaceholder("agent_scratchpad".to_string()),
        ];
        Ok(formatter)
    }

    fn construct_scratchpad(&self, intermediate_steps: &[(AgentAction, String)]) -> String {
        intermediate_steps
            .iter()
            .map(|(action, result)| {
                if let Some(thought) = &action.thought {
                    format!(
                        indoc! {"
                            Thought: {}
                            Action: {}
                            Action input: {}
                            Result:
                            {}
                        "},
                        thought, action.action, action.action_input, result
                    )
                } else {
                    format!(
                        indoc! {"
                            Action: {}
                            Action input: {}
                            Result:
                            {}
                        "},
                        action.action, action.action_input, result
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[async_trait]
impl Agent for ConversationalAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, AgentError> {
        let scratchpad = self.construct_scratchpad(intermediate_steps);
        let mut inputs = inputs.clone();
        inputs.insert("agent_scratchpad".to_string(), Value::String(scratchpad));
        let output = self.chain.call(inputs.clone()).await?.generation;
        let parsed_output = parse_agent_output(&output)?;
        Ok(parsed_output)
    }

    fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    fn log_messages(&self, inputs: PromptArgs) -> Result<(), Box<dyn Error>> {
        self.chain.log_messages(inputs)
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, sync::Arc};

    use async_trait::async_trait;
    use serde_json::Value;

    use crate::{
        agent::{chat::builder::ConversationalAgentBuilder, executor::AgentExecutor},
        chain::chain_trait::Chain,
        llm::openai::{OpenAI, OpenAIModel},
        memory::SimpleMemory,
        prompt_args,
        tools::Tool,
    };

    struct Calc {}

    #[async_trait]
    impl Tool for Calc {
        fn name(&self) -> String {
            "Calculator".to_string()
        }
        fn description(&self) -> String {
            "Usefull to make calculations".to_string()
        }
        async fn run(&self, _input: Value) -> Result<String, Box<dyn Error>> {
            Ok("25".to_string())
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_invoke_agent() {
        let llm = OpenAI::default().with_model(OpenAIModel::Gpt4.to_string());
        let memory = SimpleMemory::new();
        let tool_calc = Calc {};
        let agent = ConversationalAgentBuilder::new()
            .tools(&[Arc::new(tool_calc)])
            .build(llm)
            .unwrap();
        let input_variables = prompt_args! {
            "input" => "hola,Me llamo luis, y tengo 10 anos, y estudio Computer scinence",
        };
        let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());
        match executor.invoke(input_variables).await {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
        let input_variables = prompt_args! {
            "input" => "cuanta es la edad de luis +10 y que estudia",
        };
        match executor.invoke(input_variables).await {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
    }
}
