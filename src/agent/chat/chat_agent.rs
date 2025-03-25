use std::{collections::HashMap, error::Error, sync::Arc};

use async_openai::types::{ChatCompletionMessageToolCall, ChatCompletionToolType, FunctionCall};
use async_trait::async_trait;

use crate::{
    agent::{agent::Agent, AgentError},
    chain::chain_trait::Chain,
    prompt_template,
    schemas::{
        agent::{AgentAction, AgentEvent},
        InputVariables, Message, MessageType,
    },
    template::{MessageOrTemplate, MessageTemplate, PromptTemplate},
    text_replacements,
    tools::Tool,
};

use super::{parse::parse_agent_output, prompt::SUFFIX};

pub struct ConversationalAgent {
    pub(crate) chain: Box<dyn Chain>,
    pub(crate) tools: HashMap<String, Arc<dyn Tool>>,
}

impl ConversationalAgent {
    pub fn create_prompt(
        system_prompt: &str,
        initial_prompt: &str,
        tools: &HashMap<String, Arc<dyn Tool>>,
    ) -> Result<PromptTemplate, AgentError> {
        let tool_string = tools
            .values()
            .map(|tool| tool.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let tool_names = tools.keys().cloned().collect::<Vec<_>>().join(", ");
        let input_variables_fstring: InputVariables = text_replacements! {
            "tools" => tool_string,
            "tool_names" => tool_names
        }
        .into();

        let system_prompt = MessageTemplate::from_jinja2(
            MessageType::SystemMessage,
            &format!("{}{}", system_prompt, SUFFIX),
        )
        .format(&input_variables_fstring)?;

        let formatter = prompt_template![
            system_prompt,
            MessageOrTemplate::Placeholder("chat_history".into()),
            MessageTemplate::from_jinja2(MessageType::HumanMessage, initial_prompt),
            MessageOrTemplate::Placeholder("agent_scratchpad".into())
        ];
        Ok(formatter)
    }

    fn construct_scratchpad(&self, intermediate_steps: &[(AgentAction, String)]) -> Vec<Message> {
        intermediate_steps
            .iter()
            .flat_map(|(action, result)| {
                vec![
                    Message::new(MessageType::AIMessage, "").with_tool_calls(vec![
                        ChatCompletionMessageToolCall {
                            id: action.id.clone(),
                            r#type: ChatCompletionToolType::Function,
                            function: FunctionCall {
                                name: action.action.clone(),
                                arguments: serde_json::to_string_pretty(&action.action_input)
                                    .unwrap_or("Input parameters unknown".into()),
                            },
                        },
                    ]),
                    Message::new_tool_message(Some(action.id.clone()), result),
                ]
            })
            .collect::<Vec<_>>()
    }
}

#[async_trait]
impl Agent for ConversationalAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: &mut InputVariables,
    ) -> Result<AgentEvent, AgentError> {
        let scratchpad = self.construct_scratchpad(intermediate_steps);
        inputs.insert_placeholder_replacement("agent_scratchpad", scratchpad);
        let output = self.chain.call(inputs).await?.generation;
        log::trace!("Agent output: {}", output);
        let parsed_output = parse_agent_output(&output);
        Ok(parsed_output)
    }

    fn get_tool(&self, tool_name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(tool_name).cloned()
    }

    fn log_messages(&self, inputs: &InputVariables) -> Result<(), Box<dyn Error>> {
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
        schemas::InputVariables,
        text_replacements,
        tools::{map_tools, Tool, ToolFunction, ToolWrapper},
    };

    #[derive(Default)]
    struct Calc {}

    #[async_trait]
    impl ToolFunction for Calc {
        type Input = String;
        type Result = i128;

        fn name(&self) -> String {
            "Calculator".to_string()
        }
        fn description(&self) -> String {
            "Usefull to make calculations".to_string()
        }
        async fn parse_input(&self, input: Value) -> Result<String, Box<dyn Error + Send + Sync>> {
            Ok(input.to_string())
        }
        async fn run(&self, _input: String) -> Result<i128, Box<dyn Error + Send + Sync>> {
            Ok(25)
        }
    }

    impl From<Calc> for Arc<dyn Tool> {
        fn from(val: Calc) -> Self {
            Arc::new(ToolWrapper::new(val))
        }
    }

    #[tokio::test]
    #[ignore]
    async fn test_invoke_agent() {
        let llm = OpenAI::default().with_model(OpenAIModel::Gpt4.to_string());
        let memory = SimpleMemory::new();
        let tool_calc = Calc::default();
        let agent = ConversationalAgentBuilder::new()
            .tools(map_tools(vec![tool_calc.into()]))
            .build(llm)
            .unwrap();
        let mut input_variables: InputVariables = text_replacements! {
            "input" => "hola,Me llamo luis, y tengo 10 anos, y estudio Computer scinence",
        }
        .into();
        let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());
        match executor.invoke(&mut input_variables).await {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
        let mut input_variables: InputVariables = text_replacements! {
            "input" => "cuanta es la edad de luis +10 y que estudia",
        }
        .into();
        match executor.invoke(&mut input_variables).await {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
    }
}
