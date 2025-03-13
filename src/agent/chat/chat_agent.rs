use std::{collections::HashMap, error::Error, sync::Arc};

use async_trait::async_trait;
use indoc::indoc;

use crate::{
    agent::{agent::Agent, AgentError},
    chain::chain_trait::Chain,
    message_formatter, plain_prompt_args,
    prompt::{
        HumanMessagePromptTemplate, MessageFormatterStruct, MessageOrTemplate, PlainPromptArgs,
        PromptArgs, PromptFromatter,
    },
    schemas::{
        agent::{AgentAction, AgentEvent},
        messages::Message,
    },
    template_jinja2,
    tools::Tool,
};

use super::{parse::parse_agent_output, prompt::SUFFIX};

pub struct ConversationalAgent {
    pub(crate) chain: Box<dyn Chain<PlainPromptArgs>>,
    pub(crate) tools: HashMap<String, Arc<dyn Tool>>,
}

impl ConversationalAgent {
    pub fn create_prompt(
        system_prompt: &str,
        initial_prompt: &str,
        tools: &HashMap<String, Arc<dyn Tool>>,
    ) -> Result<MessageFormatterStruct<PlainPromptArgs>, AgentError> {
        let tool_string = tools
            .values()
            .map(|tool| tool.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let tool_names = tools.keys().cloned().collect::<Vec<_>>().join(", ");
        let input_variables_fstring = plain_prompt_args! {
            "tools" => tool_string,
            "tool_names" => tool_names
        };

        let system_prompt = template_jinja2!(
            &format!("{}{}", system_prompt, SUFFIX),
            "tools",
            "tool_names"
        )
        .format(&input_variables_fstring)?;

        let formatter = message_formatter![
            MessageOrTemplate::Message(Message::new_system_message(system_prompt)),
            MessageOrTemplate::MessagesPlaceholder("chat_history".to_string()),
            MessageOrTemplate::Template(Box::new(HumanMessagePromptTemplate::new(
                template_jinja2!(initial_prompt, "input")
            ))),
            MessageOrTemplate::MessagesPlaceholder("agent_scratchpad".to_string()),
        ];
        Ok(formatter)
    }

    fn construct_scratchpad(&self, intermediate_steps: &[(Option<AgentAction>, String)]) -> String {
        intermediate_steps
            .iter()
            .map(|(action, result)| match action {
                Some(action) => format!(
                    indoc! {"
                        Action: {}
                        Action input: {}
                        Result:
                        {}
                    "},
                    &action.action, &action.action_input, result
                ),
                None => result.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[async_trait]
impl Agent<PlainPromptArgs> for ConversationalAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(Option<AgentAction>, String)],
        inputs: &mut PlainPromptArgs,
    ) -> Result<AgentEvent, AgentError> {
        let scratchpad = self.construct_scratchpad(intermediate_steps);
        inputs.insert("agent_scratchpad".to_string(), scratchpad);
        let output = self.chain.call(inputs).await?.generation;
        log::trace!("Agent output: {}", output);
        let parsed_output = parse_agent_output(&output)?;
        Ok(parsed_output)
    }

    fn get_tool(&self, tool_name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(tool_name).cloned()
    }

    fn log_messages(&self, inputs: &PlainPromptArgs) -> Result<(), Box<dyn Error>> {
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
        plain_prompt_args,
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
        async fn parse_input(&self, input: Value) -> Result<String, Box<dyn Error>> {
            Ok(input.to_string())
        }
        async fn run(&self, _input: String) -> Result<i128, Box<dyn Error>> {
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
        let mut input_variables = plain_prompt_args! {
            "input" => "hola,Me llamo luis, y tengo 10 anos, y estudio Computer scinence",
        };
        let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());
        match executor.invoke(&mut input_variables).await {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
        let mut input_variables = plain_prompt_args! {
            "input" => "cuanta es la edad de luis +10 y que estudia",
        };
        match executor.invoke(&mut input_variables).await {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
    }
}
