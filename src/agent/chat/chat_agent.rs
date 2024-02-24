use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use serde_json::json;

use crate::{
    agent::{
        agent::{Agent, AgentOutputParser},
        chat::prompt::FORMAT_INSTRUCTIONS,
    },
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

use super::prompt::TEMPLATE_TOOL_RESPONSE;

pub struct ConversationalAgent {
    pub(crate) chain: Box<dyn Chain>,
    pub(crate) tools: Vec<Arc<dyn Tool>>,
    pub(crate) output_parser: Box<dyn AgentOutputParser>,
}

impl ConversationalAgent {
    pub fn create_prompt(
        tools: &[Arc<dyn Tool>],
        suffix: &str,
        prefix: &str,
    ) -> Result<MessageFormatterStruct, Box<dyn Error>> {
        let tool_string = tools
            .iter()
            .map(|tool| format!("> {}: {}", tool.name(), tool.description()))
            .collect::<Vec<_>>()
            .join("\n");
        let tool_names = tools
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>()
            .join(", ");

        let sufix_prompt = template_jinja2!(suffix, "tools", "format_instructions");

        let input_variables_fstring = prompt_args! {
            "tools" => tool_string,
            "format_instructions" => FORMAT_INSTRUCTIONS,
            "tool_names"=>tool_names
        };

        let sufix_prompt = sufix_prompt.format(input_variables_fstring)?;
        let formatter = message_formatter![
            MessageOrTemplate::Message(Message::new_system_message(prefix)),
            MessageOrTemplate::MessagesPlaceholder("chat_history".to_string()),
            MessageOrTemplate::Template(
                HumanMessagePromptTemplate::new(template_jinja2!(
                    &sufix_prompt.to_string(),
                    "input"
                ))
                .into()
            ),
            MessageOrTemplate::MessagesPlaceholder("agent_scratchpad".to_string()),
        ];
        return Ok(formatter);
    }

    fn construct_scratchpad(
        &self,
        intermediate_steps: &[(AgentAction, String)],
    ) -> Result<Vec<Message>, Box<dyn Error>> {
        let mut thoughts: Vec<Message> = Vec::new();
        for (action, observation) in intermediate_steps.into_iter() {
            thoughts.push(Message::new_ai_message(&action.log));
            let tool_response = template_jinja2!(TEMPLATE_TOOL_RESPONSE, "observation")
                .format(prompt_args!("observation"=>observation))?;
            thoughts.push(Message::new_human_message(&tool_response));
        }
        Ok(thoughts)
    }
}

#[async_trait]
impl Agent for ConversationalAgent {
    async fn plan(
        &self,
        intermediate_steps: &[(AgentAction, String)],
        inputs: PromptArgs,
    ) -> Result<AgentEvent, Box<dyn Error>> {
        let scratchpad = self.construct_scratchpad(&intermediate_steps)?;
        let mut inputs = inputs.clone();
        inputs.insert("agent_scratchpad".to_string(), json!(scratchpad));
        let output = self.chain.call(inputs.clone()).await?.generation;
        let parsed_output = self.output_parser.parse(&output)?;
        Ok(parsed_output)
    }

    fn get_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::{error::Error, sync::Arc};

    use async_trait::async_trait;

    use crate::{
        agent::{
            chat::{builder::ConversationalAgentBuilder, output_parser::ChatOutputParser},
            executor::AgentExecutor,
        },
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
        async fn call(&self, _input: &str) -> Result<String, Box<dyn Error>> {
            Ok("25".to_string())
        }
    }

    #[tokio::test]
    async fn test_invoke_agent() {
        let llm = OpenAI::default().with_model(OpenAIModel::Gpt4);
        let memory = SimpleMemory::new();
        let tool_calc = Calc {};
        let agent = ConversationalAgentBuilder::new()
            .tools(vec![Arc::new(tool_calc)])
            .output_parser(ChatOutputParser::new().into())
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
