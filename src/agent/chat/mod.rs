use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use serde_json::json;

use crate::{
    agent::chat::prompt::FORMAT_INSTRUCTIONS,
    chain::{chain_trait::Chain, llm_chain::LLMChain},
    language_models::llm::LLM,
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
    tools::tool::Tool,
};

use super::agent::{Agent, AgentOutputParser};

pub mod output_parser;
pub mod prompt;

pub struct ConversationalAgent {
    chain: Box<dyn Chain>,
    tools: Vec<Arc<dyn Tool>>,
    output_parser: Box<dyn AgentOutputParser>,
}

impl ConversationalAgent {
    fn create_prompt(
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

    pub fn from_llm_and_tools<L: LLM + 'static>(
        llm: L,
        tools: Vec<Arc<dyn Tool>>,
        output_parser: Box<dyn AgentOutputParser>,
    ) -> Result<Self, Box<dyn Error>> {
        let prompt = ConversationalAgent::create_prompt(&tools, prompt::SUFFIX, prompt::PREFIX)?;
        let chain = Box::new(LLMChain::new(prompt, llm));
        Ok(Self {
            chain,
            tools,
            output_parser,
        })
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
        println!("Output: {:?}", output);
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
            chat::{output_parser::ChatOutputParser, ConversationalAgent},
            executor::AgentExecutor,
        },
        chain::chain_trait::Chain,
        llm::openai::{OpenAI, OpenAIModel},
        prompt_args,
        schemas::memory::SimpleMemory,
        tools::tool::Tool,
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
        let agent = ConversationalAgent::from_llm_and_tools(
            llm,
            vec![Arc::new(tool_calc)],
            ChatOutputParser::new().into(),
        )
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
