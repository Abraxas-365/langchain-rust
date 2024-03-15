use std::{error::Error, sync::Arc};

use async_openai::config::OpenAIConfig;
use async_trait::async_trait;
use langchain_rust::{
    agent::{AgentExecutor, ChatOutputParser, ConversationalAgentBuilder},
    chain::Chain,
    language_models::options::CallOptions,
    llm::openai::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    prompt_args,
    tools::Tool,
};

//First whe sould create a tool, i will create a mock tool
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
#[tokio::main]
async fn main() {
    let llm =
        OpenAI::new(OpenAIConfig::default(), CallOptions::default()).with_model(OpenAIModel::Gpt4);
    let memory = SimpleMemory::new();
    let tool_calc = Calc {};
    let agent = ConversationalAgentBuilder::new()
        .tools(vec![Arc::new(tool_calc)])
        .output_parser(ChatOutputParser::new().into())
        .build(llm)
        .unwrap();

    let input_variables = prompt_args! {
        "input" => "Hello, my name is Luis, I am 10 years old, and I study Computer Science.",
    };

    let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());
    match executor.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }

    let input_variables = prompt_args! {
        "input" => "What is the age of Luis +10 and what does he study?",
    };

    match executor.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
