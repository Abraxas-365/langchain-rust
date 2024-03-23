use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::{options::ChainCallOptions, Chain},
    llm::openai::OpenAI,
    memory::SimpleMemory,
    prompt_args,
    tools::{SerpApi, Tool,CommandExecutor},
};

use serde_json::Value;
struct Date {}

#[async_trait]
impl Tool for Date {
    fn name(&self) -> String {
        "Date".to_string()
    }
    fn description(&self) -> String {
        "Useful when you need to get the date,input is  a query".to_string()
    }
    async fn run(&self, _input: Value) -> Result<String, Box<dyn Error>> {
        Ok("25  of november of 2025".to_string())
    }
}

#[tokio::main]
async fn main() {
    let llm = OpenAI::default();
    let memory = SimpleMemory::new();
    let serpapi_tool = SerpApi::default();
    let tool_calc = Date {};
    let command_executor = CommandExecutor::default().with_disallowed_commands(vec![("rm", vec!["-rf"])]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(serpapi_tool), Arc::new(tool_calc),Arc::new(command_executor)])
        .options(ChainCallOptions::new().with_max_tokens(1000))
        .build(llm)
        .unwrap();

    let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());

    let input_variables = prompt_args! {
        "input" => "What do I have in the current dir, what is the name of the current dir and Whats the current date?",
    };

    match executor.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
