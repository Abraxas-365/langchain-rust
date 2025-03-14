use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::{options::ChainCallOptions, Chain},
    llm::openai::OpenAI,
    memory::SimpleMemory,
    plain_prompt_args,
    tools::{
        map_tools, CommandExecutor, DuckDuckGoSearch, SerpApi, Tool, ToolFunction, ToolWrapper,
    },
};

use serde_json::Value;

#[derive(Default)]
struct Date {}

#[async_trait]
impl ToolFunction for Date {
    type Input = ();
    type Result = String;

    fn name(&self) -> String {
        "Date".to_string()
    }

    fn description(&self) -> String {
        "Useful when you need to get the date, input should be an empty object ({})".to_string()
    }

    async fn parse_input(&self, _input: Value) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }

    async fn run(&self, _input: ()) -> Result<String, Box<dyn Error + Send + Sync>> {
        Ok("25 of november of 2025".to_string())
    }
}

impl From<Date> for Arc<dyn Tool> {
    fn from(val: Date) -> Self {
        Arc::new(ToolWrapper::new(val))
    }
}

#[tokio::main]
async fn main() {
    let llm = OpenAI::default();
    let memory = SimpleMemory::new();
    let serpapi_tool = SerpApi::default();
    let duckduckgo_tool = DuckDuckGoSearch::default();
    let tool_calc = Date::default();
    let command_executor = CommandExecutor::default();
    let agent = OpenAiToolAgentBuilder::new()
        .tools(map_tools(vec![
            serpapi_tool.into(),
            tool_calc.into(),
            command_executor.into(),
            duckduckgo_tool.into(),
        ]))
        .options(ChainCallOptions::new().with_max_tokens(1000))
        .build(llm)
        .unwrap();

    let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());

    let mut input_variables = plain_prompt_args! {
        "input" => "What the name of the current dir, And what date is today",
    };

    match executor.invoke(&mut input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result.replace("\n", " "));
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
