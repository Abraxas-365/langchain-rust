use std::sync::Arc;

use langchain_rust::{
    agent::{AgentExecutor, ConversationalAgentBuilder},
    chain::{options::ChainCallOptions, Chain},
    llm::openai::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    plain_prompt_args,
    tools::{map_tools, CommandExecutor},
};

#[tokio::main]
async fn main() {
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt4Turbo);
    let memory = SimpleMemory::new();
    let command_executor = CommandExecutor::default();
    let agent = ConversationalAgentBuilder::new()
        .tools(map_tools(vec![Arc::new(command_executor)]))
        .options(ChainCallOptions::new().with_max_tokens(1000))
        .build(llm)
        .unwrap();

    let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());

    let mut input_variables = plain_prompt_args! {
        "input" => "What is the name of the current dir",
    };

    match executor.invoke(&mut input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
