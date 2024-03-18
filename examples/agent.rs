use std::sync::Arc;

use langchain_rust::{
    agent::{AgentExecutor, ChatOutputParser, ConversationalAgentBuilder},
    chain::{options::ChainCallOptions, Chain},
    llm::openai::OpenAI,
    memory::SimpleMemory,
    prompt_args,
    tools::Wolfram,
};

#[tokio::main]
async fn main() {
    let llm = OpenAI::default();
    let memory = SimpleMemory::new();
    let wolfram_tool = Wolfram::default();
    let agent = ConversationalAgentBuilder::new()
        .tools(vec![Arc::new(wolfram_tool)])
        .output_parser(ChatOutputParser::new().into())
        .options(ChainCallOptions::new().with_max_tokens(1000))
        .build(llm)
        .unwrap();

    let input_variables = prompt_args! {
        "input" => "Hello",
    };

    let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());
    match executor.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }

    let input_variables = prompt_args! {
        "input" => "What is the area under the curve e^{-x^2}?",
    };

    match executor.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
