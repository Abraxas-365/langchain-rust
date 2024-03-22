use std::sync::Arc;

use langchain_rust::{
    agent::{AgentExecutor, ChatOutputParser, ConversationalAgentBuilder},
    chain::{options::ChainCallOptions, Chain},
    llm::openai::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    prompt_args,
    tools::SerpApi,
};

#[tokio::main]
async fn main() {
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt4Turbo);
    let memory = SimpleMemory::new();
    let serpapi_tool = SerpApi::default();
    let agent = ConversationalAgentBuilder::new()
        .tools(&[Arc::new(serpapi_tool)])
        .output_parser(ChatOutputParser::new().into())
        .options(ChainCallOptions::new().with_max_tokens(1000))
        .build(llm)
        .unwrap();

    let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());

    let input_variables = prompt_args! {
        "input" => "Who is the creator of vim, and how old is vim",
    };

    match executor.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
