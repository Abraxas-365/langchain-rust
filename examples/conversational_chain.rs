use std::io::{stdout, Write};

use futures_util::StreamExt;
use langchain_rust::{
    chain::{builder::ConversationalChainBuilder, Chain},
    llm::openai::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
};

#[tokio::main]
async fn main() {
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
    //We initialise a simple memroy,by default conveational chain have thsi memory, but we
    //initiliase it as an example, if you dont want to have memory use DummyMemory
    let memory = SimpleMemory::new();

    let chain = ConversationalChainBuilder::new()
        .llm(llm)
        .memory(memory.into())
        .build()
        .expect("Error building ConversationalChain");

    let input_variables = chain.pompt_builder().input("Im from Peru").build();

    let mut stream = chain.stream(input_variables).await.unwrap();
    while let Some(result) = stream.next().await {
        match result {
            Ok(data) => {
                //If you junt want to print to stdout, you can use data.to_stdout().unwrap();
                print!("{}", data.content);
                stdout().flush().unwrap();
            }
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }

    let input_variables = chain
        .pompt_builder()
        .input("Which are the typical dish")
        .build();

    match chain.invoke(input_variables).await {
        Ok(result) => {
            println!("\n");
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
