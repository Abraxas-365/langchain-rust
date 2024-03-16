use langchain_rust::{
    chain::{builder::ConversationalChainBuilder, Chain},
    llm::openai::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    prompt_args,
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

    let input_variables = prompt_args! {
        "input" => "Im from Peru",
    };

    match chain.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }

    let input_variables = prompt_args! {
        "input" => "Which are the typical dish",
    };

    match chain.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
