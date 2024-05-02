use std::io::{stdout, Write};

use futures_util::StreamExt;
use langchain_rust::{
    chain::{builder::ConversationalChainBuilder, Chain},
    // fmt_message, fmt_template,
    llm::openai::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    // message_formatter,
    // prompt::HumanMessagePromptTemplate,
    prompt_args,
    // schemas::Message,
    // template_fstring,
};

#[tokio::main]
async fn main() {
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
    //We initialise a simple memory,by default conveational chain have this memory, but we
    //initiliase it as an example, if you dont want to have memory use DummyMemory
    let memory = SimpleMemory::new();

    let chain = ConversationalChainBuilder::new()
        .llm(llm)
        //IF YOU WANT TO ADD A CUSTOM PROMPT YOU CAN UN COMMENT THIS:
        //         .prompt(message_formatter![
        //             fmt_message!(Message::new_system_message("You are a helpful assistant")),
        //             fmt_template!(HumanMessagePromptTemplate::new(
        //             template_fstring!("
        // The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
        //
        // Current conversation:
        // {history}
        // Human: {input}
        // AI:
        // ",
        //             "input","history")))
        //
        //         ])
        .memory(memory.into())
        .build()
        .expect("Error building ConversationalChain");

    let input_variables = prompt_args! {
        "input" => "Im from Peru",
    };

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

    let input_variables = prompt_args! {
        "input" => "Which are the typical dish",
    };
    match chain.invoke(input_variables).await {
        Ok(result) => {
            println!("\n");
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
