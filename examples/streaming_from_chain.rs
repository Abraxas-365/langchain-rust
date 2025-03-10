use futures::StreamExt;
use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    fmt_message, fmt_template,
    llm::openai::OpenAI,
    message_formatter, plain_prompt_args,
    prompt::{FormatPrompter, HumanMessagePromptTemplate},
    schemas::messages::Message,
    template_fstring,
};

#[tokio::main]
async fn main() {
    let open_ai = OpenAI::default();

    let prompt = message_formatter![
        fmt_message!(Message::new_system_message(
            "You are world class technical documentation writer."
        )),
        fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
            "{input}", "input"
        )))
    ];

    let chain = LLMChainBuilder::new()
        .prompt(Box::new(prompt) as Box<dyn FormatPrompter<_>>)
        .llm(open_ai.clone())
        .build()
        .unwrap();

    let mut stream = chain
        .stream(&mut plain_prompt_args! {
            "input" => "Who is the writer of 20,000 Leagues Under the Sea?",
        })
        .await
        .unwrap();

    while let Some(result) = stream.next().await {
        match result {
            Ok(value) => value.to_stdout().unwrap(),
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
    }
}
