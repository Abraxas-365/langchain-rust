use futures::StreamExt;
use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    fmt_message, fmt_template,
    llm::openai::OpenAI,
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
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
        .prompt(prompt)
        .llm(open_ai.clone())
        .build()
        .unwrap();

    let mut stream = chain
        .stream(prompt_args! {
        "input" => "Who is the writer of 20,000 Leagues Under the Sea?",
           })
        .await
        .unwrap();
    while let Some(result) = stream.next().await {
        match result {
            Ok(value) => {
                if let Some(content) = value.pointer("/choices/0/delta/content") {
                    println!("Content: {}", content.as_str().unwrap_or(""));
                }
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
    }
}
