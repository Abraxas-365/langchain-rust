use futures::StreamExt;
use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    fmt_message, fmt_template,
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::messages::Message,
    template_fstring,
};

use langchain_rust::llm::nanogpt::*;

use dotenvy::{self, dotenv};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() {
    dotenv().expect(".env file not found");
    // Serving as an exmaple of using non major LLM provider, tracing logs are incredibly useful here.

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .with_file(true)
        .with_line_number(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let nano_config = OpenAIConfig::new()
        .with_api_base("https://nano-gpt.com/api/v1")
        .with_api_key(dotenvy::var("NANOGPT_KEY").expect("provide key"));

    let open_ai = NanoGPT::default()
        .with_config(nano_config)
        .with_model("deepseek-ai/DeepSeek-V3.1:thinking");

    let prompt = message_formatter![
        fmt_message!(Message::new_system_message(
            "You excel at metacognition. You are a true philosopher, becacuse you are a piece of consciousness with no ego"
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
        "input" => "Contemplate the intricate web of causality and how that affects one's being, over and over",
           })
        .await
        .unwrap();

    while let Some(result) = stream.next().await {
        match result {
            Ok(value) => value.to_stdout().unwrap(),
            Err(e) => println!("protocol non compliance: {:?}", e),
        }
    }
}
