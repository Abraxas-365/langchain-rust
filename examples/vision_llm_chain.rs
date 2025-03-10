use base64::prelude::*;
use langchain_rust::chain::{Chain, LLMChainBuilder};
use langchain_rust::llm::OpenAI;
use langchain_rust::prompt::{FormatPrompter, HumanMessagePromptTemplate};
use langchain_rust::schemas::Message;
use langchain_rust::{
    fmt_message, fmt_template, message_formatter, plain_prompt_args, template_fstring,
};

#[tokio::main]
async fn main() {
    // Convert image to base64. Can also pass a link to an image instead.
    let image = std::fs::read("./src/llm/test_data/example.jpg").unwrap();
    let image_base64 = BASE64_STANDARD.encode(image);

    let prompt = message_formatter![
        fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
            "{input}", "input"
        ))),
        fmt_message!(Message::new_human_message_with_images(vec![format!(
            "data:image/jpeg;base64,{image_base64}"
        )])),
    ];

    // let open_ai = OpenAI::new(langchain_rust::llm::ollama::openai::OllamaConfig::default())
    //     .with_model("llava");
    let open_ai = OpenAI::default();
    let chain = LLMChainBuilder::new()
        .prompt(Box::new(prompt) as Box<dyn FormatPrompter<_>>)
        .llm(open_ai)
        .build()
        .unwrap();

    match chain
        .invoke(&mut plain_prompt_args! { "input" => "Describe this image"})
        .await
    {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
