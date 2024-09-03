use langchain_rust::chain::{Chain, LLMChainBuilder};
use langchain_rust::llm::OpenAI;
use langchain_rust::prompt::HumanMessagePromptTemplate;
use langchain_rust::schemas::Message;
use langchain_rust::{fmt_message, fmt_template, message_formatter, prompt_args, template_fstring};

#[tokio::main]
async fn main() {
    // Or can be a base64 encoded string of the image
    let image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg";
    let prompt = message_formatter![
        fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
            "{input}", "input"
        ))),
        fmt_message!(Message::new_human_message_with_images(vec![image_url])),
    ];

    let open_ai = OpenAI::default();
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(open_ai)
        .build()
        .unwrap();

    match chain
        .invoke(prompt_args! { "input" => "Describe this image"})
        .await
    {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
