use base64::prelude::*;
use langchain_rust::chain::{Chain, LLMChainBuilder};
use langchain_rust::llm::OpenAI;
use langchain_rust::prompt::HumanMessagePromptTemplate;
use langchain_rust::{prompt_args, template_fstring};

#[tokio::main]
async fn main() {
    // Images can be http(s) URLs, data: URLs or bare base64.
    let image = std::fs::read("./src/llm/test_data/example.jpg").unwrap();
    let image_base64 = BASE64_STANDARD.encode(image);

    // The human message takes its text from `input` and its images from `images`.
    let prompt = HumanMessagePromptTemplate::new(template_fstring!("{input}", "input"))
        .with_images("images");

    // Any vision model works, e.g. a local one through Ollama:
    // let open_ai = OpenAI::new(langchain_rust::llm::ollama::openai::OllamaConfig::default())
    //     .with_model("gemma4:e4b");
    let open_ai = OpenAI::default();
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(open_ai)
        .build()
        .unwrap();

    match chain
        .invoke(prompt_args! {
            "input" => "Describe this image",
            "images" => vec![image_base64],
        })
        .await
    {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
