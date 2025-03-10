use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    llm::openai::{OpenAI, OpenAIModel},
    plain_prompt_args,
    prompt::{FormatPrompter, HumanMessagePromptTemplate},
    template_jinja2,
};
use std::io::{self, Write}; // Include io Library for terminal input

#[tokio::main]
async fn main() {
    let prompt = HumanMessagePromptTemplate::new(template_jinja2!(
        "Give me a creative name for a store that sells: {{producto}}",
        "producto"
    ));

    let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
    let chain = LLMChainBuilder::new()
        .prompt(Box::new(prompt) as Box<dyn FormatPrompter<_>>)
        .llm(llm)
        .build()
        .unwrap();

    print!("Please enter a product: ");
    io::stdout().flush().unwrap(); // Display prompt to terminal

    let mut product = String::new();
    io::stdin().read_line(&mut product).unwrap(); // Get product from terminal input

    let product = product.trim();

    let output = chain
        .invoke(&mut plain_prompt_args! { "producto" => product }) // Use product input here
        .await
        .unwrap();

    println!("Output: {}", output);
}
