use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    input_variables,
    llm::openai::{OpenAI, OpenAIModel},
    schemas::{MessageTemplate, MessageType},
};
use std::io::{self, Write}; // Include io Library for terminal input

#[tokio::main]
async fn main() {
    let prompt = MessageTemplate::from_jinja2(
        MessageType::HumanMessage,
        "Give me a creative name for a store that sells: {{producto}}",
    );

    let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(llm)
        .build()
        .unwrap();

    print!("Please enter a product: ");
    io::stdout().flush().unwrap(); // Display prompt to terminal

    let mut product = String::new();
    io::stdin().read_line(&mut product).unwrap(); // Get product from terminal input

    let product = product.trim();

    let output = chain
        .invoke(&mut input_variables! { "producto" => product }) // Use product input here
        .await
        .unwrap();

    println!("Output: {}", output);
}
