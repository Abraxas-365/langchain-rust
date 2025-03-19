use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    input_variables,
    llm::openai::{OpenAI, OpenAIModel},
    schemas::{MessageTemplate, MessageType},
    sequential_chain,
};
use std::io::{self, Write}; // Include io Library for terminal input

#[tokio::main]
async fn main() {
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
    let prompt = MessageTemplate::from_jinja2(
        MessageType::HumanMessage,
        "Dame un nombre creativo para una tienda que vende: {{producto}}",
    );

    let get_name_chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(llm.clone())
        .output_key("name")
        .build()
        .unwrap();

    let prompt = MessageTemplate::from_jinja2(
        MessageType::HumanMessage,
        "Dame un slogan para el siguiente nombre: {{name}}",
    );
    let get_slogan_chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(llm.clone())
        .output_key("slogan")
        .build()
        .unwrap();

    let sequential_chain = sequential_chain!(get_name_chain, get_slogan_chain);

    print!("Please enter a product: ");
    io::stdout().flush().unwrap(); // Display prompt to terminal

    let mut product = String::new();
    io::stdin().read_line(&mut product).unwrap(); // Get product from terminal input

    let product = product.trim();
    let output = sequential_chain
        .execute(&mut input_variables! {
            "producto" => product
        })
        .await
        .unwrap();

    println!("Name: {}", output["name"]);
    println!("Slogan: {}", output["slogan"]);
}
