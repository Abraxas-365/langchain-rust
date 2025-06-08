use langchain_rust::{
    language_models::llm::LLM,
    llm::nebius::{Nebius, NebiusModel},
};

#[tokio::main]
async fn main() {
    // Set Nebius API key environment variable
    // export NEBIUS_API_KEY=your_api_key

    //Nebius Example - Basic usage
    let nebius = Nebius::new();
    let response = nebius
        .invoke("Hello! Can you explain what Rust is?")
        .await
        .unwrap();
    println!("Basic Nebius response:");
    println!("{}", response);
    println!("\n{}", "=".repeat(50));

    // Using with specific model
    let nebius = Nebius::new().with_model(NebiusModel::DeepSeekV3);

    let response = nebius
        .invoke("What are the main advantages of using Rust programming language?")
        .await
        .unwrap();
    println!("\nNebius with specific model:");
    println!("{}", response);
}
