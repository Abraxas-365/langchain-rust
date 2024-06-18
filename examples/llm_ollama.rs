#[cfg(feature = "ollama")]
use langchain_rust::{language_models::llm::LLM, llm::ollama::client::Ollama};

#[cfg(feature = "ollama")]
#[tokio::main]
async fn main() {
    let ollama = Ollama::default().with_model("llama3");

    let response = ollama.invoke("Hi").await.unwrap();
    println!("{}", response);
}

#[cfg(not(feature = "ollama"))]
fn main() {
    println!("This example requires the 'ollama' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example llm_ollama --features=ollama");
}
