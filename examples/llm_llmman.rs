// llmman (https://github.com/llmmanorg/llmman) serves the Ollama API on port 17434,
// so the Ollama integration is reused with a different default endpoint.
// Start it with `llmman serve` and pull a model with `llmman pull gemma4`.
#[cfg(feature = "ollama")]
use langchain_rust::{language_models::llm::LLM, llm::ollama::client::Ollama};

#[cfg(feature = "ollama")]
#[tokio::main]
async fn main() {
    let llmman = Ollama::llmman().with_model("gemma4");

    let response = llmman.invoke("Hi").await.unwrap();
    println!("{}", response);
}

#[cfg(not(feature = "ollama"))]
fn main() {
    println!("This example requires the 'ollama' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example llm_llmman --features=ollama");
}
