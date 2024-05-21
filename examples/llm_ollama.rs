use langchain_rust::{language_models::llm::LLM, llm::ollama::client::Ollama};

#[tokio::main]
async fn main() {
    let ollama = Ollama::default().with_model("llama3");

    let response = ollama.invoke("hola").await.unwrap();
    println!("{}", response);
}
