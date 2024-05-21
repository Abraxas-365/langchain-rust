use langchain_rust::{
    language_models::llm::LLM,
    llm::{ollama::openai::OllamaConfig, openai::OpenAI},
};

#[tokio::main]
async fn main() {
    // since Ollama is OpenAI compatible, you can use it as below:
    let ollama = OpenAI::new(OllamaConfig::default()).with_model("llama2");

    let response = ollama.invoke("hola").await.unwrap();
    println!("{}", response);
}
