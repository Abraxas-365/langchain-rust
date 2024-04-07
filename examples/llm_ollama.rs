use langchain_rust::llm::OpenAIConfig;

use langchain_rust::{language_models::llm::LLM, llm::openai::OpenAI};

#[tokio::main]
async fn main() {
    //Since Ollmama is OpenAi compatible
    //You can call Ollama this way:
    let ollama = OpenAI::default()
        .with_config(
            OpenAIConfig::default()
                .with_api_base("http://localhost:11434/v1")
                .with_api_key("ollama"),
        )
        .with_model("llama2");

    let response = ollama.invoke("hola").await.unwrap();
    println!("{}", response);
}
