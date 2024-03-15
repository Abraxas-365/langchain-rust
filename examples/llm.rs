use langchain_rust::{
    language_models::llm::LLM,
    llm::openai::{OpenAI, OpenAIConfig},
};

#[tokio::main]
async fn main() {
    //You can call Ollama this way
    // let ollama = OpenAI::default()
    //     .with_api_base("http://localhost:11434/v1")
    //     .with_api_key("ollama")
    //     .with_model("llama2");

    let open_ai = OpenAI::new(OpenAIConfig::default());
    let response = open_ai.invoke("hola").await.unwrap();
    println!("{}", response);
}
