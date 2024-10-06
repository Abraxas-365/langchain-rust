use langchain_rust::llm::OpenAIConfig;
use langchain_rust::{language_models::llm::LLM, llm::openai::OpenAI};

#[tokio::main]
async fn main() {
    let open_ai = OpenAI::default().with_config(
        OpenAIConfig::default()
            .with_api_base("http://localhost:8181/v1")
            .with_api_key("<YOUR-API-KEY>"), // Uncomment if you need to set your OpenAI key
    );
    
    let prompt = "Building a website can be done in 10 simple steps:";
    
    match open_ai.invoke(prompt).await {
        Ok(response) => println!("{}", response),
        Err(e) => eprintln!("Error: {:?}", e),
    }
}