use langchain_rust::embedding::embedder_trait::Embedder;
use langchain_rust::embedding::openai::openai_embedder::OpenAiEmbedder;
use langchain_rust::llm::OpenAIConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig::default()
        .with_api_base("http://localhost:8282/v1".to_string())
        .with_api_key("<YOUR-API-KEY>".to_string()); // llama.cpp doesn't require an API key

    let embedder = OpenAiEmbedder::new(config);

    let text = "Why is the sky blue?";
    let embedding = embedder.embed_query(text).await?;

    println!("Embedding for '{}': {:?}", text, embedding);

    Ok(())
}