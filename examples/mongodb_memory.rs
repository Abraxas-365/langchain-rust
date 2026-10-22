// Chat memory in MongoDB, shared by every process that uses the same session id.
//
// Run it twice: the second run remembers what you said in the first one.
//
//   docker run -d -p 27017:27017 mongo:8
//   OPENAI_API_KEY=... cargo run --example mongodb_memory --features mongodb
//
// Or with a local model through Ollama's OpenAI-compatible endpoint:
//   MEMORY_OLLAMA_MODEL=qwen3:14b cargo run --example mongodb_memory --features mongodb
//
// Set MONGODB_URI to use another server (any MongoDB works; Atlas isn't required).

#[cfg(feature = "mongodb")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use langchain_rust::{
        chain::{builder::ConversationalChainBuilder, Chain},
        llm::{openai::OpenAI, OpenAIConfig},
        memory::MongoMessageHistory,
        prompt_args,
        schemas::BaseMemory,
    };

    let uri = std::env::var("MONGODB_URI").unwrap_or("mongodb://localhost:27017".into());
    // One history per user/conversation, in the `chat_history.message_store` collection.
    let history = MongoMessageHistory::from_uri(&uri, "user-42").await?;
    println!("Stored so far:\n{}\n", history.to_string().await?);

    let llm = match std::env::var("MEMORY_OLLAMA_MODEL") {
        Ok(model) => OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model),
        Err(_) => OpenAI::default(),
    };
    let chain = ConversationalChainBuilder::new()
        .llm(llm)
        .memory(history.into())
        .build()?;

    let answer = chain
        .invoke(prompt_args! {"input" => "What do you remember about me? If nothing, ask my name."})
        .await?;
    println!("AI: {answer}");

    let answer = chain
        .invoke(prompt_args! {"input" => "My name is Luis and I'm from Peru."})
        .await?;
    println!("AI: {answer}");

    Ok(())
}

#[cfg(not(feature = "mongodb"))]
fn main() {
    println!("Run with: cargo run --example mongodb_memory --features mongodb");
}
