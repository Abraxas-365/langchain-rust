// Chat memory in Redis, shared by every process that uses the same session id.
//
// Run it twice: the second run remembers what you said in the first one.
//
//   redis-server   # or: docker run -p 6379:6379 redis
//   OPENAI_API_KEY=... cargo run --example redis_memory --features redis

#[cfg(feature = "redis")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Duration;

    use langchain_rust::{
        chain::{builder::ConversationalChainBuilder, Chain},
        llm::openai::OpenAI,
        memory::RedisMessageHistory,
        prompt_args,
        schemas::BaseMemory,
    };

    // One history per user/conversation; forget it after a week without messages.
    let history = RedisMessageHistory::from_url("redis://127.0.0.1/", "user-42")
        .await?
        .with_ttl(Duration::from_secs(7 * 24 * 3600));
    println!("Stored so far:\n{}\n", history.to_string().await?);

    let chain = ConversationalChainBuilder::new()
        .llm(OpenAI::default())
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

#[cfg(not(feature = "redis"))]
fn main() {
    println!("Run with: cargo run --example redis_memory --features redis");
}
