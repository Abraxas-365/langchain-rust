// Persistent chat memory backed by SQLite.
//
// Run it twice: the second run remembers what you said in the first one.
//
//   cargo run --example persistent_memory --features sqlite-vec
//
// Any feature that enables sqlx works (sqlite-vec, sqlite-vss, postgres).
// Swap `SqliteMessageHistory` for `PostgresMessageHistory` to keep history in Postgres.

#[cfg(feature = "sqlx")]
use langchain_rust::{
    chain::{builder::ConversationalChainBuilder, Chain},
    llm::openai::OpenAI,
    memory::{SqliteMessageHistory, TokenBufferMemory},
    prompt_args,
    schemas::BaseMemory,
};

#[cfg(feature = "sqlx")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pool = sqlx::sqlite::SqlitePoolOptions::new()
        .connect("sqlite://chat_history.db?mode=rwc")
        .await?;

    // One history per user/conversation.
    let history = SqliteMessageHistory::new(pool, "user-42");
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

    // In-process alternative that keeps only what fits in a token budget.
    let mut short_term = TokenBufferMemory::new(1_000).with_model("gpt-4o")?;
    short_term.add_user_message(&"hello").await?;
    println!("\nToken buffer holds {} tokens", short_term.total_tokens());

    Ok(())
}

#[cfg(not(feature = "sqlx"))]
fn main() {
    println!("This example needs sqlx. Run it with:");
    println!("cargo run --example persistent_memory --features sqlite-vec");
}
