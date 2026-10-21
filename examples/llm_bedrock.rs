//! AWS Bedrock example.
//!
//! Requires AWS credentials in the environment (`AWS_ACCESS_KEY_ID` /
//! `AWS_SECRET_ACCESS_KEY` or a configured profile) and model access enabled
//! for the chosen model in the Bedrock console.
//!
//! Run with: cargo run --features bedrock --example llm_bedrock

#[cfg(feature = "bedrock")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use langchain_rust::language_models::llm::LLM;
    use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
    use langchain_rust::schemas::Message;

    let llm = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude35Haiku)
        .with_region("us-east-1")
        .with_temperature(0.3)
        .with_max_tokens(200);

    // Simple prompt
    let answer = llm
        .invoke("What is the capital of France? Answer in one word.")
        .await?;
    println!("invoke: {answer}");

    // Chat-style messages with a system prompt
    let result = llm
        .generate(&[
            Message::new_system_message("You are a terse assistant."),
            Message::new_human_message("Name three Rust web frameworks."),
        ])
        .await?;
    println!("generate: {}", result.generation);
    if let Some(usage) = result.tokens {
        println!(
            "tokens: {} in / {} out",
            usage.prompt_tokens, usage.completion_tokens
        );
    }

    Ok(())
}

#[cfg(not(feature = "bedrock"))]
fn main() {
    println!("This example requires the 'bedrock' feature: cargo run --features bedrock --example llm_bedrock");
}
