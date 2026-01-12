//! Example usage of AWS Bedrock LLM integration
//!
//! This example demonstrates how to use the Bedrock LLM with the langchain-rust library.
//!
//! To run this example, you need:
//! 1. AWS credentials configured (via environment variables or ~/.aws/credentials)
//! 2. Access to AWS Bedrock in your region
//! 3. To run: cargo run --features aws-sdk-bedrockruntime --example llm_bedrock

use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;
use langchain_rust::schemas::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== AWS Bedrock LLM Example ===\n");

    // Example 1: Basic usage with Claude 3 Sonnet (known to work with on-demand)
    println!("Example 1: Basic Bedrock Query");
    println!("{}", "-".repeat(50));

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Sonnet)
        .with_max_tokens(100);

    let messages = vec![Message::new_human_message(
        "What is the capital of France?",
    )];

    match bedrock.generate(&messages).await {
        Ok(result) => {
            println!("Question: What is the capital of France?");
            println!("Response: {}\n", result.generation);
        }
        Err(e) => {
            eprintln!("Error: {}. Make sure you have AWS credentials configured.", e);
        }
    }

    // Example 2: With custom configuration
    println!("Example 2: Custom Configuration");
    println!("{}", "-".repeat(50));

    let bedrock_custom = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Sonnet)
        .with_region("us-west-2")
        .with_temperature(0.3) // Lower temperature for more deterministic results
        .with_max_tokens(150);

    let messages = vec![Message::new_human_message(
        "Explain quantum computing in one sentence.",
    )];

    match bedrock_custom.generate(&messages).await {
        Ok(result) => {
            println!("Question: Explain quantum computing in one sentence.");
            println!("Response: {}\n", result.generation);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    // Example 3: Using different models
    println!("Example 3: Different Models");
    println!("{}", "-".repeat(50));

    let models = vec![
        ("Claude 3 Sonnet", BedrockModel::AnthropicClaude3Sonnet),
        ("Claude 3 Haiku", BedrockModel::AnthropicClaude3Haiku),
        ("Claude 3 Opus", BedrockModel::AnthropicClaude3Opus),
        ("Titan Text Express", BedrockModel::AmazonTitanTextExpress),
    ];

    let test_message = "What is machine learning?";
    let messages = vec![Message::new_human_message(test_message)];

    for (model_name, model) in models {
        let bedrock = Bedrock::default()
            .with_model(model)
            .with_max_tokens(80);

        match bedrock.generate(&messages).await {
            Ok(result) => {
                println!("Model: {}", model_name);
                println!("Response: {}\n", result.generation);
            }
            Err(e) => {
                println!(
                    "Model: {} - Skipped ({})\n",
                    model_name, e
                );
            }
        }
    }

    // Example 4: System and user messages
    println!("Example 4: System + User Messages");
    println!("{}", "-".repeat(50));

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Sonnet)
        .with_max_tokens(100);

    let messages = vec![
        Message::new_system_message(
            "You are a helpful assistant that explains concepts simply.",
        ),
        Message::new_human_message("What is photosynthesis?"),
    ];

    match bedrock.generate(&messages).await {
        Ok(result) => {
            println!("Question: What is photosynthesis?");
            println!("Response: {}\n", result.generation);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    // Example 5: Custom model ID
    println!("Example 5: Custom Model ID");
    println!("{}", "-".repeat(50));

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::Custom(
            "anthropic.claude-3-sonnet-20240229-v1:0".to_string(),
        ))
        .with_max_tokens(80);

    let messages = vec![Message::new_human_message("Say hello!")];

    match bedrock.generate(&messages).await {
        Ok(result) => {
            println!("Using custom model: anthropic.claude-3-sonnet-20240229-v1:0");
            println!("Response: {}\n", result.generation);
        }
        Err(e) => {
            println!(
                "Custom model error (this is expected for demonstration): {}\n",
                e
            );
        }
    }

    println!("=== Examples Complete ===");
    Ok(())
}