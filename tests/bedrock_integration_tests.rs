//! Integration tests for AWS Bedrock LLM
//!
//! These tests require:
//! - Valid AWS credentials configured
//! - Access to AWS Bedrock models
//! - Internet connectivity
//!
//! Run with: cargo test --features aws-sdk-bedrockruntime --test bedrock_integration_tests -- --ignored
//! (Tests are ignored by default to avoid requiring AWS credentials in CI)

use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;
use langchain_rust::schemas::Message;

/// Helper function to check if AWS credentials are available
fn has_aws_credentials() -> bool {
    std::env::var("AWS_ACCESS_KEY_ID").is_ok()
        || std::env::var("AWS_PROFILE").is_ok()
        || std::path::Path::new(&format!(
            "{}/.aws/credentials",
            std::env::var("HOME").unwrap_or_default()
        ))
        .exists()
}

#[tokio::test]
#[ignore] // Ignore by default - requires AWS credentials
async fn test_bedrock_claude_generate() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Sonnet)
        .with_max_tokens(100);

    let messages = vec![Message::new_human_message(
        "What is 2+2? Answer with just the number.",
    )];

    let result = bedrock.generate(&messages).await;

    assert!(result.is_ok(), "Failed to generate: {:?}", result.err());

    let response = result.unwrap();
    assert!(!response.generation.is_empty(), "Response should not be empty");
    println!("Response: {}", response.generation);
}

#[tokio::test]
#[ignore]
async fn test_bedrock_with_temperature() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Sonnet)
        .with_temperature(0.3)
        .with_max_tokens(50);

    let messages = vec![Message::new_human_message(
        "Say 'Hello, World!' and nothing else.",
    )];

    let result = bedrock.generate(&messages).await;

    assert!(result.is_ok(), "Failed to generate: {:?}", result.err());

    let response = result.unwrap();
    assert!(
        response.generation.to_lowercase().contains("hello"),
        "Response should contain 'hello': {}",
        response.generation
    );
}

#[tokio::test]
#[ignore]
async fn test_bedrock_haiku_model() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Haiku)
        .with_max_tokens(100);

    let messages = vec![Message::new_human_message("What is machine learning?")];

    let result = bedrock.generate(&messages).await;

    assert!(result.is_ok(), "Failed to generate with Haiku: {:?}", result.err());

    let response = result.unwrap();
    assert!(!response.generation.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_bedrock_with_region() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Sonnet)
        .with_region("us-east-1")
        .with_max_tokens(50);

    let messages = vec![Message::new_human_message("Say 'Hello'")];

    let result = bedrock.generate(&messages).await;

    assert!(result.is_ok(), "Failed to generate in us-east-1: {:?}", result.err());
    assert!(!result.unwrap().generation.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_bedrock_custom_model_id() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    // Use custom model ID (Claude 3 Sonnet with explicit version)
    let bedrock = Bedrock::default()
        .with_model(BedrockModel::Custom(
            "anthropic.claude-3-sonnet-20240229-v1:0".to_string(),
        ))
        .with_max_tokens(50);

    let messages = vec![Message::new_human_message("Hello, how are you?")];

    let result = bedrock.generate(&messages).await;

    assert!(result.is_ok(), "Failed to generate with custom model ID: {:?}", result.err());

    let response = result.unwrap();
    assert!(!response.generation.is_empty(), "Response should not be empty");
}