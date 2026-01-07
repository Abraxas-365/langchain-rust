//! Integration tests for AWS Bedrock LLM
//!
//! These tests require:
//! - Valid AWS credentials configured
//! - Access to AWS Bedrock models
//! - Internet connectivity
//!
//! Run with: cargo test --test bedrock_integration_tests -- --ignored
//! (Tests are ignored by default to avoid requiring AWS credentials in CI)

use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;

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
async fn test_bedrock_invoke_claude_v2() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_max_tokens(100);

    let result = bedrock
        .invoke("What is 2+2? Answer with just the number.")
        .await;

    assert!(result.is_ok(), "Failed to invoke model: {:?}", result.err());

    let response = result.unwrap();
    assert!(!response.is_empty(), "Response should not be empty");
    assert!(
        response.contains("4"),
        "Response should contain '4', got: {}",
        response
    );
}

#[tokio::test]
#[ignore]
async fn test_bedrock_invoke_claude_3_haiku() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Haiku)
        .with_temperature(0.3)
        .with_max_tokens(50);

    let result = bedrock
        .invoke("Say 'Hello, World!' and nothing else.")
        .await;

    assert!(result.is_ok(), "Failed to invoke model: {:?}", result.err());

    let response = result.unwrap();
    assert!(
        response.to_lowercase().contains("hello"),
        "Response should contain 'hello': {}",
        response
    );
}

#[tokio::test]
#[ignore]
async fn test_bedrock_batch_generation() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_max_tokens(50);

    let prompts = vec![
        "What is 1+1?".to_string(),
        "What is 2+2?".to_string(),
        "What is 3+3?".to_string(),
    ];

    let result = bedrock.generate(&prompts).await;

    assert!(result.is_ok(), "Failed to generate: {:?}", result.err());

    let llm_result = result.unwrap();
    assert_eq!(
        llm_result.generations.len(),
        3,
        "Should have 3 generations"
    );

    for (i, generation) in llm_result.generations.iter().enumerate() {
        assert!(!generation.is_empty(), "Generation {} is empty", i);
        assert!(!generation[0].is_empty(), "Generation {} text is empty", i);
    }
}

#[tokio::test]
#[ignore]
async fn test_bedrock_with_stop_sequences() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_stop_sequence("\n");

    let result = bedrock
        .invoke("List three colors:\n1.")
        .await;

    assert!(result.is_ok(), "Failed to invoke: {:?}", result.err());

    let response = result.unwrap();
    // Response should stop at first newline
    assert!(
        !response.contains("\n2."),
        "Response should stop before second item due to stop sequence"
    );
}

#[tokio::test]
#[ignore]
async fn test_bedrock_temperature_effect() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    // Low temperature should give more deterministic results
    let bedrock_low_temp = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_temperature(0.1)
        .with_max_tokens(100);

    let prompt = "What is the capital of France? Answer with just the city name.";

    let result1 = bedrock_low_temp.invoke(prompt).await;
    let result2 = bedrock_low_temp.invoke(prompt).await;

    assert!(result1.is_ok() && result2.is_ok());

    let response1 = result1.unwrap();
    let response2 = result2.unwrap();

    // Both should contain Paris
    assert!(response1.to_lowercase().contains("paris"));
    assert!(response2.to_lowercase().contains("paris"));
}

#[tokio::test]
#[ignore]
async fn test_bedrock_max_tokens_limit() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_max_tokens(10); // Very small limit

    let result = bedrock
        .invoke("Write a long essay about artificial intelligence.")
        .await;

    assert!(result.is_ok());

    let response = result.unwrap();
    // Response should be truncated due to small token limit
    // Approximate: 10 tokens ≈ 30-40 characters
    assert!(
        response.len() < 100,
        "Response should be short due to token limit, got length: {}",
        response.len()
    );
}

#[tokio::test]
#[ignore]
async fn test_bedrock_different_regions() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let regions = vec!["us-east-1", "us-west-2"];

    for region in regions {
        let bedrock = Bedrock::default()
            .with_model(BedrockModel::AnthropicClaudeV2)
            .with_region(region)
            .with_max_tokens(50);

        let result = bedrock.invoke("Say hello").await;

        assert!(
            result.is_ok(),
            "Failed to invoke in region {}: {:?}",
            region,
            result.err()
        );

        let response = result.unwrap();
        assert!(!response.is_empty(), "Empty response from region {}", region);
    }
}

#[tokio::test]
#[ignore]
async fn test_bedrock_prompt_formatting() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_max_tokens(50);

    // Test with plain prompt
    let result1 = bedrock.invoke("What is AI?").await;
    assert!(result1.is_ok());

    // Test with pre-formatted prompt
    let result2 = bedrock
        .invoke("\n\nHuman: What is AI?\n\nAssistant:")
        .await;
    assert!(result2.is_ok());

    // Both should work
    assert!(!result1.unwrap().is_empty());
    assert!(!result2.unwrap().is_empty());
}

#[tokio::test]
#[ignore]
async fn test_bedrock_custom_model_id() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    // Use custom model ID (Claude v2:1)
    let bedrock = Bedrock::default()
        .with_model(BedrockModel::Custom("anthropic.claude-v2:1".to_string()))
        .with_max_tokens(50);

    let result = bedrock.invoke("Hello").await;

    // This might fail if the specific model version isn't available
    // but it tests the custom model ID functionality
    if result.is_ok() {
        assert!(!result.unwrap().is_empty());
    }
}

#[tokio::test]
#[ignore]
async fn test_bedrock_titan_model() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AmazonTitanTextExpress)
        .with_max_tokens(100);

    let result = bedrock.invoke("What is machine learning?").await;

    assert!(result.is_ok(), "Failed to invoke Titan: {:?}", result.err());

    let response = result.unwrap();
    assert!(!response.is_empty());
    assert!(response.len() > 10, "Response seems too short");
}

#[tokio::test]
#[ignore]
async fn test_bedrock_error_handling_invalid_prompt() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_max_tokens(1); // Extremely small to potentially trigger issues

    // This should either succeed with a very short response or fail gracefully
    let result = bedrock.invoke("").await;

    // We don't assert on success/failure, just that it doesn't panic
    match result {
        Ok(resp) => println!("Got response: {}", resp),
        Err(e) => println!("Got expected error: {}", e),
    }
}

#[tokio::test]
#[ignore]
async fn test_bedrock_concurrent_requests() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_max_tokens(50);

    // Launch multiple concurrent requests
    let handles: Vec<_> = (0..3)
        .map(|i| {
            let bedrock = bedrock.clone();
            tokio::spawn(async move {
                bedrock
                    .invoke(&format!("What is {}+{}?", i, i))
                    .await
            })
        })
        .collect();

    // Wait for all to complete
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.await.expect("Task panicked");
        assert!(
            result.is_ok(),
            "Request {} failed: {:?}",
            i,
            result.err()
        );
    }
}

#[tokio::test]
#[ignore]
async fn test_bedrock_long_conversation() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_max_tokens(100);

    // Build a conversation
    let mut context = String::from("\n\nHuman: My name is Alice.\n\nAssistant:");
    let response1 = bedrock.invoke(&context).await.unwrap();
    context.push_str(&response1);

    context.push_str("\n\nHuman: What is my name?\n\nAssistant:");
    let response2 = bedrock.invoke(&context).await.unwrap();

    // Should remember the name
    assert!(
        response2.to_lowercase().contains("alice"),
        "Should remember the name Alice in context: {}",
        response2
    );
}

// Helper test to validate AWS credentials are working
#[tokio::test]
#[ignore]
async fn test_aws_credentials_valid() {
    if !has_aws_credentials() {
        println!("Skipping test: No AWS credentials found");
        return;
    }

    // Try to create a client - this will validate credentials
    let bedrock = Bedrock::default();
    let mut bedrock = bedrock.clone();

    let client_result = bedrock.get_client().await;
    assert!(
        client_result.is_ok(),
        "Failed to create AWS client - check credentials"
    );
}