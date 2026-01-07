//! Basic usage example for AWS Bedrock LLM

use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Basic Bedrock Usage Example ===\n");

    // Create a Bedrock instance with default settings (Claude v2, us-east-1)
    let bedrock = Bedrock::default();

    println!("Asking: What is the capital of France?");
    let response = bedrock.invoke("What is the capital of France?").await?;
    println!("Response: {}\n", response);

    // Use a different model
    let bedrock_sonnet = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Sonnet);

    println!("Asking: Explain quantum computing in simple terms");
    let response = bedrock_sonnet
        .invoke("Explain quantum computing in simple terms")
        .await?;
    println!("Response: {}\n", response);

    Ok(())
}

// examples/bedrock_advanced_config.rs

//! Advanced configuration example for AWS Bedrock LLM

use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Configuration Example ===\n");

    // Configure all parameters
    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_region("us-west-2")
        .with_temperature(0.3) // Lower temperature for more focused responses
        .with_max_tokens(500)
        .with_top_p(0.9)
        .with_stop_sequence("\n\n");

    let prompt = "Write a haiku about programming";
    println!("Prompt: {}", prompt);

    let response = bedrock.invoke(prompt).await?;
    println!("Response:\n{}\n", response);

    // Example with multiple stop sequences
    let bedrock_with_stops = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_stop_sequence("END")
        .with_stop_sequence("DONE")
        .with_stop_sequence("FINISHED");

    let prompt = "List three programming languages:\n1.";
    println!("Prompt: {}", prompt);

    let response = bedrock_with_stops.invoke(prompt).await?;
    println!("Response:\n{}\n", response);

    Ok(())
}

// examples/bedrock_batch_processing.rs

//! Batch processing example for AWS Bedrock LLM

use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Batch Processing Example ===\n");

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Sonnet)
        .with_temperature(0.5);

    // Process multiple prompts in one call
    let prompts = vec![
        "What is machine learning?".to_string(),
        "What is deep learning?".to_string(),
        "What is a neural network?".to_string(),
        "What is reinforcement learning?".to_string(),
    ];

    println!("Processing {} prompts...\n", prompts.len());

    let result = bedrock.generate(&prompts).await?;

    for (i, generation) in result.generations.iter().enumerate() {
        println!("--- Prompt {}: {} ---", i + 1, prompts[i]);
        println!("Response: {}\n", generation[0]);
    }

    Ok(())
}

// examples/bedrock_different_models.rs

//! Example showing different Bedrock models

use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Different Models Example ===\n");

    let prompt = "What is artificial intelligence?";

    // Anthropic Claude V2
    println!("--- Using Anthropic Claude V2 ---");
    let claude_v2 = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2);
    let response = claude_v2.invoke(prompt).await?;
    println!("Response: {}\n", response);

    // Anthropic Claude 3 Haiku (faster, cheaper)
    println!("--- Using Anthropic Claude 3 Haiku ---");
    let claude_haiku = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Haiku);
    let response = claude_haiku.invoke(prompt).await?;
    println!("Response: {}\n", response);

    // Amazon Titan Text Express
    println!("--- Using Amazon Titan Text Express ---");
    let titan = Bedrock::default()
        .with_model(BedrockModel::AmazonTitanTextExpress);
    let response = titan.invoke(prompt).await?;
    println!("Response: {}\n", response);

    // Custom model ID
    println!("--- Using Custom Model ID ---");
    let custom = Bedrock::default()
        .with_model(BedrockModel::Custom("anthropic.claude-v2:1".to_string()));
    let response = custom.invoke(prompt).await?;
    println!("Response: {}\n", response);

    Ok(())
}

// examples/bedrock_error_handling.rs

//! Error handling example for AWS Bedrock LLM

use langchain_rust::llm::bedrock::{Bedrock, BedrockError, BedrockModel};
use langchain_rust::language_models::llm::LLM;

#[tokio::main]
async fn main() {
    println!("=== Error Handling Example ===\n");

    // Example 1: Handle successful invocation
    let bedrock = Bedrock::default();
    match bedrock.invoke("What is Rust?").await {
        Ok(response) => println!("Success: {}\n", response),
        Err(e) => eprintln!("Error: {}\n", e),
    }

    // Example 2: Handle invalid region
    println!("--- Testing with potentially invalid configuration ---");
    let bedrock_invalid = Bedrock::default()
        .with_model(BedrockModel::Custom("invalid-model-id".to_string()))
        .with_region("invalid-region");

    match bedrock_invalid.invoke("Test").await {
        Ok(response) => println!("Response: {}", response),
        Err(e) => {
            // Downcast to BedrockError for specific handling
            if let Some(bedrock_err) = e.downcast_ref::<BedrockError>() {
                match bedrock_err {
                    BedrockError::AwsError(msg) => {
                        eprintln!("AWS Service Error: {}", msg);
                    }
                    BedrockError::InvalidModel(msg) => {
                        eprintln!("Invalid Model Configuration: {}", msg);
                    }
                    BedrockError::InvocationError(msg) => {
                        eprintln!("Model Invocation Failed: {}", msg);
                    }
                    BedrockError::InvalidRegion(msg) => {
                        eprintln!("Invalid AWS Region: {}", msg);
                    }
                    BedrockError::SerdeError(e) => {
                        eprintln!("JSON Serialization Error: {}", e);
                    }
                }
            } else {
                eprintln!("Unknown error type: {}", e);
            }
        }
    }

    // Example 3: Retry logic
    println!("\n--- Testing with retry logic ---");
    let max_retries = 3;
    let bedrock = Bedrock::default();

    for attempt in 1..=max_retries {
        println!("Attempt {}/{}", attempt, max_retries);

        match bedrock.invoke("What is the meaning of life?").await {
            Ok(response) => {
                println!("Success on attempt {}: {}\n", attempt, response);
                break;
            }
            Err(e) => {
                eprintln!("Attempt {} failed: {}", attempt, e);
                if attempt == max_retries {
                    eprintln!("All retries exhausted");
                } else {
                    println!("Retrying...\n");
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
    }
}

// examples/bedrock_streaming.rs

//! Streaming example (conceptual - streaming support would need to be added)

use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Streaming Example (Future Enhancement) ===\n");

    // Note: This example shows the desired API for streaming
    // Actual streaming implementation would require additional methods
    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2);

    println!("For streaming support, you would typically:");
    println!("1. Use invoke_with_response_stream endpoint");
    println!("2. Process chunks as they arrive");
    println!("3. Handle partial responses");

    // For now, use regular invocation
    let response = bedrock
        .invoke("Write a short story about a robot learning to paint")
        .await?;

    println!("\nComplete Response:\n{}", response);

    Ok(())
}

// examples/bedrock_chain_integration.rs

//! Example showing integration with LangChain chains

use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    fmt_message, fmt_placeholder, fmt_template,
    llm::bedrock::{Bedrock, BedrockModel},
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    template_fstring,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Chain Integration Example ===\n");

    // Create Bedrock LLM
    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_temperature(0.5);

    // Create a simple prompt template
    let prompt = HumanMessagePromptTemplate::new(template_fstring!(
        "What is the capital of {country}?",
        "country"
    ));

    // Build the chain
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(bedrock)
        .build()?;

    // Use the chain with different countries
    let countries = vec!["France", "Japan", "Brazil", "Egypt"];

    for country in countries {
        let input = prompt_args! {
            "country" => country,
        };

        println!("Country: {}", country);
        match chain.invoke(input).await {
            Ok(result) => println!("Capital: {:?}\n", result),
            Err(e) => eprintln!("Error: {}\n", e),
        }
    }

    Ok(())
}

// examples/bedrock_conversational.rs

//! Example showing conversational usage pattern

use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Conversational Example ===\n");

    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_temperature(0.7);

    // Simulate a conversation by building context
    let mut conversation_history = String::new();

    // First message
    let user_msg_1 = "Hello! My name is Alice and I love programming.";
    conversation_history.push_str(&format!("\n\nHuman: {}\n\nAssistant:", user_msg_1));

    let response_1 = bedrock.invoke(&conversation_history).await?;
    println!("User: {}", user_msg_1);
    println!("Assistant: {}\n", response_1);

    conversation_history.push_str(&response_1);

    // Second message
    let user_msg_2 = "What's my name and what do I love?";
    conversation_history.push_str(&format!("\n\nHuman: {}\n\nAssistant:", user_msg_2));

    let response_2 = bedrock.invoke(&conversation_history).await?;
    println!("User: {}", user_msg_2);
    println!("Assistant: {}\n", response_2);

    Ok(())
}