# AWS Bedrock LLM Integration

This module provides integration with AWS Bedrock's foundation models for the langchain-rust project.

## Overview

AWS Bedrock is a fully managed service that makes foundation models from leading AI companies available via an API. You can choose from a wide range of models to find the one that best suits your use case.

## Installation

Add the following dependencies to your `Cargo.toml`:

```toml
[dependencies]
langchain-rust = "1.0"
aws-config = "1.0"
aws-sdk-bedrockruntime = "1.0"
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
async-trait = "0.1"
```

## Supported Models

### Anthropic Claude
- `BedrockModel::AnthropicClaudeV2` - Claude v2
- `BedrockModel::AnthropicClaudeInstantV1` - Claude Instant v1
- `BedrockModel::AnthropicClaude3Sonnet` - Claude 3 Sonnet
- `BedrockModel::AnthropicClaude3Haiku` - Claude 3 Haiku
- `BedrockModel::AnthropicClaude3Opus` - Claude 3 Opus

### AI21 Labs
- `BedrockModel::AI21Jurassic2Mid` - Jurassic-2 Mid
- `BedrockModel::AI21Jurassic2Ultra` - Jurassic-2 Ultra

### Amazon Titan
- `BedrockModel::AmazonTitanTextExpress` - Titan Text Express
- `BedrockModel::AmazonTitanTextLite` - Titan Text Lite

### Cohere
- `BedrockModel::CohereCommand` - Command
- `BedrockModel::CohereCommandLight` - Command Light

### Meta
- `BedrockModel::MetaLlama2Chat13B` - Llama 2 Chat 13B
- `BedrockModel::MetaLlama2Chat70B` - Llama 2 Chat 70B

### Custom Models
- `BedrockModel::Custom(String)` - Use any custom model ID

## Prerequisites

### AWS Credentials

You need AWS credentials configured on your system. The SDK will automatically look for credentials in the following order:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM role (when running on EC2, ECS, or Lambda)

### AWS Bedrock Access

Ensure you have:
1. Enabled AWS Bedrock in your AWS account
2. Requested access to the models you want to use
3. Proper IAM permissions for `bedrock:InvokeModel`

## Basic Usage

### Simple Example

```rust
use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a Bedrock instance with default settings
    let bedrock = Bedrock::default();
    
    // Invoke the model
    let response = bedrock.invoke("What is the capital of France?").await?;
    println!("Response: {}", response);
    
    Ok(())
}
```

### Custom Configuration

```rust
use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_region("us-west-2")
        .with_temperature(0.7)
        .with_max_tokens(1000)
        .with_top_p(0.9)
        .with_stop_sequence("\n\n");
    
    let response = bedrock.invoke("Explain quantum computing").await?;
    println!("Response: {}", response);
    
    Ok(())
}
```

### Batch Generation

```rust
use langchain_rust::llm::bedrock::{Bedrock, BedrockModel};
use langchain_rust::language_models::llm::LLM;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaude3Sonnet);
    
    let prompts = vec![
        "What is machine learning?".to_string(),
        "What is deep learning?".to_string(),
        "What is neural network?".to_string(),
    ];
    
    let result = bedrock.generate(&prompts).await?;
    
    for (i, generation) in result.generations.iter().enumerate() {
        println!("Response {}: {}", i + 1, generation[0]);
    }
    
    Ok(())
}
```

### Using with Chains

```rust
use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    llm::bedrock::{Bedrock, BedrockModel},
    prompt::HumanMessagePromptTemplate,
    template_fstring,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bedrock = Bedrock::default()
        .with_model(BedrockModel::AnthropicClaudeV2)
        .with_temperature(0.5);
    
    let prompt = HumanMessagePromptTemplate::new(
        template_fstring!("What is the capital of {country}?", "country")
    );
    
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(bedrock)
        .build()?;
    
    let input_variables = prompt_args! {
        "country" => "Japan",
    };
    
    let result = chain.invoke(input_variables).await?;
    println!("Result: {:?}", result);
    
    Ok(())
}
```

## Configuration Options

### Model Selection

```rust
// Use predefined models
let bedrock = Bedrock::default()
    .with_model(BedrockModel::AnthropicClaudeV2);

// Use custom model ID
let bedrock = Bedrock::default()
    .with_model(BedrockModel::Custom("my-custom-model-id".to_string()));
```

### Region Configuration

```rust
// Set specific region
let bedrock = Bedrock::default()
    .with_region("us-east-1");

// AWS SDK will also respect AWS_REGION environment variable
```

### Generation Parameters

```rust
let bedrock = Bedrock::default()
    .with_temperature(0.7)      // Controls randomness (0.0 to 1.0)
    .with_max_tokens(1000)      // Maximum tokens to generate
    .with_top_p(0.9)            // Nucleus sampling parameter
    .with_top_k(50)             // Top-k sampling parameter
    .with_stop_sequence("END"); // Stop generation at this sequence
```

### Multiple Stop Sequences

```rust
let bedrock = Bedrock::default()
    .with_stop_sequence("\n\n")
    .with_stop_sequence("END")
    .with_stop_sequence("STOP");
```

## Model-Specific Considerations

### Anthropic Claude Models

Claude models require specific prompt formatting:
- Prompts should be formatted as `\n\nHuman: {prompt}\n\nAssistant:`
- The Bedrock client automatically handles this formatting
- If you provide a pre-formatted prompt, it will be preserved

```rust
// Both of these work:
let response1 = bedrock.invoke("What is AI?").await?;
let response2 = bedrock.invoke("\n\nHuman: What is AI?\n\nAssistant:").await?;
```

### Amazon Titan Models

Titan models use a different parameter structure:
- `inputText` instead of `prompt`
- Parameters are nested under `textGenerationConfig`

### AI21 Jurassic Models

Jurassic models support:
- `maxTokens` instead of `max_tokens_to_sample`
- Standard temperature and topP parameters

## Error Handling

The module provides comprehensive error handling:

```rust
use langchain_rust::llm::bedrock::{Bedrock, BedrockError};

match bedrock.invoke("Test prompt").await {
    Ok(response) => println!("Success: {}", response),
    Err(e) => match e.downcast_ref::<BedrockError>() {
        Some(BedrockError::AwsError(msg)) => eprintln!("AWS Error: {}", msg),
        Some(BedrockError::InvalidModel(msg)) => eprintln!("Invalid Model: {}", msg),
        Some(BedrockError::InvocationError(msg)) => eprintln!("Invocation Error: {}", msg),
        _ => eprintln!("Unknown error: {}", e),
    }
}
```

## Testing

The module includes comprehensive unit tests:

```bash
cargo test --package langchain-rust --lib llm::bedrock
```

### Test Coverage

- Model ID generation
- Provider identification
- Configuration builder pattern
- Prompt formatting for different providers
- Request body construction
- Response parsing
- Error handling

## Performance Considerations

### Latency

- First invocation initializes the AWS client (adds latency)
- Subsequent calls reuse the client (faster)
- Consider connection pooling for high-throughput scenarios

### Cost Optimization

- Use appropriate token limits to control costs
- Choose model sizes appropriate for your task
- Consider using Claude Instant or Titan Lite for simple tasks

## Security Best Practices

1. **Never hardcode credentials** - Use AWS credential chain
2. **Use IAM roles** when running on AWS infrastructure
3. **Apply least privilege** - Grant only necessary Bedrock permissions
4. **Monitor usage** - Set up CloudWatch alerts for unusual activity
5. **Rotate credentials** regularly

## Troubleshooting

### "Model not found" error

Ensure you've requested access to the model in AWS Bedrock console:
1. Go to AWS Bedrock console
2. Navigate to "Model access"
3. Request access for the models you need

### Authentication errors

Check your AWS credentials:
```bash
aws sts get-caller-identity
```

### Region issues

Verify Bedrock is available in your region:
- Not all AWS regions support Bedrock
- Common regions: us-east-1, us-west-2, eu-west-1

## Examples

See the `examples/` directory for more usage examples:

- `basic_usage.rs` - Simple prompt and response
- `advanced_config.rs` - All configuration options
- `batch_processing.rs` - Processing multiple prompts
- `chain_integration.rs` - Using with LangChain chains
- `error_handling.rs` - Comprehensive error handling

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass: `cargo test`
2. Code is formatted: `cargo fmt`
3. No clippy warnings: `cargo clippy`
4. Documentation is updated

## License

This module is part of langchain-rust and follows the same license.

## References

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS SDK for Rust](https://github.com/awslabs/aws-sdk-rust)
- [LangChain Rust](https://github.com/Abraxas-365/langchain-rust)
- [Anthropic Claude](https://www.anthropic.com/claude)
