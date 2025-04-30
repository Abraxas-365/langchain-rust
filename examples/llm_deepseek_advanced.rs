use langchain_rust::{
    language_models::{llm::LLM, options::CallOptions},
    llm::{Deepseek, DeepseekModel},
    schemas::Message,
};
use std::{env, io::Write};

#[tokio::main]
async fn main() {
    // Get API key from environment variable
    let api_key =
        env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY environment variable must be set");

    // Example 1: Basic generation with options
    println!("=== Example 1: Basic Generation with Options ===");
    let deepseek = Deepseek::new()
        .with_api_key(api_key.clone())
        .with_model(DeepseekModel::DeepseekChat.to_string())
        .with_options(
            CallOptions::default()
                .with_max_tokens(500)
                .with_temperature(1.3) // Recommended for general conversation
                .with_top_p(0.9),
        );

    // Create a system and user message
    let messages = vec![
        Message::new_system_message("You are a helpful AI assistant who responds in Chinese."),
        Message::new_human_message(
            "What are the three most popular programming languages in 2023?",
        ),
    ];

    let response = deepseek.generate(&messages).await.unwrap();
    println!("Response: {}", response.generation);
    println!("Tokens used: {:?}", response.tokens);
    println!("\n");

    // Example 2: Streaming response
    println!("=== Example 2: Streaming Response ===");

    // Create a streaming callback function
    let callback = |content: String| {
        print!("{}", content);
        let _ = std::io::stdout().flush();
        async { Ok(()) }
    };

    let streaming_options = CallOptions::default()
        .with_max_tokens(100)
        .with_streaming_func(callback);

    let streaming_deepseek = Deepseek::new()
        .with_api_key(api_key.clone())
        .with_model(DeepseekModel::DeepseekChat.to_string())
        .with_options(streaming_options);

    let stream_messages = vec![Message::new_human_message(
        "Write a short poem about artificial intelligence.",
    )];

    println!("Streaming response:");
    let streaming_response = streaming_deepseek.generate(&stream_messages).await.unwrap();
    println!(
        "\n\nDone streaming. Total tokens: {:?}",
        streaming_response.tokens
    );

    // Example 3: Using the Reasoning Model
    println!("\n=== Example 3: Using DeepseekReasoner with Chain of Thought ===");

    let reasoning_deepseek = Deepseek::new()
        .with_api_key(api_key.clone())
        .with_model(DeepseekModel::DeepseekReasoner.to_string())
        .with_include_reasoning(true); // This enables the reasoning content to be included

    let reasoning_messages = vec![Message::new_human_message(
        "If I have 15 apples and give 2/5 of them to my friend, how many apples do I have left?",
    )];

    let reasoning_response = reasoning_deepseek
        .generate(&reasoning_messages)
        .await
        .unwrap();
    println!("{}", reasoning_response.generation);

    // Example 4: JSON mode
    println!("\n=== Example 4: Using JSON Mode ===");

    let json_deepseek = Deepseek::new()
        .with_api_key(api_key)
        .with_model(DeepseekModel::DeepseekChat.to_string())
        .with_json_mode(true);

    let json_messages = vec![
        Message::new_system_message("You are a helpful assistant that outputs JSON."),
        Message::new_human_message(
            "List the top 3 programming languages with their primary use cases as a JSON array.",
        ),
    ];

    let json_response = json_deepseek.generate(&json_messages).await.unwrap();
    println!("JSON Response:\n{}", json_response.generation);
}
