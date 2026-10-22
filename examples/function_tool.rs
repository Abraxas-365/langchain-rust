// Define agent tools from plain async functions with `FunctionTool`, no struct or trait
// impl needed.
//
// OpenAI: OPENAI_API_KEY=... cargo run --example function_tool
// Local Ollama: TOOLS_OLLAMA_MODEL=qwen3:14b cargo run --example function_tool
use std::sync::Arc;

use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    llm::{openai::OpenAI, OpenAIConfig},
    prompt_args,
    tools::{FunctionTool, Tool},
};
use serde::Deserialize;
use serde_json::json;

/// The arguments the model sends are deserialized into this.
#[derive(Deserialize)]
struct AddInput {
    a: f64,
    b: f64,
}

#[tokio::main]
async fn main() {
    let adder = FunctionTool::new(
        "adder",
        "Adds two numbers together",
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "number", "description": "First number" },
                "b": { "type": "number", "description": "Second number" }
            },
            "required": ["a", "b"]
        }),
        |input: AddInput| async move {
            println!("  [adder] {} + {}", input.a, input.b);
            Ok::<_, String>(format!(
                "The sum of {} and {} is {}",
                input.a,
                input.b,
                input.a + input.b
            ))
        },
    );

    // A tool that takes one string; the schema is filled in for you.
    let word_count = FunctionTool::text(
        "word_count",
        "Counts the words in a text",
        |text: String| async move {
            println!("  [word_count] {text:?}");
            Ok::<_, String>(text.split_whitespace().count())
        },
    );

    // A tool can be called directly too.
    println!(
        "direct call: {}",
        adder.call(r#"{"a": 2, "b": 3}"#).await.unwrap()
    );

    let llm = match std::env::var("TOOLS_OLLAMA_MODEL") {
        Ok(model) => OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model),
        Err(_) => OpenAI::default(),
    };
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(adder), Arc::new(word_count)];
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&tools)
        .build(llm)
        .unwrap();
    let executor = AgentExecutor::from_agent(agent).with_max_iterations(5);

    let answer = executor
        .invoke(prompt_args! {
            "input" => "How many words are in 'the quick brown fox jumps', and what is that number plus 1234.5? Use the tools.",
        })
        .await
        .unwrap();
    println!("agent: {answer}");
}
