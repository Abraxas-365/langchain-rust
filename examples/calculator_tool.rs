// An agent that does its arithmetic with the Calculator tool instead of guessing.
//
// OpenAI:            OPENAI_API_KEY=... cargo run --example calculator_tool
// Any local model:   TOOLS_OLLAMA_MODEL=qwen3:14b cargo run --example calculator_tool
//                    (talks to Ollama's OpenAI-compatible endpoint on localhost:11434)

use std::sync::Arc;

use futures::StreamExt;
use langchain_rust::{
    agent::{AgentExecutor, AgentStreamEvent, OpenAiToolAgentBuilder},
    llm::{openai::OpenAI, OpenAIConfig},
    prompt_args,
    tools::{evaluate_expression, Calculator},
};

#[tokio::main]
async fn main() {
    // The evaluator can also be used directly.
    println!(
        "direct: 2 ^ 10 - 24 = {}",
        evaluate_expression("2 ^ 10 - 24").unwrap()
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
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(Calculator::new())])
        .build(llm)
        .unwrap();
    let executor = AgentExecutor::from_agent(agent).with_max_iterations(6);

    let mut events = executor.stream_events(prompt_args! {
        "input" => "I invest 12,500 at 4.3% a year, compounded monthly, for 7 years. \
                    How much do I end up with? Round to cents."
    });
    while let Some(event) = events.next().await {
        match event.unwrap() {
            AgentStreamEvent::ToolStart { tool, input } => println!("  [{tool}] {input}"),
            AgentStreamEvent::ToolEnd { output, .. } => println!("  = {output}"),
            AgentStreamEvent::Finish { output } => println!("answer: {output}"),
            _ => {}
        }
    }
}
