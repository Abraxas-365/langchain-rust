//! Trace an agent run with `tracing`: one `agent.run` span with an `agent.plan` span per
//! step, `llm.generate` spans with model and token usage, and `tool.call` spans.
//!
//! ```sh
//! OPENAI_API_KEY=... RUST_LOG=langchain_rust=info cargo run --example tracing
//! ```
//!
//! Swap the fmt subscriber for `tracing-opentelemetry` to ship the same spans to Jaeger,
//! Honeycomb, Langfuse, etc.
use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    llm::openai::OpenAI,
    prompt_args,
    tools::Tool,
};
use serde_json::Value;
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

struct Weather;

#[async_trait]
impl Tool for Weather {
    fn name(&self) -> String {
        "weather".into()
    }
    fn description(&self) -> String {
        "Current weather for a city".into()
    }
    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        Ok(format!("Sunny, 22°C in {}", input.as_str().unwrap_or("?")))
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("langchain_rust=info")),
        )
        // Print each span with its fields and duration when it closes.
        .with_span_events(FmtSpan::CLOSE)
        .init();

    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(Weather)];
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&tools)
        .build(OpenAI::default())
        .unwrap();
    let executor = AgentExecutor::from_agent(agent);

    let answer = executor
        .invoke(prompt_args! {"input" => "What's the weather in Lima?"})
        .await
        .unwrap();
    println!("{answer}");
}
