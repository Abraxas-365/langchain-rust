//! Streams what an agent does (answer tokens, tool calls, results) and stops it with
//! Ctrl-C through a `CancellationToken`.
//!
//! cargo run --example streaming_agent
use std::{error::Error, io::Write, sync::Arc};

use async_trait::async_trait;
use futures::StreamExt;
use langchain_rust::{
    agent::{AgentExecutor, AgentStreamEvent, OpenAiToolAgentBuilder},
    language_models::{options::CallOptions, TokenUsage},
    llm::openai::OpenAI,
    memory::SimpleMemory,
    prompt_args,
    tools::Tool,
};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

struct Date;

#[async_trait]
impl Tool for Date {
    fn name(&self) -> String {
        "Date".to_string()
    }
    fn description(&self) -> String {
        "Useful when you need to get today's date, input is a query".to_string()
    }
    async fn run(&self, _input: Value) -> Result<String, Box<dyn Error>> {
        Ok("25 of november of 2025".to_string())
    }
}

#[tokio::main]
async fn main() {
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(Date)])
        .build(OpenAI::default().with_options(CallOptions::new().with_stream_usage(true)))
        .unwrap();

    // Ctrl-C stops the run, even in the middle of an LLM or tool call.
    let token = CancellationToken::new();
    tokio::spawn({
        let token = token.clone();
        async move {
            tokio::signal::ctrl_c().await.ok();
            token.cancel();
        }
    });

    let executor = AgentExecutor::from_agent(agent)
        .with_memory(SimpleMemory::new().into())
        .with_cancellation(token.clone());

    let mut events = executor.stream_events(prompt_args! {
        "input" => "What date is today? Then write a short poem about it.",
    });
    let mut tokens: Option<TokenUsage> = None;
    while let Some(event) = events.next().await {
        match event {
            Ok(AgentStreamEvent::Text(chunk)) => {
                print!("{chunk}");
                std::io::stdout().flush().unwrap();
            }
            Ok(AgentStreamEvent::ToolStart { tool, input }) => println!("[{tool}] <- {input}"),
            Ok(AgentStreamEvent::ToolEnd { tool, output }) => println!("[{tool}] -> {output}"),
            // Per model call; needs `with_stream_usage(true)` when streaming with OpenAI.
            Ok(AgentStreamEvent::Usage(usage)) => {
                tokens.get_or_insert_with(TokenUsage::default).add(&usage)
            }
            Ok(AgentStreamEvent::Finish { .. }) => println!(),
            Ok(_) => {}
            Err(_) if token.is_cancelled() => println!("\n(cancelled)"),
            Err(e) => eprintln!("\nerror: {e}"),
        }
    }
    if let Some(tokens) = tokens {
        println!("tokens used: {}", tokens.total_tokens);
    }
}
