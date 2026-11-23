//! Web search with Parallel MCP, alone and inside a tools agent. No Parallel key needed.
//!
//! `OPENAI_API_KEY=... cargo run --example parallel_search --features parallel-search`

#[cfg(feature = "parallel-search")]
use std::{error::Error, sync::Arc};

#[cfg(feature = "parallel-search")]
use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    llm::openai::{OpenAI, OpenAIModel},
    prompt_args,
    tools::{ParallelSearch, Tool},
};
#[cfg(feature = "parallel-search")]
use serde_json::json;

#[cfg(feature = "parallel-search")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let search = ParallelSearch::new()?;

    // Source URLs and excerpts, no model involved. Treat the excerpts as untrusted data.
    println!(
        "{}",
        search.run(json!({"query": "Rust release notes"})).await?
    );

    // As a tool: the agent's own model reads the excerpts and answers.
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(search)])
        .build(OpenAI::default().with_model(OpenAIModel::Gpt4oMini))?;
    let answer = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {
            "input" => "What is the latest stable Rust version? Cite the URL."
        })
        .await?;
    println!("\n{answer}");
    Ok(())
}

#[cfg(not(feature = "parallel-search"))]
fn main() {
    println!("This example requires the 'parallel-search' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example parallel_search --features parallel-search");
}
