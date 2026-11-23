//! Web search with Keenable, alone and inside a tools agent. No Keenable key needed
//! (set KEENABLE_API_KEY to use the keyed endpoint).
//!
//! `OPENAI_API_KEY=... cargo run --example keenable_search`

use std::{error::Error, sync::Arc};

use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    llm::openai::{OpenAI, OpenAIModel},
    prompt_args,
    tools::KeenableSearch,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let search = KeenableSearch::new().with_max_results(3)?;

    // Typed results, no model involved.
    for result in search.search("Rust release notes").await? {
        println!("{} - {}", result.title.unwrap_or_default(), result.url);
    }

    // As a tool: the agent's own model reads the snippets and answers.
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
