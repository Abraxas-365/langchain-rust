//! Web search with Exa, alone and inside a tools agent.
//!
//! `EXA_API_KEY=... OPENAI_API_KEY=... cargo run --example exa_search`

use std::{error::Error, sync::Arc};

use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    llm::openai::{OpenAI, OpenAIModel},
    prompt_args,
    tools::{ExaContents, ExaSearch},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let search = ExaSearch::new()
        .with_num_results(3)?
        .with_start_published_date("2025-01-01")?
        .with_contents(ExaContents::Highlights {
            max_characters: Some(500),
        })?;

    // Typed results, no model involved.
    for result in search.search("Rust release notes").await?.results {
        println!("{} - {}", result.title.unwrap_or_default(), result.url);
    }

    // As a tool: the agent's own model reads the highlights and answers.
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
