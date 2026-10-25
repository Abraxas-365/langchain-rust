//! Web search with the Perplexity Search API, alone and inside a tools agent.
//!
//! `PERPLEXITY_API_KEY=... OPENAI_API_KEY=... cargo run --example perplexity_search`

use std::{error::Error, sync::Arc};

use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    llm::openai::{OpenAI, OpenAIModel},
    prompt_args,
    tools::{PerplexitySearch, SearchRecency},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let search = PerplexitySearch::new()
        .with_max_results(3)?
        .with_max_tokens_per_page(256)?
        .with_recency(SearchRecency::Month);

    // Typed results, no model involved.
    for result in search.search("Rust release notes").await? {
        println!("{} - {}", result.title, result.url);
    }

    // As a tool: the agent's own model reads the sources and answers.
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
