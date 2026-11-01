//! Web search and page scraping with Firecrawl, alone and inside a tools agent.
//!
//! `FIRECRAWL_API_KEY=... OPENAI_API_KEY=... cargo run --example firecrawl`

use std::{error::Error, sync::Arc};

use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    llm::openai::{OpenAI, OpenAIModel},
    prompt_args,
    tools::{FirecrawlScrape, FirecrawlSearch},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let search = FirecrawlSearch::new().with_limit(3)?;
    let scrape = FirecrawlScrape::new().with_max_chars(8_000);

    // Typed results, no model involved.
    for result in search.search("Rust release notes").await? {
        println!("{} - {}", result.title.unwrap_or_default(), result.url);
    }
    let page = scrape.scrape("https://www.rust-lang.org").await?;
    println!("{:?}", page.metadata.title);

    // As tools: the agent searches, reads the page it picks, and answers.
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(search), Arc::new(scrape)])
        .build(OpenAI::default().with_model(OpenAIModel::Gpt4oMini))?;
    let answer = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {
            "input" => "What is the latest stable Rust version? Read the release post and cite the URL."
        })
        .await?;
    println!("\n{answer}");
    Ok(())
}
