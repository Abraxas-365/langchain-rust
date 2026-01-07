use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    llm::openai::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    prompt_args,
    tools::{Tool, WikipediaQuery, WikipediaQueryOptions},
};
use serde_json::json;
use std::error::Error;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Wikipedia Tool Examples ===\n");

    // Example 1: Basic usage with default settings
    basic_usage().await?;

    // Example 2: Custom configuration
    custom_configuration().await?;

    // Example 3: Multi-language support
    multi_language().await?;

    // Example 4: Integration with agent
    // agent_integration().await?; // Uncomment if you have OpenAI API key

    Ok(())
}

/// Example 1: Basic usage with default settings
async fn basic_usage() -> Result<(), Box<dyn Error>> {
    println!("--- Example 1: Basic Usage ---");
    
    let wiki = WikipediaQuery::default();
    
    println!("Tool name: {}", wiki.name());
    println!("Tool description: {}\n", wiki.description());
    
    // Query Wikipedia
    let result = wiki.run(json!("Rust programming language")).await?;
    
    println!("Query: 'Rust programming language'");
    println!("Result:\n{}\n", result);
    
    Ok(())
}

/// Example 2: Custom configuration
async fn custom_configuration() -> Result<(), Box<dyn Error>> {
    println!("--- Example 2: Custom Configuration ---");
    
    // Create a Wikipedia tool with custom settings
    let options = WikipediaQueryOptions {
        top_k_results: 2,
        max_doc_content_length: 500,
        lang: "en".to_string(),
    };
    
    let wiki = WikipediaQuery::new(options);
    
    // Or use the builder pattern
    let wiki_builder = WikipediaQuery::default()
        .with_top_k_results(2)
        .with_max_doc_content_length(500);
    
    let result = wiki_builder.run(json!("LangChain")).await?;
    
    println!("Query: 'LangChain' (with 500 char limit)");
    println!("Result:\n{}\n", result);
    
    Ok(())
}

/// Example 3: Multi-language support
async fn multi_language() -> Result<(), Box<dyn Error>> {
    println!("--- Example 3: Multi-language Support ---");
    
    // Query Spanish Wikipedia
    let wiki_es = WikipediaQuery::with_lang("es");
    let result_es = wiki_es.run(json!("Inteligencia artificial")).await?;
    
    println!("Query: 'Inteligencia artificial' (Spanish)");
    println!("Result:\n{}\n", result_es);
    
    // Query French Wikipedia
    let wiki_fr = WikipediaQuery::with_lang("fr");
    let result_fr = wiki_fr.run(json!("Intelligence artificielle")).await?;
    
    println!("Query: 'Intelligence artificielle' (French)");
    println!("Result:\n{}\n", result_fr);
    
    Ok(())
}

/// Example 4: Integration with OpenAI Agent
/// Requires OPENAI_API_KEY environment variable
#[allow(dead_code)]
async fn agent_integration() -> Result<(), Box<dyn Error>> {
    println!("--- Example 4: Agent Integration ---");
    
    // Initialize OpenAI LLM
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt4oMini.to_string());
    
    // Create Wikipedia tool
    let wiki_tool = Arc::new(WikipediaQuery::default());
    
    // Create agent with Wikipedia tool
    let agent = OpenAiToolAgentBuilder::new()
        .tools(vec![wiki_tool])
        .build(llm)?;
    
    // Create executor
    let executor = AgentExecutor::from_agent(agent)
        .with_memory(SimpleMemory::new().into());
    
    // Ask a question that requires Wikipedia
    let input = prompt_args! {
        "input" => "When was Rust programming language first released and who created it?"
    };
    
    println!("Question: When was Rust programming language first released and who created it?");
    
    let result = executor.invoke(input).await?;
    
    println!("Agent response:\n{:?}\n", result);
    
    Ok(())
}

/// Example 5: Error handling
#[allow(dead_code)]
async fn error_handling() -> Result<(), Box<dyn Error>> {
    println!("--- Example 5: Error Handling ---");
    
    let wiki = WikipediaQuery::default();
    
    // Empty query
    match wiki.run(json!("")).await {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error for empty query: {}", e),
    }
    
    // Invalid input format
    match wiki.run(json!(123)).await {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Expected error for invalid input: {}", e),
    }
    
    // Non-existent topic (should handle gracefully)
    match wiki.run(json!("xyzabc123nonexistent999")).await {
        Ok(result) => println!("Graceful handling of non-existent topic: {}", result),
        Err(e) => println!("Error: {}", e),
    }
    
    Ok(())
}