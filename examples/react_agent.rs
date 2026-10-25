//! A ReAct agent: the model picks tools by writing `Action:` / `Action Input:` in plain
//! text, so it works with models that have no native tool calling.
//!
//! Runs on a local Ollama by default: `REACT_OLLAMA_MODEL=qwen3:14b cargo run --example react_agent`
use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use langchain_rust::{
    agent::{AgentExecutor, ParserErrorHandler, ReActAgentBuilder, INTERMEDIATE_STEPS_KEY},
    chain::Chain,
    llm::{ollama::openai::OllamaConfig, openai::OpenAI},
    prompt_args,
    tools::Tool,
};
use serde_json::Value;

struct Multiply;

#[async_trait]
impl Tool for Multiply {
    fn name(&self) -> String {
        "multiply".into()
    }
    fn description(&self) -> String {
        "Multiplies two integers. Input: the two numbers separated by a comma, e.g. `3, 4`.".into()
    }
    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let text = input.as_str().unwrap_or_default().to_string();
        let nums: Vec<i64> = text
            .split(|c: char| !c.is_ascii_digit() && c != '-')
            .filter_map(|n| n.parse().ok())
            .collect();
        match nums.as_slice() {
            [a, b] => Ok((a * b).to_string()),
            _ => Err(format!("expected two integers, got `{text}`").into()),
        }
    }
}

#[tokio::main]
async fn main() {
    let model = std::env::var("REACT_OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:14b".into());
    let llm = OpenAI::new(OllamaConfig::default()).with_model(model);

    let agent = ReActAgentBuilder::new()
        .tools(&[Arc::new(Multiply)])
        .build(llm)
        .unwrap();
    let executor = AgentExecutor::from_agent(agent)
        // Format slips go back to the model instead of failing the run.
        .with_parser_error_handler(ParserErrorHandler::new())
        .with_return_intermediate_steps(true);

    let out = executor
        .execute(prompt_args! { "input" => "What is 1234 times 5678?" })
        .await
        .unwrap();
    for step in out[INTERMEDIATE_STEPS_KEY].as_array().unwrap() {
        println!(
            "{}({}) -> {}",
            step["tool"].as_str().unwrap(),
            step["tool_input"].as_str().unwrap(),
            step["observation"].as_str().unwrap()
        );
    }
    println!("Answer: {}", out["output"].as_str().unwrap());
}
