// Caps what an agent run spends in US dollars, with `with_usage_limit` and llm-budget's
// per-model prices.
//
// cargo run --example agent_budget --features budget
#[cfg(feature = "budget")]
use std::sync::Arc;

#[cfg(feature = "budget")]
use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    language_models::TokenUsage,
    llm::openai::{OpenAI, OpenAIModel},
    prompt_args,
    tools::CommandExecutor,
};
#[cfg(feature = "budget")]
use llm_budget::{PriceTable, Usage};

/// Most a single run may spend.
#[cfg(feature = "budget")]
const CAP_USD: f64 = 0.01;

#[cfg(feature = "budget")]
#[tokio::main]
async fn main() {
    let model = OpenAIModel::Gpt4oMini.to_string();
    let price = PriceTable::builtin()
        .require(&model)
        .expect("no price for the model")
        .clone();

    let llm = OpenAI::default().with_model(&model);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(CommandExecutor::default())])
        .build(llm)
        .unwrap();

    // Called after each model call with the run's total usage; returning false stops the
    // run before its next tool call or model call.
    let executor = AgentExecutor::from_agent(agent).with_usage_limit(move |total: &TokenUsage| {
        let usage = Usage {
            input_tokens: total.prompt_tokens.into(),
            cached_input_tokens: total.cache_read_tokens.into(),
            cache_write_tokens: total.cache_creation_tokens.into(),
            output_tokens: total.completion_tokens.into(),
        };
        let spent = price.cost_usage(&usage);
        println!("spent ${spent:.6} of ${CAP_USD}");
        spent < CAP_USD
    });

    match executor
        .invoke(prompt_args! {
            "input" => "What is the name of the current directory, and how many files are in it?",
        })
        .await
    {
        Ok(answer) => println!("Answer: {answer}"),
        Err(e) => println!("Stopped: {e}"),
    }
}

#[cfg(not(feature = "budget"))]
fn main() {
    println!("This example requires the 'budget' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example agent_budget --features=budget");
}
