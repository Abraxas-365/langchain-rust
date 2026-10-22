// A constitutional chain checks an answer against principles and revises it when a
// critique finds a problem.
//
// OpenAI:            OPENAI_API_KEY=... cargo run --example constitutional_chain
// Any local model:   CONSTITUTIONAL_OLLAMA_MODEL=qwen3:14b cargo run --example constitutional_chain
//                    (talks to Ollama's OpenAI-compatible endpoint on localhost:11434)

use langchain_rust::{
    chain::{ConstitutionalChain, ConstitutionalPrinciple, LLMChainBuilder},
    llm::{openai::OpenAI, OpenAIConfig},
    prompt_args, template_jinja2,
};

#[tokio::main]
async fn main() {
    let llm = match std::env::var("CONSTITUTIONAL_OLLAMA_MODEL") {
        Ok(model) => OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model),
        Err(_) => OpenAI::default(),
    };

    // A deliberately bad assistant, so the principles have something to fix.
    let snarky = LLMChainBuilder::new()
        .prompt(template_jinja2!(
            "Answer in one short sentence, and mock the person for asking: {{question}}",
            "question"
        ))
        .llm(llm.clone())
        .build()
        .unwrap();

    let principles = vec![
        // One of the 49 built-in principles (see `ConstitutionalPrinciple::builtin_names`).
        ConstitutionalPrinciple::get("thoughtful").unwrap(),
        // Or your own.
        ConstitutionalPrinciple::new(
            "Point out any factual errors in the response.",
            "Rewrite the response to fix the factual errors, keeping everything else.",
        )
        .with_name("factual"),
    ];

    let chain = ConstitutionalChain::new(snarky, llm, principles);
    let out = chain
        .critique_and_revise(prompt_args! { "question" => "How many legs does a spider have?" })
        .await
        .unwrap();

    println!("initial answer: {}\n", out.initial_output);
    for step in &out.critiques_and_revisions {
        println!("[{}] critique: {}", step.principle, step.critique);
        match &step.revision {
            Some(revision) => println!("[{}] revision: {revision}\n", step.principle),
            None => println!("[{}] no revision\n", step.principle),
        }
    }
    println!("final answer: {}", out.output);
}
