// A TransformChain cleans the input before an LLM chain sees it.
//
// OpenAI:            OPENAI_API_KEY=... cargo run --example transform_chain
// Any local model:   TRANSFORM_OLLAMA_MODEL=qwen3:14b cargo run --example transform_chain
//                    (talks to Ollama's OpenAI-compatible endpoint on localhost:11434)

use std::collections::HashMap;

use langchain_rust::{
    chain::{Chain, LLMChainBuilder, TransformChain},
    llm::{openai::OpenAI, OpenAIConfig},
    prompt::HumanMessagePromptTemplate,
    prompt_args, sequential_chain, template_fstring,
};
use serde_json::json;

#[tokio::main]
async fn main() {
    let llm = match std::env::var("TRANSFORM_OLLAMA_MODEL") {
        Ok(model) => OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model),
        Err(_) => OpenAI::default(),
    };

    // Keep only the first paragraph of a long text, so the prompt stays small.
    let first_paragraph = TransformChain::new(["text"], ["paragraph"], |inputs| async move {
        let text = inputs["text"].as_str().unwrap_or_default();
        let paragraph = text.split("\n\n").next().unwrap_or_default().trim();
        Ok(HashMap::from([("paragraph".to_string(), json!(paragraph))]))
    });

    let summarize = LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_fstring!(
            "Summarize in one short sentence:\n\n{paragraph}",
            "paragraph"
        )))
        .llm(llm)
        .output_key("summary")
        .build()
        .unwrap();

    let chain = sequential_chain!(first_paragraph, summarize);
    let text = "Rust is a systems programming language. Its ownership model gives memory safety without a garbage collector, and it compiles to fast native code.\n\nThe rest of this document covers installation, tooling and the package registry in detail.";
    let output = chain.execute(prompt_args! {"text" => text}).await.unwrap();
    println!("paragraph: {}", output["paragraph"]);
    println!("summary: {}", output["summary"]);
}
