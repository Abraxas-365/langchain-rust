// An LLM picks which chain answers each question.
//
// OpenAI:            OPENAI_API_KEY=... cargo run --example router_chain
// Any local model:   ROUTER_OLLAMA_MODEL=qwen3:14b cargo run --example router_chain
//                    (talks to Ollama's OpenAI-compatible endpoint on localhost:11434)

use langchain_rust::{
    chain::{Chain, LLMChain, LLMChainBuilder, RouterChainBuilder},
    llm::{openai::OpenAI, OpenAIConfig},
    prompt::HumanMessagePromptTemplate,
    prompt_args, template_jinja2,
};

fn expert(llm: &OpenAI<OpenAIConfig>, persona: &str) -> LLMChain {
    LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_jinja2!(
            format!("{persona} Answer in one or two sentences.\n\nQuestion: {{{{input}}}}"),
            "input"
        )))
        .llm(llm.clone())
        .build()
        .unwrap()
}

#[tokio::main]
async fn main() {
    let llm = match std::env::var("ROUTER_OLLAMA_MODEL") {
        Ok(model) => OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model),
        Err(_) => OpenAI::default(),
    };

    let router = RouterChainBuilder::new()
        .llm(llm.clone())
        .destination(
            "physics",
            "Questions about physics: forces, energy, light, the universe",
            expert(&llm, "You are a physics professor."),
        )
        .destination(
            "cooking",
            "Questions about recipes, ingredients and cooking techniques",
            expert(&llm, "You are a chef."),
        )
        .default_chain(expert(&llm, "You are a helpful assistant."))
        .build()
        .unwrap();

    for question in [
        "Why is the sky blue?",
        "How long should I rest a steak after grilling it?",
        "What's the capital of Peru?",
    ] {
        let output = router
            .execute(prompt_args! { "input" => question })
            .await
            .unwrap();
        println!(
            "[{}] {question}\n{}\n",
            output["destination"].as_str().unwrap(),
            output["output"].as_str().unwrap().trim()
        );
    }
}
