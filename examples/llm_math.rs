// LLM math: the model turns a word problem into an expression, and the built-in calculator
// computes it, so the arithmetic is exact. Nothing the model writes runs as code.
//
// OpenAI:            OPENAI_API_KEY=... cargo run --example llm_math
// Any local model:   LLM_MATH_OLLAMA_MODEL=qwen3:14b cargo run --example llm_math
//                    (talks to Ollama's OpenAI-compatible endpoint on localhost:11434)

use langchain_rust::{
    chain::LLMMathChain,
    llm::{openai::OpenAI, OpenAIConfig},
    prompt_args,
};

#[tokio::main]
async fn main() {
    let llm = match std::env::var("LLM_MATH_OLLAMA_MODEL") {
        Ok(model) => OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model),
        Err(_) => OpenAI::default(),
    };
    let chain = LLMMathChain::new(llm);

    for question in [
        "What is forty plus three, multiplied by ten thousand divided by 7324.3?",
        "If I invest 1500 at 4% a year compounded yearly, how much do I have after 10 years?",
        "What is the hypotenuse of a right triangle with legs 7 and 24?",
    ] {
        let answer = chain
            .solve(prompt_args! { "question" => question })
            .await
            .unwrap();
        println!("{question}");
        println!(
            "  {} = {}\n",
            answer.expression.unwrap_or_default(),
            answer.answer
        );
    }
}
