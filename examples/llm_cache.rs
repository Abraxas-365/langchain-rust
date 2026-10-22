// Response caching: wrap any LLM in `CachedLLM` and repeated requests are answered from
// the cache instead of calling the model again (no latency, no tokens).
//
// OpenAI:            OPENAI_API_KEY=... cargo run --example llm_cache
// Any local model:   CACHE_OLLAMA_MODEL=qwen3:14b cargo run --example llm_cache
//                    (talks to Ollama's OpenAI-compatible endpoint on localhost:11434)

use std::{sync::Arc, time::Duration, time::Instant};

use langchain_rust::{
    language_models::llm::LLM,
    llm::{openai::OpenAI, CachedLLM, InMemoryLLMCache, OpenAIConfig},
};

#[tokio::main]
async fn main() {
    let (llm, model) = match std::env::var("CACHE_OLLAMA_MODEL") {
        Ok(model) => (
            OpenAI::default()
                .with_config(
                    OpenAIConfig::default()
                        .with_api_base("http://localhost:11434/v1")
                        .with_api_key("ollama"),
                )
                .with_model(model.clone()),
            model,
        ),
        Err(_) => (OpenAI::default(), "openai-default".to_string()),
    };

    // Keep up to 1000 answers for an hour. Implement `LLMCache` to use Redis, a DB, ...
    let cache = Arc::new(
        InMemoryLLMCache::new()
            .with_capacity(1000)
            .with_ttl(Duration::from_secs(3600)),
    );
    // The namespace keeps different models apart when they share a cache.
    let llm = CachedLLM::new(llm, cache.clone()).with_namespace(model);

    for attempt in 1..=2 {
        let start = Instant::now();
        let answer = llm
            .invoke("In one sentence, why is the sky blue?")
            .await
            .unwrap();
        println!("#{attempt} ({:?}): {answer}", start.elapsed());
    }
    println!("cached answers: {}", cache.len());
}
