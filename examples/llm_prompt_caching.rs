//! Anthropic prompt caching: the second call reads the long system prompt from the cache
//! and the usage shows it. OpenAI, DeepSeek and Qwen cache automatically and report the
//! cached part in `TokenUsage::cache_read_tokens` too.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run --example llm_prompt_caching
//! ```
use langchain_rust::{
    language_models::{llm::LLM, options::CallOptions},
    llm::Claude,
    schemas::{CacheControl, Message},
};

#[tokio::main]
async fn main() {
    // Automatic mode: the breakpoint goes on the last cacheable block and moves forward
    // with the conversation. For a fixed breakpoint use `Message::with_cache_control`.
    let claude =
        Claude::new().with_options(CallOptions::new().with_prompt_caching(CacheControl::Ephemeral));

    // Caching needs a prefix of at least ~1-4k tokens, depending on the model.
    let manual = "The quick brown fox jumps over the lazy dog. ".repeat(600);
    for question in ["How many times is 'fox' repeated?", "And 'dog'?"] {
        let messages = [
            Message::new_system_message(format!("Answer about this text:\n{manual}")),
            Message::new_human_message(question),
        ];
        let result = claude.generate(&messages).await.unwrap();
        let tokens = result.tokens.unwrap_or_default();
        println!(
            "{question} -> {}\n  prompt {} (cache read {}, cache write {})",
            result.generation.trim(),
            tokens.prompt_tokens,
            tokens.cache_read_tokens,
            tokens.cache_creation_tokens
        );
    }
}
