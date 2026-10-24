//! Reasoning ("thinking") with a local model: the reasoning streams separately from the
//! answer. The same `ThinkingConfig` works with Claude (`thinking` / effort), OpenAI
//! (`reasoning_effort`) and DeepSeek reasoner (always on).
//!
//! ```sh
//! ollama pull qwen3
//! cargo run --example llm_thinking --features=ollama
//! ```
#[cfg(feature = "ollama")]
use {
    futures::StreamExt,
    langchain_rust::{
        language_models::{llm::LLM, options::CallOptions, LLMEvent, ThinkingConfig},
        llm::ollama::client::Ollama,
        schemas::Message,
    },
    std::io::Write,
};

#[cfg(feature = "ollama")]
#[tokio::main]
async fn main() {
    let mut llm = Ollama::default().with_model("qwen3");
    llm.add_options(CallOptions::new().with_thinking(ThinkingConfig::enabled()));

    let question = [Message::new_human_message(
        "Which is larger, 9.11 or 9.8? Answer in one sentence.",
    )];
    let mut events = llm.generate_stream(&question);
    let mut answering = false;
    while let Some(event) = events.next().await {
        match event.unwrap() {
            LLMEvent::Thinking(chunk) => print!("\x1b[2m{chunk}\x1b[0m"),
            LLMEvent::Text(chunk) => {
                if !answering {
                    answering = true;
                    println!("\n\nAnswer:");
                }
                print!("{chunk}");
            }
            LLMEvent::Done(result) => {
                println!("\n\n({} chars of thinking)", result.thinking_text().len())
            }
            _ => {}
        }
        std::io::stdout().flush().unwrap();
    }
}

#[cfg(not(feature = "ollama"))]
fn main() {
    println!("This example requires the 'ollama' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example llm_thinking --features=ollama");
}
