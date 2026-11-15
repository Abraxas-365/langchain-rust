// Prompt regression diff: ask the same question under two prompt variants
// and measure exactly what changed in the answers with llm-diff — word-level
// ops, a similarity ratio, and a structural JSON diff when both answers parse
// as JSON. This is the loop you want before shipping a prompt change: not
// "the answer feels different", but "0.87 similar, these three facts moved".
//
// OpenAI:            OPENAI_API_KEY=... cargo run --example prompt_diff
// Any local model:   PROMPT_DIFF_OLLAMA_MODEL=qwen3:14b cargo run --example prompt_diff
//                    (talks to Ollama's OpenAI-compatible endpoint on localhost:11434)

use langchain_rust::{
    language_models::llm::LLM,
    llm::{openai::OpenAI, OpenAIConfig},
};
use llm_diff::{json_diff, TextDiff};

const QUESTION: &str = "What is the population of Paris, and what is it best known for?";

// Baseline vs. candidate: the change under test is the added style constraint.
const PROMPT_A: &str = "Answer in one short paragraph.";
const PROMPT_B: &str = "Answer in one short paragraph of plain text. Do not use lists.";

#[tokio::main]
async fn main() {
    let llm = match std::env::var("PROMPT_DIFF_OLLAMA_MODEL") {
        Ok(model) => OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model),
        Err(_) => OpenAI::default(),
    };

    let answer_a = llm
        .invoke(&format!("{PROMPT_A}\n\n{QUESTION}"))
        .await
        .expect("baseline call failed");
    let answer_b = llm
        .invoke(&format!("{PROMPT_B}\n\n{QUESTION}"))
        .await
        .expect("candidate call failed");

    let diff = TextDiff::compute_words(&answer_a, &answer_b);
    println!("similarity: {:.0}%", diff.ratio * 100.0);
    if diff.is_identical() {
        println!("answers are identical — the prompt change is a no-op");
        return;
    }
    println!("changed words:");
    for op in diff.ops.iter().filter(|o| o.kind() != "=") {
        println!("  {}{}", op.kind(), op.text());
    }

    // When both answers are JSON, a text diff overstates the change (one moved
    // element marks the whole array changed); the structural diff reports the
    // one JSON Pointer that actually moved.
    if let Ok(ops) = json_diff(&answer_a, &answer_b) {
        println!("structural changes: {}", ops.len());
        for op in &ops {
            println!("  {op:?}");
        }
    }
}
