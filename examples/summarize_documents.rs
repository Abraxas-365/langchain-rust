//! Summarize and answer questions over documents too large for a single prompt.
//!
//! - `MapReduceDocuments`: one LLM call per document (concurrently), then one call to combine.
//! - `RefineDocuments`: builds the answer document by document, sequentially.
//! - `MapRerankDocuments`: answers from each document separately, keeps the best-scored one.
//! - `RetrievalQA`: retrieves relevant documents, then answers with any of the above.
//!
//! ```sh
//! OPENAI_API_KEY=... cargo run --example summarize_documents
//! ```
use std::error::Error;

use async_trait::async_trait;
use futures::StreamExt;
use langchain_rust::{
    chain::{
        Chain, MapReduceDocuments, MapRerankDocuments, RefineDocuments, RetrievalQA,
        MAP_RERANK_SCORE_KEY,
    },
    language_models::LLMEvent,
    llm::openai::OpenAI,
    prompt_args,
    schemas::{Document, Retriever},
};

fn chapters() -> Vec<Document> {
    [
        "Chapter 1. Ana opens a small bakery in Lima. Her sourdough sells out every morning.",
        "Chapter 2. A supermarket chain opens next door and her sales drop by half.",
        "Chapter 3. Ana starts teaching weekend baking classes; they become more profitable than the bread.",
    ]
    .into_iter()
    .map(Document::new)
    .collect()
}

/// A stand-in retriever; use a vector store's retriever in real code.
struct Chapters;

#[async_trait]
impl Retriever for Chapters {
    async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>, Box<dyn Error>> {
        Ok(chapters())
    }
}

#[tokio::main]
async fn main() {
    let llm = OpenAI::default();

    let summary = MapReduceDocuments::load_summarize(llm.clone())
        .invoke(prompt_args! { "input_documents" => chapters() })
        .await
        .unwrap();
    println!("map-reduce summary:\n{summary}\n");

    // Refine streams the last refinement as it's written.
    let refine = RefineDocuments::load_summarize(llm.clone());
    let mut events = refine.call_stream(prompt_args! { "input_documents" => chapters() });
    println!("refine summary:");
    while let Some(event) = events.next().await {
        match event.unwrap() {
            LLMEvent::Text(chunk) => print!("{chunk}"),
            LLMEvent::Done(result) => println!("\n(tokens: {:?})\n", result.tokens),
            _ => {}
        }
    }

    let qa = RetrievalQA::new(Chapters, MapReduceDocuments::load_qa(llm.clone()));
    let answer = qa
        .invoke(prompt_args! { "query" => "What made Ana's business profitable again?" })
        .await
        .unwrap();
    println!("answer: {answer}");

    // Map-rerank answers from each chapter on its own and keeps the best-scored answer.
    let rerank = MapRerankDocuments::load_qa(llm);
    let out = rerank
        .execute(prompt_args! {
            "input_documents" => chapters(),
            "question" => "Why did Ana's sales drop?",
        })
        .await
        .unwrap();
    println!(
        "map-rerank: {} (score {})",
        out["output"], out[MAP_RERANK_SCORE_KEY]
    );
}
