//! Transcribe audio with AssemblyAI and load it as documents, one per paragraph.
//!
//! `ASSEMBLYAI_API_KEY=... cargo run --example assemblyai_loader [audio URL or local file]`

use std::error::Error;

use futures_util::StreamExt;
use langchain_rust::document_loaders::{AssemblyAILoader, AudioSource, Loader, TranscriptFormat};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let input = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "https://assembly.ai/wildfires.mp3".to_string());
    let source = if input.starts_with("http://") || input.starts_with("https://") {
        AudioSource::url(input)
    } else {
        // Local files are uploaded first.
        AudioSource::file(input)
    };

    let mut docs = AssemblyAILoader::new(source)
        .with_format(TranscriptFormat::Paragraphs)
        .with_param("speaker_labels", json!(true))
        .load()
        .await?;
    while let Some(doc) = docs.next().await {
        let doc = doc?;
        println!(
            "[{} ms - {} ms] {}\n",
            doc.metadata["start"], doc.metadata["end"], doc.page_content
        );
    }
    Ok(())
}
