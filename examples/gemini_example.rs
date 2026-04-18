use futures::StreamExt;
use langchain_rust::{
    embedding::Embedder,
    language_models::llm::LLM,
    llm::{Gemini, GeminiEmbedder},
    schemas::Message,
};
use std::io::{self, Write};

#[tokio::main]
async fn main() {
    // Requires GEMINI_API_KEY environment variable to be set
    let llm = Gemini::default();

    println!("--- Basic Chat ---");
    match llm.invoke("What is the capital of France?").await {
        Ok(res) => println!("Response: {}", res),
        Err(e) => eprintln!("Error: {:?}", e),
    }

    println!("\n--- Streaming Chat ---");
    let messages = vec![Message::new_human_message(
        "Tell me a short story about a brave knight.",
    )];
    match llm.stream(&messages).await {
        Ok(mut stream) => {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(data) => {
                        print!("{}", data.content);
                        io::stdout().flush().unwrap();
                    }
                    Err(e) => {
                        eprintln!("\nStream Error: {:?}", e);
                    }
                }
            }
            println!();
        }
        Err(e) => eprintln!("Error: {:?}", e),
    }

    println!("\n--- Embeddings ---");
    let embedder = GeminiEmbedder::default();
    let vec: Vec<f64> = embedder
        .embed_query("This is a test sentence.")
        .await
        .unwrap();
    println!("Embedded vector length: {}", vec.len());
}
