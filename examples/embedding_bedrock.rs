//! Amazon Bedrock embeddings (Titan Text Embeddings V2, or Cohere Embed).
//!
//! Requires AWS credentials (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` or a
//! configured profile) and model access enabled in the Bedrock console.
//!
//! Run with: cargo run --features bedrock --example embedding_bedrock

#[cfg(feature = "bedrock")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use langchain_rust::embedding::{
        bedrock::{BedrockEmbedder, BedrockEmbeddingModel},
        Embedder,
    };

    let titan = BedrockEmbedder::new()
        .with_region("us-east-1")
        .with_model(BedrockEmbeddingModel::TitanEmbedTextV2)
        .with_dimensions(512);
    let vectors = titan
        .embed_documents(&["Hello world".to_string(), "Hola mundo".to_string()])
        .await?;
    println!(
        "titan: {} vectors of {} dims",
        vectors.len(),
        vectors[0].len()
    );

    // Cohere embeds documents and queries differently (input_type), which the
    // embedder handles for you.
    let cohere = BedrockEmbedder::new()
        .with_region("us-east-1")
        .with_model(BedrockEmbeddingModel::CohereEmbedMultilingualV3);
    let query = cohere.embed_query("¿Qué es langchain-rust?").await?;
    println!("cohere query: {} dims", query.len());
    Ok(())
}

#[cfg(not(feature = "bedrock"))]
fn main() {
    println!("Run with: cargo run --features bedrock --example embedding_bedrock");
}
