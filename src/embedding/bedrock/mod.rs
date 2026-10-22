//! Amazon Bedrock embeddings: Amazon Titan and Cohere Embed models through `InvokeModel`.
//!
//! ```rust,no_run
//! use langchain_rust::embedding::{bedrock::{BedrockEmbedder, BedrockEmbeddingModel}, Embedder};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let embedder = BedrockEmbedder::new()
//!     .with_model(BedrockEmbeddingModel::TitanEmbedTextV2)
//!     .with_dimensions(512);
//! let vector = embedder.embed_query("What is langchain-rust?").await?;
//! # Ok(()) }
//! ```
//!
//! Credentials and region come from the AWS SDK's default chain, like
//! [`Bedrock`](crate::llm::bedrock::Bedrock).

mod bedrock_embedder;
#[cfg(test)]
mod tests;

pub use bedrock_embedder::*;
