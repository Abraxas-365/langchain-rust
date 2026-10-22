//! [Voyage AI](https://www.voyageai.com/) embeddings (`voyage-3.5`, `voyage-code-3`, ...).
//!
//! ```rust,no_run
//! use langchain_rust::embedding::{voyage::VoyageEmbedder, Embedder};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Reads VOYAGE_API_KEY.
//! let embedder = VoyageEmbedder::new().with_model("voyage-3.5-lite");
//! let vector = embedder.embed_query("What is langchain-rust?").await?;
//! # Ok(()) }
//! ```

#[cfg(test)]
mod tests;
mod voyage_embedder;

pub use voyage_embedder::*;
