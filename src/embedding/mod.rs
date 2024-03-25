pub mod embedder_trait;
pub use embedder_trait::*;
mod error;
pub mod ollama;
pub mod openai;
pub use error::*;
