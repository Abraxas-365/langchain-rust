mod error;

pub mod embedder_trait;
pub use embedder_trait::*;

#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "ollama")]
pub use ollama::*;

pub mod openai;
pub use error::*;

mod fastembed;
pub use fastembed::*;
