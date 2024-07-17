mod error;

pub mod embedder_trait;
pub use embedder_trait::*;

#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "ollama")]
pub use ollama::*;

pub mod openai;
pub use error::*;

#[cfg(feature = "fastembed")]
mod fastembed;
#[cfg(feature = "fastembed")]
pub use fastembed::*;

#[cfg(feature = "mistralai")]
pub mod mistralai;
#[cfg(feature = "mistralai")]
pub use mistralai::*;
