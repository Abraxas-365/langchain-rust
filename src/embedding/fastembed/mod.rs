#[allow(clippy::module_inception)]
mod fastembed;
pub use fastembed::*;

extern crate fastembed as ext_fastembed;
pub use ext_fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
