mod fastembed;
pub use fastembed::*;

extern crate fastembed as ext_fastembed;
#[allow(deprecated)]
pub use ext_fastembed::InitOptions;
pub use ext_fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
