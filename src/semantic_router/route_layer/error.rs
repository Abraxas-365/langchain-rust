use thiserror::Error;

use crate::{
    chain::ChainError, embedding::EmbedderError, language_models::LLMError,
    semantic_router::IndexError,
};
use serde_json::Error as SerdeJsonError;

#[derive(Error, Debug)]
pub enum RouterBuilderError {
    #[error("Invalid Router configuration: at least one of utterances or embedding must be provided, and utterances cannot be an empty vector.")]
    InvalidConfiguration,
}

#[derive(Error, Debug)]
pub enum RouteLayerBuilderError {
    #[error("Route layer should have an embedder")]
    MissingEmbedder,

    #[error("Route layer should have an LLM")]
    MissingLLM,

    #[error("Missing Index")]
    MissingIndex,

    #[error("Route layer error: {0}")]
    RouteLayerError(#[from] RouteLayerError),

    #[error("Index error: {0}")]
    IndexError(#[from] IndexError),

    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbedderError),

    #[error("Chain error: {0}")]
    ChainError(#[from] ChainError),
}

#[derive(Error, Debug)]
pub enum RouteLayerError {
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbedderError),

    #[error("Index error: {0}")]
    IndexError(#[from] IndexError),

    #[error("LLM error: {0}")]
    LLMError(#[from] LLMError),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] SerdeJsonError),

    #[error("Chain error: {0}")]
    ChainError(#[from] ChainError),
}
