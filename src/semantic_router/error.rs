use thiserror::Error;

use crate::embedding::EmbedderError;

#[derive(Error, Debug)]
pub enum RouterBuilderError {
    #[error("Invalid Router configuration: at least one of utterances or embedding must be provided, and utterances cannot be an empty vector.")]
    InvalidConfiguration,
}

#[derive(Error, Debug)]
pub enum RouteLayerBuilderError {
    #[error("All routers must have an embedding if the route layer lacks a global embedder.")]
    MissingEmbedderForRoutes,

    #[error("Route layer error: {0}")]
    RouteLayerError(#[from] RouteLayerError),
}

#[derive(Error, Debug)]
pub enum RouteLayerError {
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbedderError),
}
