use thiserror::Error;

#[derive(Error, Debug)]
pub enum IndexError {
    #[error("No Emedding on Route: {0}")]
    MissingEmbedding(String),

    #[error("Error: {0}")]
    OtherError(String),
}
