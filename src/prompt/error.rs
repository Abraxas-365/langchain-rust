use serde_json::Error as SerdeJsonError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PromptError {
    #[error("Variable {0} is missing from input variables")]
    MissingVariable(String),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] SerdeJsonError),
}
