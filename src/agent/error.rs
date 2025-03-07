use thiserror::Error;

use crate::{chain::ChainError, language_models::LLMError, prompt::PromptError};

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("LLM error: {0}")]
    LLMError(#[from] LLMError),

    #[error("Chain error: {0}")]
    ChainError(#[from] ChainError),

    #[error("Prompt error: {0}")]
    PromptError(#[from] PromptError),

    #[error("Tool error: {0}")]
    ToolError(String),

    #[error("Missing Object On Builder: {0}")]
    MissingObject(String),

    #[error("Missing input variable: {0}")]
    MissingInputVariable(String),

    #[error("Serde json error: {0}")]
    SerdeJsonError(#[from] serde_json::Error),

    #[error("Error: {0}")]
    OtherError(String),

    #[error("Invalid format, remember the instructions regarding the format and try again")]
    InvalidFormatError,
}
