use thiserror::Error;

use crate::{language_models::LLMError, output_parsers::OutputParserError, template::TemplateError};

#[derive(Error, Debug)]
pub enum ChainError {
    #[error("LLM error: {0}")]
    LLMError(#[from] LLMError),

    #[error("Retriever error: {0}")]
    RetrieverError(String),

    #[error("OutputParser error: {0}")]
    OutputParser(#[from] OutputParserError),

    #[error("Prompt error: {0}")]
    PromptError(#[from] TemplateError),

    #[error("Missing Object On Builder: {0}")]
    MissingObject(String),

    #[error("Missing input variable: {0}")]
    MissingInputVariable(String),

    #[error("Serde json error: {0}")]
    SerdeJsonError(#[from] serde_json::Error),

    #[error("Incorrect input variable: expected type {expected_type}, {source}")]
    IncorrectInputVariable {
        source: serde_json::Error,
        expected_type: String,
    },

    #[error("Error: {0}")]
    OtherError(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Agent error: {0}")]
    AgentError(String),
}
