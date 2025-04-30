use async_openai::error::OpenAIError;
#[cfg(feature = "ollama")]
use ollama_rs::error::OllamaError;
use reqwest::Error as ReqwestError;
use serde_json::Error as SerdeJsonError;
use thiserror::Error;
use tokio::time::error::Elapsed;

use crate::llm::{AnthropicError, DeepseekError, QwenError};

#[derive(Error, Debug)]
pub enum LLMError {
    #[error("OpenAI error: {0}")]
    OpenAIError(#[from] OpenAIError),

    #[error("Anthropic error: {0}")]
    AnthropicError(#[from] AnthropicError),

    #[error("Qwen error: {0}")]
    QwenError(#[from] QwenError),

    #[error("Deepseek error: {0}")]
    DeepseekError(#[from] DeepseekError),

    #[cfg(feature = "ollama")]
    #[error("Ollama error: {0}")]
    OllamaError(#[from] OllamaError),

    #[error("Network request failed: {0}")]
    RequestError(#[from] ReqwestError),

    #[error("JSON serialization/deserialization error: {0}")]
    SerdeError(#[from] SerdeJsonError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Operation timed out")]
    Timeout(#[from] Elapsed),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Content not found in response: Expected at {0}")]
    ContentNotFound(String),

    #[error("Parsing error: {0}")]
    ParsingError(String),

    #[error("Error: {0}")]
    OtherError(String),
}
